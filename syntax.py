'''
==================================== To-do ====================================
=== Bug-fixes ===
See about fixing wonky leaves in constituency trees when the parent node is wider than the leaf

=== Implementation ===
Anti-aliasing: https://stackoverflow.com/questions/14350645/is-there-an-antialiasing-method-for-python-pil

=== Features ===
Make `mode` optional in Tree()
- __init__ will try to infer the correct value
- if the tree is formatted wrongly or ambiguously, raise an error
Multi-word labels for constituency trees
Non-projecting clitics

=== Style ===
'''

from math import floor
from PIL import Image, ImageDraw, ImageFont
from .core import split, LangException

# == Constants == #

POINT_SIZE = 16
FONT = ImageFont.truetype('calibri.ttf', POINT_SIZE)
PADDING = (POINT_SIZE, floor(POINT_SIZE*1.5))  # Will need calibrating

# == Exceptions == #

class TreeException(LangException):
    pass

class TreeFormatError(TreeException):
    pass

# == Classes == #

class Tree():
    __slots__ = ('label', 'bparent', 'branches', 'parent', 'children', 'padding')
    
    def __init__(self, label='', branches=None, padding=PADDING):
        self.label = label
        self.bparent = None
        self.branches = branches or []
        self.parent = None
        self.children = []
        self.padding = padding
    
    def __str__(self):
        branches = ' '.join(str(branch) for branch in self.branches)
        if branches:
            return f'[{self.label} {branches}]'
        else:
            return f'[{self.label}]'
    
    def __repr__(self):
        return f'Tree("{self}")'
    
    ## Tree geometry ##
    
    @property
    def isleaf(self):
        return len(self.children) == 0
    
    @property
    def isroot(self):
        return self.parent is None
    
    @property
    def depth(self):
        if self.isleaf:
            return 0
        else:
            return max(child.depth for child in self.children) + 1
        
    @property
    def isdependency(self):
        if sum(child.isleaf for child in self.children) != 1:
            return False
        return all((child.isdependency if not child.isleaf else child.label == self.label) for child in self.children)
    
    @property
    def root(self):
        tree = self
        while not tree.isroot:
            tree = tree.parent
        return tree
    
    @property
    def layer(self):
        if self.isroot:
            return 0
        elif self.parent.isdependency and self.isleaf:
            return self.root.depth
        else:
            return self.parent.layer + 1
    
    @property
    def discontinuities(self):
        parentless = []
        childless = []
        for i, branch in enumerate(self.branches):
            if isinstance(branch.parent, str):  # discontinuity
                if branch.label == '':  # child placeholder
                    childless.append((self, i))
                else:  # parentless tree
                    parentless.append(branch)
            _parentless, _childless = branch.discontinuities
            parentless.extend(_parentless)
            childless.extend(_childless)
        return parentless, childless
    
    ## Tree Size ##
    
    @property
    def gapwidth(self):
        return self.padding[0]
    
    @property
    def layerheight(self):
        return self.padding[1] + POINT_SIZE
        
    @property
    def pixelwidth(self):
        return max(self.labelwidth, self.gapwidth*(len(self.branches)-1)+sum(branch.pixelwidth for branch in self.branches))
    
    @property
    def pixelheight(self):
        return POINT_SIZE + self.depth * self.layerheight
    
    @property
    def size(self):
        return (self.pixelwidth+self.padding[0]*2, self.pixelheight+self.padding[1]*2)
    
    @property
    def topleft(self):
        if self.bparent is None:
            return self.padding
        ix = self.bparent.branches.index(self)
        ptopleft = self.bparent.topleft
        left = ptopleft[0] + self.bparent.gapwidth*ix + sum(branch.pixelwidth for branch in self.bparent.branches[:ix])
        top = self.root.padding[1] + self.layerheight*self.layer
        return (left, top)
    
    ## Label properties ##
        
    @property
    def labelwidth(self):
        return FONT.getsize(self.label)[0]
    
    @property
    def labelleft(self):
        if self.isdependency:  # Dependency nodes go above their projections
            left = 0
            for branch in self.branches:
                if branch.isleaf:
                    return left
                else:
                    left += branch.pixelwidth + self.gapwidth
        else:  # Constituency nodes are centred
            return self.labelmiddle - floor(self.labelwidth/2)
    
    @property
    def labelmiddle(self):
        if self.isdependency:  # Dependency nodes go above their projections
            return self.labelleft + floor(self.labelwidth/2)
        else:  # Constituency nodes are centred
            if self.isleaf:
                return floor(self.pixelwidth/2)
            else:
                return floor((self.branches[0].labelmiddle+self.branches[-1].labelmiddle+self.pixelwidth-self.branches[-1].pixelwidth)/2)
    
    ## Methods ##
    
    def draw(self, draw):
        topleft = self.topleft
        # Draw line
        if self.parent is not None:
            ptopleft = self.parent.topleft
            linecolour = 'grey' if self.isleaf and self.parent.isdependency else 'black'
            labeltop = (topleft[0]+self.labelmiddle, topleft[1]-1)
            plabelbottom = (ptopleft[0]+self.parent.labelmiddle, ptopleft[1]+POINT_SIZE+1)
            draw.line([labeltop, plabelbottom], linecolour, 1)
        # Draw label
        labelcolour = 'red' if self.isleaf else 'blue'
        draw.text((topleft[0]+self.labelleft, topleft[1]), self.label, labelcolour, FONT)
        # Draw descendents
        for branch in self.branches:
            branch.draw(draw)

# == Functions == #

def maketree(tree, mode, padding=PADDING):
    if tree[0] == '[' and tree[-1] == ']':  # Tree
        tree = split(tree[1:-1], nesting=(0,'{[','}]'), minimal=True)
        if mode == 'cons':  # Constituency trees take labelled lists
            label = tree[0]
            tree = tree[1:]
        if mode == 'dep':  # Dependency trees take unlabelled lists
            ix = None
            for i, branch in enumerate(tree):
                if branch[0] not in '{[' and branch[-1] not in '}]':  # Leaf
                    if ix is None:  # First leaf
                        ix = i
                        label = branch
                    else:  # More than one leaf is an error
                        raise FormatError('malformatted dependency tree')
        tree = Tree(label, [maketree(branch, mode, padding) for branch in tree], padding)
        for branch in tree.branches:
            branch.bparent = tree
            if branch.parent is None:
                branch.parent = tree
                tree.children.append(branch)
        return tree
    elif tree[0] == '{' and tree[-1] == '}':  # Discontinuity
        tree = split(tree[1:-1], nesting=(0,'{[','}]'), minimal=True)
        parent = tree[0]
        if len(tree) == 2:
            tree = maketree(tree[1], mode, padding)
        else:
            tree = Tree()
        tree.parent = parent
        return tree
    elif tree[0] not in '{[' and tree[-1] not in '}]':  # Leaf
        label = tree
        return Tree(label, padding=padding)

def fixtree(tree):
    parentless, childless = tree.discontinuities
    for subtree in parentless:
        link = subtree.parent
        for parent, i in childless:
            if parent.branches[i].parent == link:
                subtree.parent = parent
                parent.children.append(subtree)
                parent.branches[i].parent = None
                break
        else:
            raise TreeFormatError('unpaired link node')
    childless = sorted(childless, key=lambda c: -c[1])
    for parent, i in childless:
        if parent.branches[i].parent != None:
            raise TreeFormatError('unpaired link node')
        del parent.branches[i]

def drawtree(tree, mode, padding=PADDING):
    tree = maketree(tree, mode, padding)
    fixtree(tree)
    im = Image.new('RGB', tree.size, 'white')
    draw = ImageDraw.Draw(im)
    tree.draw(draw)
    return im
