'''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Anti-aliasing: https://stackoverflow.com/questions/14350645/is-there-an-antialiasing-method-for-python-pil
Multi-word labels require that triangle thing

=== Features ===

=== Style ===
'''

from math import floor
from PIL import Image, ImageDraw, ImageFont
from .core import split, LangException

# == Constants == #
POINT_SIZE = 16
FONT = ImageFont.truetype('calibri.ttf', POINT_SIZE)
GAP_WIDTH = POINT_SIZE  # minimum horizontal spacing between trees
GAP_HEIGHT = floor(POINT_SIZE*1.5)  # minimum vertical spacing between layers
LAYER_HEIGHT = GAP_HEIGHT + POINT_SIZE  # Needs calibration
PADDING = POINT_SIZE  # Padding around the edge of the image

# == Exceptions == #
class TreeException(LangException):
    pass

class TreeFormatError(TreeException):
    pass

# == Classes == #
class Tree():
    __slots__ = ('parent', 'label', 'children', 'leaves')
    
    def __init__(self, parent, tree, leaves):
        self.parent = parent
        if tree.startswith('['):  # Tree
            if not tree.endswith(']'):
                raise TreeFormatError(f'unbalanced brackets: {tree}')
            tree = tree[1:-1]
        tree = split(tree, nesting=(0,'([',')]'), minimal=True)
        self.label = tree[0]
        self.children = tree[1:]
        if isinstance(leaves, str):
            leaves = split(leaves, nesting=(0,'(',')'), minimal=True)
            for i, leaf in enumerate(leaves):
                if leaf.startswith('('):
                    leaves[i] = leaf[1:-1]
        self.leaves = leaves
    
    def __len__(self):
        return len(self.children)
    
    def __getitem__(self, key):
        return self.children[key]
    
    def __iter__(self):
        return iter(self.children)
    
    def __contains__(self, item):
        return item in self.children
    
    def __str__(self):
        children = ' '.join(str(child) for child in self)
        if children:
            return f'[{self.label} {children}]'
        else:
            return f'[{self.label}]'
    
    def __repr__(self):
        return f'Tree("{self}")'
    
    ## Tree geometry ##
    @property
    def isleaf(self):
        return False
    
    @property
    def isroot(self):
        return self.parent is None
    
    @property
    def depth(self):
        return max(child.depth for child in self) + 1
    
    @property
    def layer(self):
        if self.isroot:
            return 0
        else:
            return self.parent.layer + 1
    
    ## Tree Size ##
    @property
    def pixelwidth(self):
        return max(self.labelwidth, GAP_WIDTH*(len(self)-1)+sum(child.pixelwidth for child in self))
    
    @property
    def pixelheight(self):
        return POINT_SIZE + self.depth * LAYER_HEIGHT
    
    @property
    def imsize(self):
        return (self.pixelwidth + PADDING*2, self.pixelheight + PADDING*2)
    
    @property
    def labelwidth(self):
        return textwidth(self.label)
    
    @property
    def labelleft(self):
        return self.labelmiddle - floor(self.labelwidth/2)
    
    @property
    def labeltop(self):
        return PADDING + self.layer * LAYER_HEIGHT
    
    @property
    def labelmiddle(self):
        return floor((self[0].labelmiddle + self[-1].labelmiddle)/2)
    
    ## Methods ##
    def draw(self, draw):
        # Draw label
        labelcolour = 'red' if self.isleaf else 'blue'
        draw.text((self.labelleft, self.labeltop), self.label, labelcolour, FONT)
        # Draw line
        if not self.isroot:
            linetop = (self.parent.labelmiddle, self.parent.labeltop + POINT_SIZE + 1)
            linebottom = (self.labelmiddle, self.labeltop - 1)
            linecolour = 'lightgrey' if self.isleaf else 'black'
            draw.line([linetop, linebottom], linecolour, 1)
        # Draw descendents
        for child in self:
            child.draw(draw)

class Leaf(Tree):
    __slots__ = ('leaves', 'index')
    
    def __init__(self, tree):
        self.parent = tree.parent
        self.label = tree.label
        self.children = tree.children
        self.leaves = tree.leaves
        self.index = None
        if self.label.startswith('('):
            self.label = self.label[1:-1]
            if ' ' not in self.label:  # Index
                self.index = int(self.label)
                self.label = self.leaves[self.index]
        if self.index is None:
            if self.label not in self.leaves:
                raise TreeFormatError(f'explicit leaf labels must be present in `leaves`: {self.label}')
            self.index = self.leaves.index(self.label)
        self.leaves[self.index] = self
    
    @property
    def isleaf(self):
        return True
    
    @property
    def depth(self):
        return 0
    
    @property
    def pixelwidth(self):
        return max(self.labelwidth, self.parent.labelwidth if len(self.parent) == 1 else 0)
    
    @property
    def labelmiddle(self):
        return PADDING + self.index * GAP_WIDTH + sum(leaf.pixelwidth for leaf in self.leaves[:self.index]) + floor(self.pixelwidth/2)

class DependencyTree(Tree):
    def __init__(self, *args):
        super().__init__(*args)
        for i, child in enumerate(self):
            self.children[i] = DependencyTree(self, child, self.leaves)
            if len(self[i]) == 0:
                self.children[i] = DependencyLeaf(self[i])
        if sum(child.isleaf for child in self) > 1:
            raise TreeFormatError('dependency tree nodes can have at most one child that is a leaf')
    
    @property
    def labelmiddle(self):
        for child in self:
            if child.isleaf:
                return child.labelmiddle
        else:
            return super().labelmiddle

class DependencyLeaf(Leaf):
    @property
    def layer(self):
        return max(leaf.parent.layer for leaf in self.leaves) + 1

class ConstituencyTree(Tree):
    def __init__(self, *args):
        super().__init__(*args)
        for i, child in enumerate(self):
            self.children[i] = ConstituencyTree(self, child, self.leaves)
            if len(self[i]) == 0:
                self.children[i] = ConstituencyLeaf(self[i])

class ConstituencyLeaf(Leaf):
    pass

def textwidth(text):
    return FONT.getsize(text)[0]

def drawtree(tree, leaves, mode):
    if mode == 'dep':
        tree = DependencyTree(None, tree, leaves)
    else:
        tree = ConstituencyTree(None, tree, leaves)
    im = Image.new('RGB', tree.imsize, 'white')
    draw = ImageDraw.Draw(im)
    tree.draw(draw)
    return im
