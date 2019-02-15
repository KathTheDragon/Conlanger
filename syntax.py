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

=== Style ===
'''

import os
from math import floor
from PIL import Image, ImageDraw, ImageFont
from .core import split, FormatError

# == Constants == #

POINT_SIZE = 16
FONT = ImageFont.truetype('calibri.ttf', POINT_SIZE)
PADDING = (POINT_SIZE, floor(POINT_SIZE*1.5))  # Will need calibrating

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# == Classes == #

class Tree(list):
    __slots__ = ('label', 'padding')
    
    def __init__(self, tree, mode, padding=PADDING):
        if tree[0] == '[' and tree[-1] == ']':  # Tree
            tree = tree[1:-1]
            tree = split(tree, nesting=(0,'[',']'), minimal=True)
            if mode == 'cons':  # Constituency trees take labelled lists
                self.label = tree[0]
                tree = tree[1:]
            if mode == 'dep':  # Dependency trees take unlabelled lists
                ix = None
                for i, branch in enumerate(tree):
                    if branch[0] != '[' and branch[-1] != ']':  # Leaf
                        if ix is None:  # First leaf
                            ix = i
                            self.label = branch
                        else:  # More than one leaf is an error
                            raise FormatError('malformatted dependency tree')
            super().__init__(Tree(branch, mode, padding) for branch in tree)
        elif tree[0] != '[' and tree[-1] != ']':  # Leaf
            self.label = tree
            super().__init__([])
        self.padding = padding
    
    def __str__(self):
        branches = ' '.join(str(branch) for branch in self)
        if branches:
            return f'[{self.label} {branches}]'
        else:
            return f'[{self.label}]'
    
    def __repr__(self):
        return f'Tree("{self}")'
    
    ## Tree geometry ##
    
    @property
    def isleaf(self):
        return len(self) == 0
    
    @property
    def depth(self):
        if self.isleaf:
            return 0
        else:
            return max(branch.depth for branch in self) + 1
        
    @property
    def isdependency(self):
        if sum(branch.isleaf for branch in self) != 1:
            return False
        return all((branch.isdependency if not branch.isleaf else branch.label == self.label) for branch in self)
    
    ## Tree layout ##
    
    @property
    def gapwidth(self):
        return self.padding[0]
    
    @property
    def layerheight(self):
        return self.padding[1] + POINT_SIZE
    
    ## Label properties ##
        
    @property
    def labelwidth(self):
        return FONT.getsize(self.label)[0]
    
    @property
    def labelleft(self):
        if self.isdependency:  # Dependency nodes go above their projections
            left = 0
            for branch in self:
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
                return floor((self[0].labelmiddle+self[-1].labelmiddle+self.pixelwidth-self[-1].pixelwidth)/2)
    
    ## Tree size ##
    
    @property
    def pixelwidth(self):
        return max(self.labelwidth, self.gapwidth*(len(self)-1)+sum(branch.pixelwidth for branch in self))
    
    @property
    def pixelheight(self):
        return POINT_SIZE + self.depth * self.layerheight
    
    @property
    def size(self):
        return (self.pixelwidth+self.padding[0]*2, self.pixelheight+self.padding[1]*2)
    
    ## Methods ##
    
    def draw(self, draw, topleft=None, depth=None):  # topleft = (h, v)
        if topleft is None:
            topleft = self.padding
        if depth is None:
            depth = self.depth
        if self.isleaf:
            labelcolour = 'red'
        else:
            labelcolour = 'blue'
        draw.text((topleft[0]+self.labelleft, topleft[1]), self.label, labelcolour, FONT)
        linetop = (topleft[0]+self.labelmiddle, topleft[1]+POINT_SIZE+1)
        treeleft = topleft[0]
        for branch in self:
            treetop = topleft[1]+self.layerheight
            colour = 'black'
            if self.isdependency and branch.isleaf:
                treetop = topleft[1]+self.layerheight*depth
                colour = 'grey'
            draw.line([linetop, (treeleft+branch.labelmiddle, treetop-1)], colour, 1)
            branch.draw(draw, (treeleft, treetop), depth-1)
            treeleft += branch.pixelwidth + self.gapwidth

# == Functions == #

def displaytree(tree, name, mode, padding=PADDING):
    tree = Tree(tree, mode, padding)
    im = Image.new('RGB', tree.size, 'white')
    draw = ImageDraw.Draw(im)
    tree.draw(draw)
    im.save(f'trees/{name}.png')
