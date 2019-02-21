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
    def __init__(self, parent, tree, leaves):
        self.parent = parent
        if tree.startswith('['):  # Tree
            tree = tree[1:-1]
        tree = split(tree, nesting=(0,'([',')]'), minimal=True)
        self.label = tree[0]
        self.children = [self.__class__(self, child, leaves) for child in tree[1:]]
        self.leaves = leaves
        # Extra processing now to save work later
        if self.label.startswith('('):
            self.label = self.label[1:-1]
            if ' ' not in self.label:  # Index
                if not self.isleaf:
                    raise TreeFormatError('indices can only be used with leaves')
                self.label = int(self.label)
        if self.isleaf and isinstance(self.label, str):
            if self.leaves.count(self.label) != 1:
                raise TreeFormatError(f'explicit leaf labels must be unique and present in `leaves`: {self.label}')
            self.label = self.leaves.index(self.label)
    
    def __str__(self):
        children = ' '.join(str(child) for child in self.children)
        if children:
            return f'[{self.labeltext} {children}]'
        else:
            return f'[{self.labeltext}]'
    
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
    def layer(self):
        if self.isroot:
            return 0
        else:
            return self.parent.layer + 1
    
    ## Tree Size ##
    @property
    def pixelwidth(self):
        return max(self.labelwidth, GAP_WIDTH*(len(self.children)-1)+sum(child.pixelwidth for child in self.children))
    
    @property
    def pixelheight(self):
        return POINT_SIZE + self.depth * LAYER_HEIGHT
    
    @property
    def imsize(self):
        return (self.pixelwidth + PADDING*2, self.pixelheight + PADDING*2)
    
    ## Label properties ##
    @property
    def labeltext(self):
        if self.isleaf:
            return self.leaves[self.label]
        else:
            return self.label
    
    @property
    def labelwidth(self):
        return textwidth(self.labeltext)
    
    # Revise this
    @property
    def labelleft(self):
        if self.isleaf:
            return PADDING + self.label * GAP_WIDTH + sum(textwidth(leaf) for leaf in self.leaves[:self.label])
        else:
            return self.labelmiddle - floor(self.labelwidth/2)
    
    @property
    def labeltop(self):
        return PADDING + self.layer * LAYER_HEIGHT
    
    # Revise this
    @property
    def labelmiddle(self):
        if self.isleaf:
            return self.labelleft + floor(self.labelwidth/2)
        else:
            return floor((self.children[0].labelmiddle + self.children[-1].labelmiddle)/2)
    
    ## Methods ##
    def draw(self, draw):
        # Draw label
        labelcolour = 'red' if self.isleaf else 'blue'
        draw.text((self.labelleft, self.labeltop), self.labeltext, labelcolour, FONT)
        # Draw line
        if not self.isroot:
            linetop = (self.parent.labelmiddle, self.parent.labeltop + POINT_SIZE + 1)
            linebottom = (self.labelmiddle, self.labeltop - 1)
            linecolour = 'grey' if self.isleaf else 'black'
            draw.line([linetop, linebottom], linecolour, 1)
        # Draw descendents
        for child in self.children:
            child.draw(draw)

class DependencyTree(Tree):
    def __init__(self, *args):
        super().__init__(*args)
        if sum(child.isleaf for child in self.children) > 1:
            raise TreeFormatError('dependency tree nodes can have at most one child that is a leaf')
    
    @property
    def root(self):
        tree = self
        while not tree.isroot:
            tree = tree.parent
        return tree
    
    @property
    def layer(self):
        if self.isleaf:
            return self.root.depth
        else:
            return super().layer
    
    @property
    def labelmiddle(self):
        if not self.isleaf:
            for child in self.children:
                if child.isleaf:
                    return child.labelmiddle
        return super().labelmiddle

def textwidth(text):
    return FONT.getsize(text)[0]

def drawtree(tree, leaves, mode):
    if mode == 'dep':
        tree = DependencyTree(None, tree, leaves)
    else:
        tree = Tree(None, tree, leaves)
    im = Image.new('RGB', tree.imsize, 'white')
    draw = ImageDraw.Draw(im)
    tree.draw(draw)
    return im
