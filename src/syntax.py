'''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Multi-word labels require that triangle thing
Need to reimplement discontinuities somehow

=== Features ===

=== Style ===
'''

import re
from dataclasses import dataclass, field
from math import floor
from PIL import Image, ImageDraw, ImageFont
from .core import LangException, Token, CompilerError, TokenError

## Constants
SCALE = 10
POINT_SIZE = 16*SCALE
FONT = ImageFont.truetype('calibri.ttf', POINT_SIZE)
GAP_WIDTH = POINT_SIZE  # minimum horizontal spacing between trees
GAP_HEIGHT = POINT_SIZE  # minimum vertical spacing between layers
LAYER_HEIGHT = GAP_HEIGHT + POINT_SIZE
PADDING = POINT_SIZE  # Padding around the edge of the image

## Tokens
TOKENS = {
    'LBRACKET': r'\[',
    'RBRACKET': r'\]',
    'WHITESPACE': r' +',
    'QUOTED': r'\".*?\"',
    'INDEX': r'[₀-₉]+',
    'STRING': r'[^\[\]₀-₉ ]+',
    'UNKNOWN': r'.'
}
TOKEN_REGEX = re.compile('|'.join(f'(?P<{type}>{regex})' for type, regex in TOKENS.items()))

## Exceptions
class TreeException(LangException):
    pass

class TreeFormatError(TreeException):
    pass

class UnexpectedToken(TokenError):
    def __init__(self, token, expected=None):
        type = token.type.lower()
        if expected is None:
            super().__init__(f'Unexpected {type} token', token)
        else:
            super().__init__(f'Unexpected {type} token, expected {expected}', token)

## Classes
@dataclass
class Tree:
    label: str
    children: list = field(default_factory=list)

    def __len__(self):
        return len(self.children)

    def __getitem__(self, key):
        return self.children[key]

    def __iter__(self):
        yield from self.children

    def __str__(self):
        children = ' '.join(str(child) for child in self)
        return f'[{self.label} {children}]' if children else f'[{self.label}]'

    def __repr__(self):
        return f'Tree("{self}")'

    @staticmethod
    def make(string):
        tokens = list(tokenise(string))
        if tokens[0].type == 'LBRACKET' and tokens[-1].type == 'RBRACKET':
            return compileTree(tokens[1:-1])
        else:
            raise TreeFormatError('invalid syntax')

    ## Tree geometry
    @property
    def isleaf(self):
        return self.children == []

    @property
    def depth(self):
        if self.isleaf:
            return 0
        else:
            return max(child.depth for child in self) + 1

    ## Tree Size
    @property
    def childrenwidth(self):
        return max(0, GAP_WIDTH*(len(self)-1) + sum(child.width for child in self))

    @property
    def width(self):
        return max(self.labelwidth, self.childrenwidth)

    @property
    def height(self):
        return POINT_SIZE + self.depth * LAYER_HEIGHT

    ## Label
    @property
    def labelwidth(self):
        return FONT.getsize(self.label)[0]

    @property
    def labelmiddle(self):
        if len(self) <= 1:
            return floor(self.width/2)
        else:
            return floor((self[0].labelmiddle + (self.width - self[-1].width + self[-1].labelmiddle))/2)

    @property
    def labelleft(self):
        return self.labelmiddle - floor(self.labelwidth/2)

    @property
    def deplabelmiddle(self):
        ix = None
        leaf = None
        for i, child in enumerate(self):
            if child.isleaf:
                if leaf is None:
                    ix = i
                    leaf = child
                else:
                    raise TreeFormatError('dependency nodes may have at most one leaf child')
        if leaf is None:
            return self.labelmiddle
        else:
            return sum(child.width for child in self[:ix]) + GAP_WIDTH*ix + floor((self.width-self.childrenwidth)/2) + floor(leaf.labelwidth/2)

    @property
    def deplabelleft(self):
        return self.deplabelmiddle - floor(self.labelwidth/2)

## Compiling Functions
def tokenise(string):
    for match in TOKEN_REGEX.finditer(string):
        type = match.lastgroup
        value = match.group()
        column = match.start()
        if type == 'WHITESPACE':
            continue
        elif type == 'QUOTED':
            type = 'STRING'
            value = value.strip('"')
        elif type == 'INDEX':
            value = value.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
        elif type == 'UNKNOWN':
            raise CompilerError('unexpected character', value, 0, column)
        yield Token(type, value, 0, column)

def matchBrackets(tokens, start=0):
    if tokens[start].type != 'LBRACKET':
        raise UnexpectedToken(tokens[start], 'lbracket')
    depth = 0
    for i, token in enumerate(tokens[start:], start+1):
        if token.type == 'LBRACKET':
            depth += 1
        elif token.type == 'RBRACKET':
            depth -= 1
            if depth == 0:
                return i
    raise TokenError(f'unmatched bracket', tokens[start])

def compileTree(tokens):
    if tokens[0].type != 'STRING':
        raise UnexpectedToken(tokens[0], 'string')
    label = tokens[0].value
    children = []
    i = 1
    while i < len(tokens):
        type, value = tokens[i]
        if type == 'LBRACKET':
            j = matchBrackets(tokens, i)
            children.append(compileTree(tokens[i+1:j-1]))
            i = j
        elif type == 'STRING':
            children.append(Tree(value))
            i += 1
        else:
            raise UnexpectedToken(tokens[i])
    return Tree(label, children)

## Drawing Functions
def drawDependency(tree, draw, leaftop, top, left):
    # Draw label
    labelcolour = 'red' if tree.isleaf else 'blue'
    draw.text((left+tree.deplabelleft, top), tree.label, labelcolour, FONT)
    # Draw descendents
    linetop = (left+tree.deplabelmiddle, top+POINT_SIZE+SCALE)  # We want 1px gap between label and line after rescaling
    top += LAYER_HEIGHT
    left += floor((tree.width - tree.childrenwidth)/2)
    for child in tree:
        if child.isleaf:
            _top = leaftop
        else:
            _top = top
        # Draw line
        linebottom = (left+child.deplabelmiddle, _top-SCALE)  # Again, 1px gap between label and line after rescaling
        linecolour = 'darkgrey' if child.isleaf else 'black'
        draw.line([linetop, linebottom], linecolour, SCALE)  # Similarly, 1px line width after rescaling
        # Draw child
        drawDependency(child, draw, leaftop, _top, left)
        # Next child
        left += child.width + GAP_WIDTH

def drawConstituency(tree, draw, top, left):
    # Draw label
    labelcolour = 'red' if tree.isleaf else 'blue'
    draw.text((left+tree.labelleft, top), tree.label, labelcolour, FONT)
    # Draw descendents
    linetop = (left+tree.labelmiddle, top+POINT_SIZE+SCALE)  # We want 1px gap between label and line after rescaling
    top += LAYER_HEIGHT
    left += floor((tree.width - tree.childrenwidth)/2)
    for child in tree:
        # Draw line
        linebottom = (left+child.labelmiddle, top-SCALE)  # Again, 1px gap between label and line after rescaling
        linecolour = 'darkgrey' if child.isleaf else 'black'
        draw.line([linetop, linebottom], linecolour, SCALE)  # Similarly, 1px line width after rescaling
        # Draw child
        drawConstituency(child, draw, top, left)
        # Next child
        left += child.width + GAP_WIDTH

def drawTree(string, mode):
    tree = Tree.make(string)
    size = (tree.width + PADDING*2, tree.height + PADDING*2)
    im = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(im)
    if mode == 'dep':
        leaftop = PADDING + tree.depth*LAYER_HEIGHT
        drawDependency(tree, draw, leaftop, PADDING, PADDING)
    else:
        drawConstituency(tree, draw, PADDING, PADDING)
    return im.resize((size[0]//SCALE, size[1]//SCALE), resample=Image.ANTIALIAS)
