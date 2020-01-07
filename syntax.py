'''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Anti-aliasing: https://stackoverflow.com/questions/14350645/is-there-an-antialiasing-method-for-python-pil
Multi-word labels require that triangle thing

=== Features ===

=== Style ===
'''

import re
from dataclasses import dataclass, field
from math import floor
from PIL import Image, ImageDraw, ImageFont
from .core import LangException, Token, CompilerError, TokenError

## Constants
POINT_SIZE = 16
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

class UnexpectedToken(Exception):
    def __init__(self, token, expected=None):
        type, value, linenum, column = token.type.lower(), token.value, token.linenum, token.column
        if expected is None:
            super().__init__(f'Unexpected {type} token: {value} @ {linenum}:{column}')
        else:
            super().__init__(f'Unexpected {type} token, expected {expected}: {value} @ {linenum}:{column}')

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
            return compileNode(tokens[1:-1])
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
            children.append(compileNode(tokens[i+1:j-1]))
            i = j
        elif type == 'STRING':
            children.append(Node(value))
            i += 1
        else:
            raise UnexpectedToken(tokens[i])
    return Node(label, children)

## Drawing Functions
def drawDependency(tree, draw, depth, top, left):
    # Draw label

def drawConstituency(tree, draw, top, left):
    # Draw label
    labelcolour = 'red' if tree.isleaf else 'blue'
    draw.text((left+tree.labelleft, top), tree.label, labelcolour, FONT)
    # Draw descendents
    linetop = (left+tree.labelmiddle, top+POINT_SIZE+1)
    top += LAYER_HEIGHT
    left += floor((tree.width - tree.childrenwidth)/2)
    for child in tree:
        # Draw line
        linebottom = (left+child.labelmiddle, top-1)
        linecolour = 'darkgrey' if child.isleaf else 'black'
        draw.line([linetop, linebottom], linecolour, 1)
        # Draw child
        drawConstituency(child, draw, top, left)
        # Next child
        left += child.width + GAP_WIDTH

def drawTree(string, mode):
    tree = Node.make(string)
    size = (tree.width + PADDING*2, tree.height + PADDING*2)
    im = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(im)
    if mode == 'dep':
        drawDependency(tree, draw, tree.depth, PADDING, PADDING)
    else:
        drawConstituency(tree, draw, PADDING, PADDING)
    return im
