'''Pattern parsing and matching

Classes:
    Token       -- Class utilised by the tokeniser
    Element     -- Base class for pattern elements
    Grapheme    -- Element matching a specific grapheme
    Ditto       -- Element matching the second of two identical segments
    SylBreak    -- Element matching a syllable boundary
    Category    -- Element matching a category of graphemes
    Wildcard    -- Element matching one or more arbitrary segments
    WildcardRep -- Element matching one or more copies of the previous element
    Optional    -- Element matching an optional sequence of elements
    Comparison  -- Element used for indicating the number of another element
    TargetRef   -- Element used to refer to the target

Functions:
    escape         -- processes escaped characters in a string
    tokenise       -- returns a generator producing tokens
    parsePattern  -- parses a string utilising pattern notation into a list of elements
    parsePatterns -- parses a collection of strings using pattern notation
    matchPattern  -- matches a list of elements to a specified slice of a word
''''''
==================================== To-do ====================================
=== Bug-fixes ===
catixes in matchPattern should be redone to cope with non-linearity

=== Implementation ===
Replace super-disgusting hacky wildcard repetition workaround in matchPattern with something better
- How though
Handling of optionals needs a lot of work

=== Features ===

=== Style ===
'''
import re
from dataclasses import dataclass, InitVar
from typing import Dict, List
from .core import FormatError, CompilerError, TokenError, Token, Cat

## Constants
TOKENS = {
    'COMMA': r', ?',
    'NULL': r'\[\]',
    'LOPT': r'\(',
    'ROPT': r'\)\??',
    'LCAT': r'\[',
    'RCAT': r'\]',
    'WILDCARDREP': r'\{\*\??\}',
    'COMPARISON': r'\{(?:!=|[=<>]=?)\d+\}',
    'ESCAPE': r'\\.',
    'REPETITION': r'\{\d+\}',
    'WILDCARD': r'\*\*?\??',
    'TARGETREF': r'%|<',
    'DITTO': r'\"',
    'SYLBREAK': r'\$',
    'TEXT': r'[^ >\/!+\-[\](){}*?\\"%<$^,&_~@]+',
    'UNKNOWN': r'.',
}
TOKEN_REGEX = re.compile('|'.join(f'(?P<{type}>{regex})' for type, regex in TOKENS.items()))
TOKENS = {type: re.compile(regex) for type, regex in TOKENS.items()}

## Classes
@dataclass(repr=False, eq=False)
class Element:
    def __str__(self):
        return ''

    def __repr__(self):
        return f'{self.type}({str(self)!r})'

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        elif type(self) == type(other):
            return str(self) == str(other)
        else:
            return NotImplemented

    @property
    def type(self):
        return self.__class__.__name__

    # This method should not be called directly, as it does not check its arguments for correctness
    @classmethod
    def make(cls, string=None, cats=None):
        return cls()

    @classmethod
    def fromString(cls, string=None, cats=None):
        if TOKENS[cls.__name__.upper()].match(string) is not None:  # Sanity check
            return cls.make(string, cats)
        raise TokenError(f'invalid {cls.__name__}', tokens[0])

    @classmethod
    def fromTokens(cls, tokens=None, cats=None):
        if len(tokens) != 1:
            raise CompilerError(f'too many tokens', tokens, tokens[0].linenum, tokens[0].column)
        type, value = tokens[0]
        if type == cls.__name__.upper() and TOKENS[type].match(value) is not None:  # Sanity check
            return cls.make(value, cats)
        raise TokenError(f'invalid {cls.__name__}', tokens[0])

    # This method must guarantee that the last two return values are [] if the first is False
    def match(self, word, pos, ix, step, istep):
        # matched, length, ilength, stack, catixes
        return False, 0, 0, [], []

## Matching elements ##
@dataclass(repr=False, eq=False)
class Grapheme(Element):
    grapheme: str

    def __str__(self):
        return self.grapheme

    @staticmethod
    def make(string, cats=None):
        return Grapheme(grapheme=string)

    @staticmethod
    def fromString(string=None, cats=None):
        if TOKENS['ESCAPE'].match(string) is not None:  # Sanity check
            return Grapheme(grapheme=value[1])
        raise TokenError(f'invalid {cls.__name__}', tokens[0])

    @staticmethod
    def fromTokens(tokens, cats=None):
        if len(tokens) != 1:
            raise CompilerError(f'too many tokens', tokens, tokens[0].linenum, tokens[0].column)
        type, value = tokens[0]
        if type == 'ESCAPE' and TOKENS['ESCAPE'].match(value) is not None:  # Sanity check
            return Grapheme(grapheme=value[1])
        raise TokenError('invalid Grapheme', tokens[0])

    def match(self, word, pos, ix, step, istep):
        return self.grapheme == word[pos], step, istep, [], []

@dataclass(repr=False, eq=False)
class Ditto(Element):
    def __str__(self):
        return '"'

    def match(self, word, pos, ix, step, istep):
        return word[pos] == word[pos-1], step, istep, [], []

@dataclass(repr=False, eq=False)
class SylBreak(Element):
    def __str__(self):
        return '$'

    def match(self, word, pos, ix, step, istep):
        return (pos in word.syllables), 0, istep, [], []

@dataclass(repr=False, eq=False)
class Category(Element):
    cat: Cat

    def __str__(self):
        if self.cat.name is None:
            return str(self.cat)
        else:
            return f'[{self.cat.name}]'

    def __eq__(self, other):
        if isinstance(other, Category):
            return self.cat == other.cat
        else:
            return self.cat == other

    @staticmethod
    def make(string, cats=None):
        return Category(cat=Cat.make(string, cats))

    @staticmethod
    def fromString(string, cats=None):
        return Category.make(string, cats)

    @staticmethod
    def fromTokens(tokens, cats=None):
        string = ''.join(token.value for token in tokens)
        return Category.make(string, cats)

    def match(self, word, pos, ix, step, istep):
        if word[pos] in self.cat:  # This might change
            return True, step, istep, [], [self.cat.index(word[pos])]
        return False, 0, 0, [], []

@dataclass(repr=False, eq=False)
class Wildcard(Element):
    greedy: bool
    extended: bool

    def __str__(self):
        return ('**' if self.extended else '*') + ('' if self.greedy else '?')

    @staticmethod
    def make(string, cats=None):
        greedy = not string.endswith('?')
        extended = string.startswith('**')
        return Wildcard(greedy=greedy, extended=extended)

    def match(self, word, pos, ix, step, istep):
        if self.extended or word[pos] != '#':
            if self.greedy:
                stack = [(pos+step, ix+istep)]
                istep = 0
            else:
                stack = [(pos+step, ix)]
            return True, step, istep, stack, []
        return False, 0, 0, [], []

@dataclass(repr=False, eq=False)
class WildcardRep(Element):
    greedy: bool

    def __str__(self):
        return '{*}' if self.greedy else '{*?}'

    @staticmethod
    def make(string, cats=None):
        if string == '{*}':
            return WildcardRep(greedy=True)
        else:
            return WildcardRep(greedy=False)

    def match(self, word, pos, ix, step, istep):
        if not self.greedy:
            istep *= -1
        return True, 0, istep, [(pos, ix-istep)], []

## Non-matching elements ##
@dataclass(repr=False, eq=False)
class Optional(Element):
    greedy: bool
    pattern: List[Token]

    def __str__(self):
        string = unparsePattern(self.pattern)
        return f'({string})' if self.greedy else f'({string})?'

    @staticmethod
    def make(string, cats=None):
        greedy = not string.endswith('?')
        pattern = parsePattern(string.rstrip('?')[1:-1], cats)
        if len(pattern) == 1 and isinstance(pattern[0], Wildcard):
            pattern[0].greedy = greedy
        return Optional(greedy=greedy, pattern=pattern)

    @staticmethod
    def fromString(string, cats=None):
        return Optional.make(string, cats)

    @staticmethod
    def fromTokens(tokens, cats=None):
        if tokens[0].type != 'LOPT' or tokens[-1].type != 'ROPT':
            raise FormatError(f'the given tokens are not a valid optional: {tokens}')
        greedy = not tokens[-1].value.endswith('?')
        pattern = compile(tokens[1:-1], cats)
        if len(pattern) == 1 and isinstance(pattern[0], Wildcard):
            pattern[0].greedy = greedy
        return Optional(greedy=greedy, pattern=pattern)

    # Somehow I need to adapt the special matching code for this framework - won't be easy

@dataclass(repr=False, eq=False)
class Comparison(Element):
    operation: str
    value: int

    def __str__(self):
        return f'{{{self.operation}{self.value}}}'.replace('==', '=')

    @staticmethod
    def make(string, cats=None):
        string = string[1:-1]
        for op in ('==', '=', '!=', '>=', '>', '<=', '<'):
            if string.startswith(op):
                value = int(string[len(op):])
                if op == '=':
                    op = '=='
                return Comparison(operation=op, value=value)

@dataclass(repr=False, eq=False)
class TargetRef(Element):
    direction: int

    def __str__(self):
        return '%' if self.direction == 1 else '<'

    @staticmethod
    def make(string, cats=None):
        if string == '%':
            return TargetRef(direction=1)
        else:
            return TargetRef(direction=-1)

    def resolveTarget(self, target):
        return [Grapheme(graph) for graph in (target if self.direction == 1 else reversed(target))]

ELEMENT_DICT = {
    'LOPT': Optional,
    'LCAT': Category,
    'WILDCARDREP': WildcardRep,
    'COMPARISON': Comparison,
    'ESCAPE': Grapheme,
    'WILDCARD': Wildcard,
    'TARGETREF': TargetRef,
    'DITTO': Ditto,
    'SYLBREAK': SylBreak,
}

# Don't slice the string when calling this
def tokenise(string, colstart=None, linenum=0):
    '''Tokenise a string using pattern notation.

    Arguments:
        string   -- the input string using pattern notation (str)
        colstart -- the column to start token indexing at (int)

    Yields Token objects
    '''
    if colstart is None:
        nested = False
        colstart = 0
    else:
        nested = True
    if not string:
        if nested:
            return colstart
        return
    brackets = []
    for match in TOKEN_REGEX.finditer(string, colstart):
        type = match.lastgroup
        value = match.group()
        column = match.start()
        colstart = match.end()
        if type == 'COMMA':
            if not (brackets and brackets[-1] == '['):
                if not brackets and nested:
                    return column
                raise CompilerError(f'unexpected comma', value, linenum, column)
        elif type in ('LOPT', 'LCAT'):  # Left brackets
            if value == '(' and brackets and brackets[-1] == '[':
                raise CompilerError(f'optionals may not appear inside categories', value, linenum, column)
            brackets.append(value)
        elif type in ('ROPT', 'RCAT'):  # Right brackets
            if not brackets:
                raise CompilerError(f'unexpected bracket', value, linenum, column)
            bracket = brackets.pop()
            if bracket+value[0] not in ('()', '[]'):
                raise CompilerError(f'mismatched brackets', value, linenum, column)
        elif type == 'UNKNOWN':
            if nested:
                return column
            else:
                raise CompilerError(f'unexpected character', value, linenum, column)
        yield Token(type, value, linenum, column)
    if nested:
        return colstart

def matchBrackets(tokens, start=0):
    if tokens[start].type not in ('LOPT', 'LCAT'):
        raise TokenError(f'expected bracket', tokens[start])
    else:
        left = tokens[start].type
        right = left.replace('L', 'R')
    depth = 0
    for i, token in enumerate(tokens[start:], start+1):
        if token.type == left:
            depth += 1
        elif token.type == right:
            depth -= 1
            if depth == 0:
                return i
    raise TokenError(f'unmatched bracket', tokens[start])

def compile(tokens, cats=None):
    from .core import parseWord
    tokens = list(tokens)
    if not tokens:
        return []
    if cats is not None and 'graphs' in cats:
        graphs = cats['graphs']
    else:
        graphs = ()
    elements = []
    i = 0
    while i < len(tokens):
        type, value = tokens[i]
        if type in ('LOPT', 'LCAT'):
            j = matchBrackets(tokens, i)
        else:
            j = i+1
        if type == 'NULL':
            pass
        elif type == 'REPETITION':
            elements[-1:] = elements[-1:]*int(value[1:-1])
        elif type == 'TEXT':
            elements.extend([Grapheme(graph) for graph in parseWord(value, graphs)])
        elif type in ELEMENT_DICT:
            cls = ELEMENT_DICT[type]
            elements.append(cls.fromTokens(tokens[i:j], cats))
        else:
            raise TokenError(f'unexpected token', tokens[i])
        i = j
    return elements

def parsePattern(pattern, cats=None):
    '''Parse a string using pattern notation.

    Arguments:
        pattern -- the input string using pattern notation (str or Word)
        cats    -- a dictionary of categories to use for interpreting categories (dict)

    Returns a list
    '''
    from .core import Word
    if isinstance(pattern, Word):
        return [Grapheme(graph) for graph in pattern]
    try:
        return compile(tokenise(pattern), cats)
    except CompilerError as e:
        raise FormatError(f'invalid pattern: {pattern!r}; {e.args[0]}')

def unparsePattern(pattern, graphs=(), separator=''):
    from .core import unparseWord
    # Add collapsing repeated tokens
    elements = []
    for element in pattern:
        if isinstance(element, Optional):
            string = unparsePattern(element.pattern, graphs, separator)
            elements.append(f'({string})' if self.greedy else f'({string})?')
        else:
            elements.append(str(element))
    return unparseWord(elements, graphs, separator)

def parsePatterns(patterns, cats=None):
    '''Parses generation patterns.

    Arguments:
        patterns -- set of patterns to parse (str, list, or dict)

    Returns a list
    '''
    if isinstance(patterns, str):
        patterns = patterns.splitlines()
    if isinstance(patterns, list):
        _patterns = []
        for pattern in patterns:
            #Remove comments
            if isinstance(pattern, str):
                pattern = pattern.split('//')[0]
            if not pattern:
                continue
            if isinstance(pattern, str):
                _patterns.append(parsePattern(pattern, cats))
            else:
                _patterns.append(pattern)
    elif isinstance(patterns, dict):
        _patterns = {key: parsePatterns(patterns[key], cats) for key in patterns}
    else:
        _patterns = None
    return _patterns

def matchPattern(word, pattern, start, end, step, stack=None):
    '''Match a pattern sequence to the word.

    Return if the sequence matches the end of the given slice of the word, the far end of the match, and category indexes.

    Arguments:
        word -- the word to match to
        pattern -- the sequence being matched
        start, end, step -- determine the slice of the word to match within
        stack -- used to pass stack references into an optional segment

    Returns a tuple.
    '''
    pos = start if step > 0 else end-1
    ix = 0 if step > 0 else (len(pattern)-1)
    istep = 1 if step > 0 else -1
    if stack is None:
        stack = []  # This stores the positions in the word and sequence that we branched at
        _returnstack = False
    else:
        if stack:
            pos, ix = stack.pop()
        _returnstack = True
    catixes = []  # This records the index of each category match. This needs to be redone to cope with non-linearity
    # Hacky thing for now to make wildcard repetitions actually work in rtl
    pattern = pattern.copy()
    if step < 0:
        for i, element in enumerate(pattern):
            if element.type == 'WildcardRep':
                pattern[i-1:i+1] = reversed(pattern[i-1:i+1])
    matched = True
    while 0 <= ix < len(pattern):
        if start <= pos < end:  # Still in the slice
            element = pattern[ix]
            if not isinstance(element, Optional):
                matched, length, ilength, _stack, _catixes = element.match(word, pos, ix, step, istep)
                stack.extend(_stack)
                catixes.extend(_catixes)
            else:  # Optionals require special handling
                if not matched:  # Jumped here via the stack, check if we've got a nested stack reference
                    if stack and isinstance(stack[-1], list):
                        _stack = stack.pop()
                else:
                    _stack = []
                if element.greedy:  # Greedy
                    if ix < len(pattern)-istep and pattern[ix+istep].type == 'WildcardRep':  # We need to make sure to step past a wildcard repetition
                        stack.append((pos, ix+istep*2))
                    else:
                        stack.append((pos, ix+istep))
                    ilength = step
                elif matched:  # Non-greedy, we stepped in normally
                    stack.append((pos, ix))
                    if ix < len(pattern)-istep and pattern[ix+istep].type == 'WildcardRep':  # We need to make sure to step past a wildcard repetition
                        ilength = istep*2
                    else:
                        ilength = istep
                    matched = True
                    length = 0
                if element.greedy or not matched:
                    _start, _end = (pos, end) if istep > 0 else (start, pos+1)
                    matched, rpos, _catixes, _stack = matchPattern(word, element.pattern, _start, _end, step, _stack)
                    # Merge in the stack - if a reference has an index within element, nest it and push a reference to
                    # the element, else correct the index and push it directly
                    for _pos, _ix in _stack:
                        if _ix >= len(element.pattern):
                            _ix -= len(element.pattern)-1
                            stack.append((_pos, _ix))
                        else:
                            if len(stack) >= 2 and isinstance(stack[-2], list):
                                stack[-2].append((_pos, _ix))
                            else:
                                stack.append([(_pos, _ix)])
                                stack.append((_pos, ix))
                    length = rpos-pos
                    if matched:
                        catixes.extend(_catixes)
        else:
            matched, length, ilength = False, 0, 0
        if matched:
            ix += ilength
            pos += length
        elif stack:  # This segment failed to match, so we jump back to the next branch
            pos, ix = stack.pop()
        else:  # Total match failure
            if _returnstack:
                return False, 0, [], []  # Maybe?
            else:
                return False, 0, []
    if _returnstack:
        return True, pos, catixes, stack
    else:
        return True, pos, catixes
