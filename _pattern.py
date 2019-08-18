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
    parse_pattern  -- parses a string utilising pattern notation into a list of elements
    parse_patterns -- parses a collection of strings using pattern notation
    match_pattern  -- matches a list of elements to a specified slice of a word
''''''
==================================== To-do ====================================
=== Bug-fixes ===
catixes in match_pattern should be redone to cope with non-linearity

=== Implementation ===
Replace super-disgusting hacky wildcard repetition workaround in match_pattern with something better
- How though
Handling of optionals needs a lot of work

=== Features ===

=== Style ===
'''
import re
from dataclasses import dataclass, InitVar
from typing import Dict, List
from .core import Cat, FormatError

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
    'ESCAPE': r'\{u\d+\}',
    'REPETITION': r'\{\d+\}',
    'WILDCARD': r'\*\*?\??',
    'TARGETREF': r'%|<',
    'DITTO': r'\"',
    'SYLBREAK': r'\$',
    'TEXT': r'[\w#~]+',
    'UNKNOWN': r'.',
}
TOKEN_REGEX = re.compile('|'.join(f'(?P<{type}>{regex})' for type, regex in TOKENS.items()))

## Classes
@dataclass
class Token:
    type: str
    value: str
    column: int

    def __iter__(self):
        return iter((self.type, self.value))

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

    @classmethod
    def make(cls, string=None, cats=None):
        return cls()

    @classmethod
    def fromTokens(cls, tokens=None, cats=None):
        if len(tokens) != 1:
            raise FormatError(f'too many tokens: {tokens}')
        if tokens[0].type == cls.__name__.upper():
            return cls()
        raise FormatError(f'invalid token: {tokens[0].value} @ {tokens[0].column}')


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
    def fromTokens(tokens, cats=None):
        if len(tokens) != 1:
            raise FormatError(f'too many tokens: {tokens}')
        type, value = tokens[0]
        if type == 'ESCAPE':
            return Grapheme(grapheme=chr(int(value[2:-1])))
        raise FormatError(f'invalid token: {value} @ {tokens[0].column}')

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
            return f'[{self.cat}]'
        else:
            return f'[{self.cat.name}]'

    def __eq__(self, other):
        if isinstance(other, Category):
            return self.cat == other.cat
        else:
            return self.cat == other

    @staticmethod
    def make(string, cats=None):
        from .core import Cat
        if string.startswith('[') and string.endswith(']'):
            cat = string[1:-1]
        else:
            raise FormatError(f'{string!r} is not a valid category')
        if ',' in cat:  # Nonce cat
            return Category(cat=Cat(cat, cats))
        elif cats is not None and cat in cats:
            return Category(cat=cats[cat])
        else:
            raise FormatError(f'{string!r} is not a defined category')

    @staticmethod
    def fromTokens(tokens, cats=None):
        from .core import Cat
        if tokens[0].type != 'LCAT' or tokens[-1].type != 'RCAT':
            raise FormatError(f'the given tokens are not a valid category: {tokens}')
        cat = ''.join(token.value for token in tokens[1:-1])
        if ',' in cat:  # Nonce cat
            return Category(cat=Cat(cat, cats))
        elif cats is not None and cat in cats:
            return Category(cat=cats[cat])
        else:
            raise FormatError(f'{cat!r} is not a defined category')

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
        if string in ('*', '**', '*?', '**?'):
            greedy = not string.endswith('?')
            extended = string.startswith('**')
            return Wildcard(greedy=greedy, extended=extended)
        else:
            raise FormatError(f'{string!r} is an invalid wildcard')

    @staticmethod
    def fromTokens(tokens, cats=None):
        if len(tokens) != 1:
            raise FormatError(f'too many tokens: {tokens}')
        type, value = tokens[0]
        if type == 'WILDCARD':
            if value in ('*', '**', '*?', '**?'):
                greedy = not value.endswith('?')
                extended = value.startswith('**')
                return Wildcard(greedy=greedy, extended=extended)
        raise FormatError(f'invalid token: {value} @ {tokens[0].column}')

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
        elif string == '{*?}':
            return WildcardRep(greedy=False)
        else:
            raise FormatError(f'{string!r} is an invalid wildcard repetition')

    @staticmethod
    def fromTokens(tokens, cats=None):
        if len(tokens) != 1:
            raise FormatError(f'too many tokens: {tokens}')
        type, value = tokens[0]
        if type == 'WILDCARDREP':
            if value == '{*}':
                return WildcardRep(greedy=True)
            elif value == '{*?}':
                return WildcardRep(greedy=False)
        raise FormatError(f'invalid wildcard repetition: {value} @ {tokens[0].column}')

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
        pattern = ''.join(str(token) for token in self.pattern)  # Could be improved probably
        return f'({pattern})' if self.greedy else f'({pattern})?'

    @staticmethod
    def make(string, cats=None):
        greedy = not string.endswith('?')
        pattern = parse_pattern(string.rstrip('?')[1:-1], cats)
        if len(pattern) == 1 and isinstance(pattern[0], Wildcard):
            pattern[0].greedy = greedy
        return Optional(greedy=greedy, pattern=pattern)

    @staticmethod
    def fromTokens(tokens, cats=None):
        if tokens[0].type != 'LOPT' or tokens[-1].type != 'ROPT':
            raise FormatError(f'the given tokens are not a valid category: {tokens}')
        greedy = not tokens[-1].value.endswith('?')
        pattern = compile_tokens(tokens[1:-1], cats)
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
        if not string.startswith('{') or not string.endswith('}'):
            raise FormatError(f'{string!r} is an invalid comparison')
        string = string[1:-1]
        for op in ('==', '=', '!=', '>=', '>', '<=', '<'):
            if string.startswith(op):
                try:
                    value = int(string[len(op):])
                except ValueError:
                    raise FormatError(f'{string!r} has an invalid value')
                if op == '=':
                    op = '=='
                return Comparison(operation=op, value=value)
        raise FormatError(f'{string!r} does not have a valid operation')

    @staticmethod
    def fromTokens(tokens, cats=None):
        if len(tokens) != 1:
            raise FormatError(f'too many tokens: {tokens}')
        type, value = tokens[0]
        if type == 'COMPARISON':
            if not value.startswith('{') or not value.endswith('}'):
                raise FormatError(f'{value!r} is an invalid comparison')
            comparison = value[1:-1]
            for op in ('==', '=', '!=', '>=', '>', '<=', '<'):
                if comparison.startswith(op):
                    try:
                        num = int(comparison[len(op):])
                    except ValueError:
                        raise FormatError(f'{value!r} has an invalid value')
                    if op == '=':
                        op = '=='
                    return Comparison(operation=op, value=num)
        raise FormatError(f'invalid token: {value} @ {tokens[0].column}')

@dataclass(repr=False, eq=False)
class TargetRef(Element):
    direction: int

    def __str__(self):
        return '%' if self.direction == 1 else '<'

    @staticmethod
    def make(string, cats=None):
        if string == '%':
            return TargetRef(direction=1)
        elif string == '<':
            return TargetRef(direction=-1)
        else:
            raise FormatError(f'{string!r} is not a target reference')

    @staticmethod
    def fromTokens(tokens, cats=None):
        if len(tokens) != 1:
            raise FormatError(f'too many tokens: {tokens}')
        type, value = tokens[0]
        if type == 'TARGETREF':
            if value == '%':
                return TargetRef(direction=1)
            elif value == '<':
                return TargetRef(direction=-1)
        raise FormatError(f'invalid token: {value} @ {tokens[0].column}')

    def resolve_target(self, target):
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

def escape(string):
    while True:
        ix = string.find('\\')
        if ix == -1:
            break
        string = string[:ix] + f'{{u{ord(string[ix+1])}}}' + string[ix+2:]
    return string

def tokenise(string, colstart=None):
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
            yield Token('END', '', colstart)
        return
    brackets = []
    for match in TOKEN_REGEX.finditer(string):
        type = match.lastgroup
        value = match.group()
        column = match.start() + colstart
        if type == 'COMMA':
            if not (brackets and brackets[-1] == '['):  # Commas are only licensed inside categories
                if nested:
                    yield Token('END', value, column)
                    return
                else:
                    raise FormatError(f'unexpected character: {value} @ {column}')
        elif type in ('LOPT', 'LCAT'):  # Left brackets
            if value == '(' and brackets and brackets[-1] == '[':
                raise FormatError(f'optionals may not appear inside categories: {value} @ {column}')
            brackets.append(value)
        elif type in ('ROPT', 'RCAT'):  # Right brackets
            if not brackets:
                raise FormatError(f'unexpected bracket: {value} @ {column}')
            bracket = brackets.pop()
            if bracket+value[0] not in ('()', '[]'):
                raise FormatError(f'mismatched brackets: {value} @ {column}')
        elif type == 'UNKNOWN':
            if nested:
                yield Token('END', value, column)
                return
            else:
                raise FormatError(f'unexpected character: {value} @ {column}')
        yield Token(type, value, column)

def parse_pattern(pattern, cats=None):
    '''Parse a string using pattern notation.

    Arguments:
        pattern -- the input string using pattern notation (str or Word)
        cats    -- a dictionary of categories to use for interpreting categories (dict)

    Returns a list
    '''
    from .core import Word, split, parse_word
    if isinstance(pattern, Word):
        return [Grapheme(graph) for graph in Word]
    if cats is not None and 'graphs' in cats:
        graphs = cats['graphs']
    else:
        graphs = Cat("'")
    for char in '([{}])*?"$%<':
        pattern = pattern.replace(char, f' {char} ')
    pattern = pattern.replace('  ?', '?').replace('*  *', '**')
    pattern = split(pattern, ' ', nesting=(0, '([{', '}])'), minimal=True)
    for i, token in reversed(list(enumerate(pattern))):
        token = token.replace(' ', '')
        if not token or token == '[]':  # Blank or null
            del pattern[i]
        elif token[0] == '(':  # Optional
            if i < len(pattern)-1 and pattern[i+1].type == 'WildcardRep':
                token = token.rstrip('?') if pattern[i+1].greedy else (token+'?')
            pattern[i] = Optional.make(token, cats)
            # To-do - reimplement flattening optionals
        elif token[0] == '[':  # Category
            pattern[i] = Category.make(token, cats)
        elif token[0] == '{':  # Numbers - a few types of this
            token = token[1:-1]
            if token[0] in '!=<>':  # Comparison - parse to tuple
                pattern[i] = Comparison.make(token)
            elif token in ('*', '*?'):  # Wildcard repetition
                pattern[i] = WildcardRep.make(token)
            elif token.startswith('u'):  # Escaped character
                pattern[i] = Grapheme.make(chr(token[1:]))
            else:  # Repetitions - parse to int
                pattern[i] = int(token)
        elif token in ('*', '**', '*?', '**?'):  # Wildcard
            pattern[i] = Wildcard.make(token)
        elif token in ('%', '<'):  # Target reference
            pattern[i] = TargetRef.make(token)
        elif token == '"':  # Ditto
            pattern[i] = Ditto.make()
        elif token == '$':  # Syllable break
            pattern[i] = SylBreak.make()
        else:  # Text - parse as word
            pattern[i:i+1] = [Grapheme(graph) for graph in parse_word(token, graphs)]
    for i, token in reversed(list(enumerate(pattern))):  # Second pass to evaluate repetitions
        if isinstance(token, int):
            pattern[i-1:i+1] = [pattern[i-1]]*token
    return pattern

def parse_patterns(patterns, cats=None):
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
                _patterns.append(parse_pattern(pattern, cats))
            else:
                _patterns.append(pattern)
    elif isinstance(patterns, dict):
        _patterns = {key: parse_patterns(patterns[key], cats) for key in patterns}
    else:
        _patterns = None
    return _patterns

def match_pattern(word, pattern, start, end, step, stack=None):
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
        for i, token in enumerate(pattern):
            if token.type == 'WildcardRep':
                pattern[i-1:i+1] = reversed(pattern[i-1:i+1])
    matched = True
    while 0 <= ix < len(pattern):
        if start <= pos < end:  # Still in the slice
            token = pattern[ix]
            if not isinstance(token, Optional):
                matched, length, ilength, _stack, _catixes = token.match(word, pos, ix, step, istep)
                stack.extend(_stack)
                catixes.extend(_catixes)
            else:  # Optionals require special handling
                if not matched:  # Jumped here via the stack, check if we've got a nested stack reference
                    if stack and isinstance(stack[-1], list):
                        _stack = stack.pop()
                else:
                    _stack = []
                if token.greedy:  # Greedy
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
                if token.greedy or not matched:
                    _start, _end = (pos, end) if istep > 0 else (start, pos+1)
                    matched, rpos, _catixes, _stack = match_pattern(word, token.pattern, _start, _end, step, _stack)
                    # Merge in the stack - if a reference has an index within token, nest it and push a reference to
                    # the token, else correct the index and push it directly
                    for _pos, _ix in _stack:
                        if _ix >= len(token.pattern):
                            _ix -= len(token.pattern)-1
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
