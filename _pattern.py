'''Pattern parsing and matching

Classes:
    Token       -- Base class for pattern tokens
    Grapheme    -- Token matching a specific grapheme
    Ditto       -- Token matching the second of two identical segments
    SylBreak    -- Token matching a syllable boundary
    Category    -- Token matching a category of graphemes
    Wildcard    -- Token matching one or more arbitrary segments
    WildcardRep -- Token matching one or more copies of the previous token
    Optional    -- Token matching an optional sequence of tokens

Functions:
    match_pattern  -- matches a list of tokens to a specified slice of a word
    parse_pattern  -- parses a string utilising pattern notation into a list of tokens
    parse_patterns -- parses a collection of strings using pattern notation
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

class Token():
    __slots__ = ()
    type = None
    
    def __str__(self):
        return ''
    
    def __repr__(self):
        return f'Token({self})'
    
    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        elif type(self) == type(other):
            return str(self) == str(other)
        else:
            return NotImplemented
    
    # This method must guarantee that the last two return values are [] if the first is False
    def match(self, word, pos, ix, step, istep):
        # matched, length, ilength, stack, catixes
        return False, 0, 0, [], []

## Matching tokens ##
class Grapheme(Token):
    __slots__ = ('grapheme',)
    type = 'grapheme'
    
    def __init__(self, grapheme):
        self.grapheme = grapheme
    
    def __str__(self):
        return self.grapheme
    
    def match(self, word, pos, ix, step, istep):
        return self.grapheme == word[pos], step, istep, [], []

class Ditto(Token):
    __slots__ = ()
    type = 'ditto'
    
    def __str__(self):
        return '"'
    
    def match(self, word, pos, ix, step, istep):
        return word[pos] == word[pos-1], step, istep, [], []

class SylBreak(Token):
    __slots__ = ()
    type = 'sylbreak'
    
    def __str__(self):
        return '$'
    
    def match(self, word, pos, ix, step, istep):
        return (pos in word.syllables), 0, istep, [], []

class Category(Token):
    __slots__ = ('cat',)
    type = 'category'
    
    def __init__(self, cat, cats):
        from .core import FormatError, Cat
        cat = cat[1:-1]
        if ',' in cat:  # Nonce cat
            self.cat = Cat(cat, cats)
        elif cat in cats:
            self.cat = cats[cat]
        else:
            raise FormatError(f'`{token}` is not a defined category')
    
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
    
    def match(self, word, pos, ix, step, istep):
        if word[pos] in self.cat:  # This might change
            return True, step, istep, [], [self.cat.index[word[pos]]]
        return False, 0, 0, [], []

class Wildcard(Token):
    __slots__ = ('greedy', 'extended')
    type = 'wildcard'

    def __init__(self, wildcard):
        self.greedy = not wildcard.endswith('?')
        self.extended = wildcard.startswith('**')
        
    def __str__(self):
        return ('**' if self.extended else '*') + ('' if self.greedy else '?')

    def match(self, word, pos, ix, step, istep):
        if self.extended or word[pos] != '#':
            if self.greedy:
                stack = [(pos+step, ix+istep)]
                istep = 0
            else:
                stack = [(pos+step, ix)]
            return True, step, istep, stack, []
        return False, 0, 0, [], []

class WildcardRep(Token):
    __slots__ = ('greedy',)
    type = 'wildcardrep'
    
    def __init__(self, wildcardrep):
        self.greedy = not wildcardrep.endswith('?')
        
    def __str__(self):
        return '{*}' if self.greedy else '{*?}'
    
    def match(self, word, pos, ix, step, istep):
        if not self.greedy:
            istep *= -1
        return True, 0, istep, [(pos, ix-istep)], []

## Non-matching tokens ##
class Optional(Token):
    __slots__ = ('greedy', 'pattern')
    type = 'optional'
    
    def __init__(self, optional, cats):
        self.greedy = not optional.endswith('?')
        self.pattern = parse_pattern(optional.rstrip('?')[1:-1], cats)
        if len(self.pattern) == 1 and isinstance(self.pattern[0], Wildcard):
            self.pattern[0].greedy = self.greedy
    
    def __str__(self):
        return '()' if self.greedy else '()?'
    
    # Somehow I need to adapt the special matching code for this framework - won't be easy

class Comparison(Token):
    __slots__ = ('operation', 'value')
    type = 'comparison'
    
    def __init__(self, comparison):
        op = comparison[0]
        if comparison[1] == '=' or op == '=':
            op += '='
        self.operation = op
        self.value = int(comparison.strip('=<>'))
    
    def __str__(self):
        return f'{{{self.operation}{self.value}}}'.replace('==', '=')

class TargetRef(Token):
    __slots__ = ('direction')
    type = 'targetref'
    
    def __init__(self, targetref):
        self.direction = 1 if targetref == '%' else -1 if targetref == '<' else None
    
    def __str__(self):
        return '%' if self.direction == 1 else '<'
    
    def resolve_target(self, target):
        return [Grapheme(graph) for graph in (target if self.direction == 1 else reversed(target))]

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
            if isinstance(token, tuple) and token[0].startswith('*'):
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
                    if isinstance(stack[-1], list):
                        _stack = stack.pop()
                else:
                    _stack = []
                if token.greedy:  # Greedy
                    if isinstance(pattern[ix+istep], WildcardRep):  # We need to make sure to step past a wildcard repetition
                        stack.append((pos, ix+istep*2))
                    else:
                        stack.append((pos, ix+istep))
                elif matched:  # Non-greedy, we stepped in normally
                    stack.append((pos, ix))
                    if isinstance(pattern[ix+istep], WildcardRep):  # We need to make sure to step past a wildcard repetition
                        ilength = istep*2
                    else:
                        ilength = istep
                    matched = True
                    length = 0
                if token.greedy or not matched:
                    _start, _end = (pos, end) if istep > 0 else (start, pos+1)
                    matched, rpos, _catixes, _stack = self.match_pattern(token.pattern, _start, _end, step, _stack)
                    # Merge in the stack - if a reference has an index within token, nest it and push a reference to
                    # the token, else correct the index and push it directly
                    for _pos, _ix in _stack:
                        if _ix >= len(token):
                            _ix -= len(token)-1
                            stack.append((_pos, _ix))
                        else:
                            if isinstance(stack[-2], list):
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

def parse_pattern(pattern, cats=None):
    '''Parse a string using pattern notation.
    
    Arguments:
        pattern -- the input string using pattern notation (str or Word)
        cats    -- a list of cats to use for interpreting categories (list)
    
    Returns a list
    '''
    from .core import Word, Cat, split, parse_word
    if isinstance(pattern, Word):
        return [Grapheme(graph) for graph in Word]
    if cats is None:
        cats = {}
    else:
        cats = cats.copy()
    if 'graphs' not in cats:
        cats['graphs'] = Cat("'")
    for char in '([{}])*?"$%<':
        pattern = pattern.replace(char, f' {char} ')
    pattern = pattern.replace('  ?', '?').replace('*  *', '**')
    pattern = split(pattern, ' ', nesting=(0, '([{', '}])'), minimal=True)
    for i, token in reversed(list(enumerate(pattern))):
        token = token.replace(' ', '')
        if not token or token == '[]':  # Blank or null
            del pattern[i]
        elif token[0] == '(':  # Optional
            if isinstance(pattern[i+1], WildcardRep):
                token = token.rstrip('?') if pattern[i+1].greedy else (token+'?')
            pattern[i] = Optional(token, cats)
            # To-do - reimplement flattening optionals
        elif token[0] == '[':  # Category
            pattern[i] = Category(token, cats)
        elif token[0] == '{':  # Numbers - a few types of this
            token = token[1:-1]
            if token[0] in '=<>':  # Comparison - parse to tuple
                pattern[i] = Comparison(token)
            elif token in ('*', '*?'):  # Wildcard repetition
                pattern[i] = WildcardRep(token)
            else:  # Repetitions - parse to int
                pattern[i] = int(token)
        elif token in ('*', '**', '*?', '**?'):  # Wildcard
            pattern[i] = Wildcard(token)
        elif token in ('%', '<'):  # Target reference
            pattern[i] = TargetRef(token)
        elif token == '"':  # Ditto
            pattern[i] = Ditto()
        elif token == '$':  # Syllable break
            pattern[i] = SylBreak()
        else:  # Text - parse as word
            pattern[i:i+1] = [Grapheme(graph) for graph in parse_word(token, cats['graphs'])]
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
