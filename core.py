'''Base classes and functions

Exceptions:
    LangException -- Base exception
    FormatError   -- Error for incorrect formatting

Classes:
    Cat    -- represents a category of phonemes
    Word   -- represents a run of text
    Config -- collection of gen.py configuration data

Functions:
    parse_syms -- parses a string using pattern notation
    split      -- splits a string
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Investigate reindexing Word
Break out format checking into separate functions
I want to change supplying the actual syllable boundaries to Word to giving a syllabifier function - this is obviously language-dependent
Perhaps adjust Cat.__init__ to allow sequences of graphemes to be stored
After everything, look into using metaclasses in Word

=== Features ===

=== Style ===
Consider where to raise/handle exceptions
Go over docstrings
'''

from collections import namedtuple
from string import whitespace

# == Exceptions == #
class LangException(Exception):
    '''Base class for exceptions in this package'''

class FormatError(LangException):
    '''Exception raised for errors in formatting objects.'''

# == Classes == #
class Cat(list):
    '''Represents a category of graphemes.'''
    
    __slots__ = []
    
    def __init__(self, values=None, cats=None):
        '''Constructor for Cat.
        
        Arguments:
            values -- the values in the category (str, list)
            cats   -- dictionary of categories (dict)
        '''
        _values = []
        if values is None:
            values = []
        elif isinstance(values, str):  # We want an iterable with each value as an element
            values = split(values, ',', minimal=True)
        for value in values:
            if isinstance(value, Cat):  # Another category
                _values.extend(value)
            elif '[' in value:
                if cats is not None and value.strip('[]') in cats:
                    _values.extend(cats[value.strip('[]')])
                else:
                    continue
            else:
                _values.append(value)
        list.__init__(self, _values)
    
    def __repr__(self):
        return f"Cat('{self!s}')"
    
    def __str__(self):
        return ', '.join(self)
    
    def __and__(self, cat):
        return Cat(value for value in self if value in cat)
    
    def __add__(self, cat):
        return Cat(list.__add__(self, cat))
    
    def __sub__(self, cat):
        return Cat(value for value in self if value not in cat)

class Word(list):
    '''Represents a word as a list of graphemes.
    
    Instance variables:
        graphs    -- a list of graphemes (list)
        syllables -- a list of tuples representing syllables (list)
    
    Methods:
        find      -- match a list using pattern notation to the word
        match_env -- match a sound change environment to the word
        strip     -- remove leading and trailing graphemes
    '''
    
    __slots__ = ['graphs', 'syllables']
    
    def __init__(self, lexeme=None, graphs=None, syllables=None):
        '''Constructor for Word
        
        Arguments:
            lexeme    -- the word (str)
            syllables -- list of tuples representing syllables (list)
            graphs    -- category of graphemes (Cat)
        '''
        if graphs is None:
            graphs = Cat("'")
        self.graphs = graphs
        if lexeme is None:
            lexeme = []
        elif isinstance(lexeme, str):
            lexeme = parse_word(f' {lexeme} ', self.graphs)
        else:
            for i in reversed(range(1, len(lexeme))):
                if not isinstance(lexeme[i], str):  # Make sure we only get strings
                    raise ValueError('Iterable values must be strings.')
                if lexeme[i-1] == lexeme[i] == '#':  # Make sure we don't have multiple adjacent '#'s
                    del lexeme[i]
        list.__init__(self, lexeme)
        if syllables is None:
            syllables = []
        self.syllables = syllables  # Do a bit of sanity checking here
    
    def __repr__(self):
        return f"Word('{self!s}')"
    
    def __str__(self):
        word = test = ''
        separator = self.graphs[0]
        polygraphs = (graph for graph in self.graphs if len(graph) > 1)
        for graph in self:
            if not any(graph in poly for poly in polygraphs):
                test = ''  # Can't ever be ambiguous
            elif not test:
                test = graph  # Nothing earlier to be ambiguous with
            else:
                test += graph
                if any(test == poly or poly in test for poly in polygraphs):
                    word += separator  # Ambiguous, so add the separator
                    test = graph
                elif not any(test in poly for poly in polygraphs):
                    test = test[1:]  # Could still be ambiguous with something later
            word += graph
        return word.strip(separator+'#').replace('#',' ')
    
    def __contains__(self, item):
        if isinstance(item, (list, Word)):
            return self.find(item) != -1
        else:
            return list.__contains__(self, item)
    
    def __getitem__(self, item):
        if isinstance(item, slice):
            return Word(list.__getitem__(self, item), self.graphs)
        else:
            return list.__getitem__(self, item)
    
    __setitem__ = None
    __delitem__ = None
    
    def __add__(self, other):
        return Word(list(self) + list(other), self.graphs + other.graphs[1:], self.syllables + other.syllables)
    
    def __mul__(self, other):
        return Word(list(self) * other, self.graphs, self.syllables * other)
    
    def __rmul__(self, other):
        return Word(list(self) * other, self.graphs, self.syllables * other)
    
    def __iadd__(*args):
        return NotImplemented
    
    def __imul__(*args):
        return NotImplemented
    
    append = None
    clear = None
    copy = None
    extend = None
    insert = None
    pop = None
    remove = None
    reverse = None
    sort = None
    
    def strip(self, chars=None):
        if chars is None:
            chars = '#'
        for i in range(len(self)):
            if self[i] not in chars:
                start = i
                break
        else:
            self[:] = []
            return
        for i in reversed(range(len(self))):
            if self[i] not in chars:
                end = i+1
                break
        return self[start:end]
    
    def find(self, sub, start=None, end=None, return_match=False):
        '''Match a sequence using pattern notation to the word.
        
        Arguments:
            sub   -- the list to be found (list)
            start -- the index of the beginning of the range to check (int)
            end   -- the index of the end of the range to check (int)
        
        Returns an int
        '''
        # Interpret start and end according to slice notation
        if start is None:
            start = 0
        elif start < 0:
            start += len(self)
        if end is None:
            end = len(self)
        elif end < 0:
            end += len(self)
        if isinstance(sub, Word):
            sub = sub.strip()  # We want to strip out the leading and trailing '#'s so that this works like finding substrings
        if isinstance(sub[-1], tuple):  # Counting
            matches = 0
            op, count = sub[-1]
            for pos in range(start, end):
                match, length = self.match_pattern(sub[:-1], pos, end)
                if match:
                    matches += 1
            if eval(f'matches {op} count'):
                return (1, []) if return_match else 1
        else:
            for pos in range(start, end):
                match, length = self.match_pattern(sub, pos, end)
                if match:
                    return (pos-start, self[pos:pos+length]) if return_match else pos-start
        return (-1, []) if return_match else -1
    
    def match_pattern(self, seq, start=None, end=None, return_cats=False):
        '''Match a pattern sequence to the word.
        
        Return if the sequence matches the start of the given slice of the word, and how much of the word was matched.
        
        Arguments:
            seq   -- the sequence being matched
            start -- the index to begin matching from
            end   -- the index to match until
        
        Returns a tuple.
        '''
        # Interpret start and end according to slice notation
        if start is None:
            start = 0
        elif start < 0:
            start += len(self)
        if end is None:
            end = len(self)
        elif end < 0:
            end += len(self)
        stack = []  # This stores the positions in the word and sequence that we may need to jump to
        catixes = []  # This records the index of each category match. Due to limitations, this doesn't include categories in optional sequences
        pos = start  # This keeps track of the position in the word, as it doesn't increase linearly
        ix = 0  # This keeps track of the position in the sequence, as it isn't necessarily monotonic
        while ix < len(seq):
            matched = False
            if pos < end:  # Still in the slice
                seg = seq[ix]
                if isinstance(seg, str) and seg in '**?':  # Wildcards
                    # wrange will be reversed since we push to the stack in the reverse order to how we want to check
                    if '?' in seg:  # Non-greedy - advance ltr
                        wrange = reversed(range(pos+1, end))
                    else:  # Greedy - advance rtl
                        wrange = range(pos+1, end)
                    for wpos in wrange:
                        # wpos is valid if wildcard is extended, or if wildcard is unextended but not matching '#'
                        if '**' in seg or '#' not in self[pos:wpos]:
                            stack.append((wpos, ix+1))
                elif isinstance(seg, (str, Cat)):
                    matched = self.match_segment(seg, pos)
                    length = 1
                elif isinstance(seg, list):  # Optional sequence
                    if seg[-1] == '?':  # Non-greedy
                        seg = seg[:-1]
                        stack.append((pos, ix))
                        stack.append((pos, ix+len(seg)))
                    else:  # Greedy
                        stack.append((pos, ix+len(seg)))
                        stack.append((pos, ix))
                    seq = seq[:ix] + seg + seq[ix+1:]
                elif isinstance(seg, tuple) and seg[0].startswith('*'):  # Presently only wildcard repetitions
                    stack.append((pos, ix-1))
                    matched = True
                    length = 0
            if matched:
                if isinstance(seg, Cat):
                    catixes.append(seg.index(self[pos]))
                ix += 1
                pos += length
            elif stack:
                pos, ix = stack.pop()
            else:
                break  # Total match failure
        else:
            return (True, pos-start) + ((catixes,) if return_cats else ())
        return (False, 0) + (([],) if return_cats else ())
    
    def rmatch_pattern(self, seq, start=None, end=None, return_cats=False):
        '''Match a pattern sequence to the word.
        
        Return if the sequence matches the start of the given slice of the word, and how much of the word was matched.
        
        Arguments:
            seq   -- the sequence being matched
            start -- the index to begin matching from
            end   -- the index to match until
        
        Returns a tuple.
        '''
        # Interpret start and end according to slice notation
        if start is None:
            start = 0
        elif start < 0:
            start += len(self)
        if end is None:
            end = len(self)
        elif end < 0:
            end += len(self)
        stack = []  # This stores the positions in the word and sequence that we may need to jump to
        catixes = []  # This records the index of each category match. Due to limitations, this doesn't include categories in optional sequences
        pos = end - 1  # This keeps track of the position in the word, as it doesn't decrease linearly
        ix = len(seq) - 1  # This keeps track of the position in the sequence, as it isn't necessarily monotonic
        while ix >= 0:
            matched = False
            if pos >= start:  # Still in the slice
                seg = seq[ix]
                if isinstance(seg, str) and seg in '**?':  # Wildcards
                    # wrange will be reversed since we push to the stack in the reverse order to how we want to check
                    if '?' in seg:  # Non-greedy - advance rtl
                        wrange = range(start, pos)
                    else:  # Greedy - advance ltr
                        wrange = reversed(range(start, pos))
                    for wpos in wrange:
                        # Match is valid if wildcard is extended, or if wildcard is unextended but not matching '#'
                        if '**' in seg or '#' not in self[wpos+1:pos+1]:
                            stack.append((wpos, ix-1))
                elif isinstance(seg, (str, Cat)):
                    matched = self.match_segment(seg, pos)
                    length = 1
                elif isinstance(seg, list):  # Optional sequence
                    if seg[-1] == '?':  # Non-greedy
                        seg = seg[:-1]
                        stack.append((pos, ix))
                        stack.append((pos, ix-len(seg)))
                    else:  # Greedy
                        stack.append((pos, ix-len(seg)))
                        stack.append((pos, ix))
                    seq = seq[:ix] + seg + seq[ix+1:]
                elif isinstance(seg, tuple) and seg[0].startswith('*'):  # Presently only wildcard repetitions
                    stack.append((pos, ix+1))
                    matched = True
                    length = 0
            if matched:
                if isinstance(seg, Cat):
                    catixes.append(seg.index(self[pos]))
                ix -= 1
                pos -= length
            elif stack:
                pos, ix = stack.pop()
            else:
                break  # Total match failure
        else:
            return (True, end-pos-1) + ((catixes,) if return_cats else ())
        return (False, 0) + (([],) if return_cats else ())
    
    def match_segment(self, seg, pos=0):
        if seg == '"':  # Ditto mark
            return (self[pos] == self[pos-1])
        elif isinstance(seg, str):  # Grapheme
            return (self[pos] == seg)
        elif isinstance(seg, Cat):  # Category
            return (self[pos] in seg)  # This may change if categories are allowed to contain sequences
        
    def match_env(self, env, pos=0, length=0):  # Test if the env matches the word
        '''Match a sound change environment to the word.
        
        Arguments:
            env -- the environment to be matched (list)
            pos -- the index of the left edge of the target (int)
            length -- the length of the target (int)
        
        Returns a bool
        '''
        env = env.copy()
        tar = self[pos:pos+length]
        for j in range(len(env)):
            for i in reversed(range(len(env[j]))):
                if env[j][i] == '%':
                    env[j][i:i+1] = tar
                if env[j][i] == '<':
                    env[j][i:i+1] = reversed(tar)
        if len(env) == 1:
            return env[0] in self
        else:
            if pos:
                matchleft = self.rmatch_pattern(env[0], 0, pos)[0]
            else:  # At the left edge, which can only be matched by a null env
                matchleft = False if env[0] else True
            matchright = self.match_pattern(env[1], pos+length)[0]
            return matchleft and matchright

Config = namedtuple('Config', 'patterns, counts, constraints, freq, monofreq')

# == Functions == #
def parse_syms(syms, cats=None):
    '''Parse a string using pattern notation.
    
    Arguments:
        syms -- the input string using pattern notation (str)
        cats -- a list of cats to use for interpreting categories (list)
    
    Returns a list
    '''
    if cats is None:
        cats = {}
    else:
        cats = cats.copy()
    if 'graphs' not in cats:
        cats['graphs'] = Cat("'")
    cats['graphs'] += ['**', '*?', '**?']  # Easiest way to have multi-letter symbols parse correctly
    for char in '([{}])':
        syms = syms.replace(char, f' {char} ')
    syms = split(syms, ' ', nesting=(0, '([{', '}])'), minimal=True)
    for i in reversed(range(len(syms))):
        syms[i] = syms[i].replace(' ', '')
        if not syms[i]:
            del syms[i]
        elif syms[i][0] == '(':  # Optional - parse to list
            syms[i] = parse_syms(syms[i].strip('()'), cats)
        elif syms[i][0] == '[':  # Category - parse to Cat
            syms[i] = syms[i].strip('[]')
            if ',' in syms[i]:  # Nonce cat
                syms[i] = Cat(syms[i], cats)
            else:  # Named cat
                syms[i] = cats[syms[i]]
        elif syms[i][0] == '{':  # Numbers - a few types of this
            syms[i] = syms[i].strip('{}')
            if syms[i][0] in '=<>':  # Comparison - parse to tuple
                op = syms[i][0]
                if syms[i][1] == '=' or op == '=':
                    op += '='
                syms[i] = (op, int(syms[i].strip('=<>')))
            elif syms[i].startswith('*'):  # Wildcard repetition - parse to tuple
                syms[i] = (syms[i],)
            else:  # Repetitions - parse to int
                syms[i] = int(syms[i])
        else:  # Text - parse as word
            syms[i:i+1] = parse_word(syms[i], cats['graphs'])
    for i in reversed(range(len(syms))):  # Second pass to evaluate repetitions and ?
        if isinstance(syms[i], int):
            syms[i-1:i+1] = [syms[i-1]]*syms[i]
        elif syms[i] == '?':
            if isinstance(syms[i-1], list) and not isinstance(syms[i-1], Cat):  # Optional
                syms[i-1].append('?')
                del syms[i]
    return syms

def parse_word(word, graphs=None):
    '''Parse a string of graphemes.
    
    Arguments:
        word       -- the word to be parsed (str)
        separator  -- disambiguation character (str)
        polygraphs -- list of polygraphs (list)
    
    Returns a list.
    '''
    # Black magic
    # While test isn't a single character and doesn't begin any polygraph
    #     From i=len(test) to i=1
    #         Does test begin with a valid graph? Single characters are always valid
    #             Add this valid graph to the output
    #             Remove the graph from test, and remove leading instances of separator
    #     End
    test = ''
    if graphs is None:
        graphs = ["'"]
    separator = graphs[0]
    polygraphs = [graph for graph in graphs if len(graph) > 1]
    graphemes = []
    for char in '#'.join(f'.{word}.'.split()).strip('.')+separator:  # Convert all whitespace to a single #
        test += char
        while len(test) > 1 and not any(g.startswith(test) for g in polygraphs):
            for i in reversed(range(1, len(test)+1)):
                if i == 1 or test[:i] in polygraphs:
                    graphemes.append(test[:i])
                    test = test[i:].lstrip(separator)
                    break
    return graphemes

def split(string, sep=None, nesting=None, minimal=False):
    '''String splitting.
    
    Arguments:
        string  -- the string to be split (str)
        sep     -- the character(s) to split on (str)
        nesting -- a tuple of the form (depth, open, close) containing the nesting depth, and opening and closing nesting characters (tuple)
        minimal -- whether or not to perform the minimal number of splits, similar to str.split() with no arguments
    
    Returns a list.
    '''
    if sep is None:
        sep = whitespace
        minimal = True
    result = []
    depth = 0
    while True:
        if minimal and (nesting is None or depth == nesting[0]):
            string = string.lstrip(sep)
        for i in range(len(string)):
            if string[i] in sep and (nesting is None or depth == nesting[0]):
                result.append(string[:i])
                string = string[i+1:]
                break
            elif nesting is not None and string[i] in nesting[1]:
                depth += 1
            elif nesting is not None and string[i] in nesting[2]:
                depth -= 1
        else:
            if not minimal or string != '':
                result.append(string)
            break
    return result

