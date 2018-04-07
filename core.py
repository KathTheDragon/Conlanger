'''Base classes and functions

Exceptions:
    LangException -- Base exception
    FormatError   -- Error for incorrect formatting

Classes:
    Cat    -- represents a category of phonemes
    Word   -- represents a run of text

Functions:
    parse_syms -- parses a string using pattern notation
    split      -- splits a string
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Break out format checking into separate functions
Perhaps adjust Cat.__init__ to allow sequences of graphemes to be stored
After everything, look into using metaclasses in Word
Replace super-disgusting hacky workaround in Word.match_env with something better

=== Features ===
Work on syllabification
- lots of sub-steps here
Something something punctuation

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

# == Decorators == #
# Implements a decorator we can use as a variation on @property, where the value is calculated once and then stored
class lazyproperty(object):
    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value
    
# == Classes == #
class Cat(list):
    '''Represents a category of graphemes.'''
    
    __slots__ = ()
    
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
        s = str(self)
        if "'" not in s:
            return f"Cat('{s}')"
        else:
            return f'Cat("{s}")'
    
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
        graphs      -- a category of graphemes (Cat)
        syllabifier -- a function that syllabifies the input word (Syllabifier)
    
    Methods:
        find          -- find a match of a list using pattern notation to the word
        match_pattern -- match a list using pattern notation to the word
        match_env     -- match a sound change environment to the word
        apply_match   -- apply a single match to the word
        strip         -- remove leading and trailing graphemes
    '''
    
    __slots__ = ('graphs', 'syllabifier')
    
    def __init__(self, lexeme=None, graphs=None, syllabifier=None):
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
        if syllabifier is None:
            syllabifier = def_syllabifier
        self.syllabifier = syllabifier
    
    @lazyproperty
    def syllables(self):
        return self.syllabifier(self)
    
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
        return word.strip(separator+'#').replace('#', ' ')
    
    def __contains__(self, item):
        if isinstance(item, (list, Word)):
            return self.find(item) != -1
        else:
            return list.__contains__(self, item)
    
    def __getitem__(self, item):
        if isinstance(item, slice):
            return Word(list.__getitem__(self, item), self.graphs, self.syllabifier)
        else:
            return list.__getitem__(self, item)
    
    __setitem__ = None
    __delitem__ = None
    
    def __add__(self, other):
        if isinstance(other, Word):
            graphs = self.graphs + other.graphs[1:]
        else:
            graphs = self.graphs
        return Word(list(self) + list(other), graphs, self.syllabifier)
    
    def __mul__(self, other):
        return Word(list(self) * other, self.graphs, self.syllabifier)
    
    def __rmul__(self, other):
        return Word(list(self) * other, self.graphs, self.syllabifier)
    
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
    
    def find(self, sub, start=None, end=None):
        '''Match a sequence using pattern notation to the word.
        
        Arguments:
            sub   -- the list to be found (list)
            start -- the index of the beginning of the range to check (int)
            end   -- the index of the end of the range to check (int)
        
        Returns an int
        '''
        start, end = slice_indices(self, start, end)
        if isinstance(sub, Word):
            sub = sub.strip()  # We want to strip out '#'s at the edge so that this works like finding substrings
        if isinstance(sub[-1], tuple):  # Counting
            matches = 0
            op, count = sub[-1]
            for pos in range(start, end):
                match, length = self.match_pattern(sub[:-1], pos, end)[:2]
                if match:
                    matches += 1
            if eval(f'matches {op} count'):
                return 1
        else:
            for pos in range(start, end):
                match, length = self.match_pattern(sub, pos, end)[:2]
                if match:
                    return pos-start
        return -1
    
    def match_pattern(self, seq, start=None, end=None, step=1, pos=None, ix=None):
        '''Match a pattern sequence to the word.
        
        Return if the sequence matches the end of the given slice of the word, the length of the match, and category indexes.
        
        Arguments:
            seq -- the sequence being matched
            start, end, step -- determine the slice of the word to match within
            pos -- the initial position in the word
            ix  -- the initial position in the sequence
        
        Returns a tuple.
        '''
        start, end = slice_indices(self, start, end)
        end = end-1
        if pos is None:
            pos = start if step > 0 else end
        if ix is None:
            ix = 0 if step > 0 else (len(seq) - 1)
        istep = 1 if step > 0 else -1
        stack = []  # This stores the positions in the word and sequence that we branched at
        catixes = []  # This records the index of each category match. This needs to be redone to cope with non-linearity
        while 0 <= ix < len(seq):
            matched = False
            length = step
            ilength = istep
            if start <= pos <= end:  # Still in the slice
                seg = seq[ix]
                if isinstance(seg, str):
                    if seg.startswith('*'):  # Wildcard
                        wrange = range(pos+step, (end if step > 0 else start)+step, step)
                        if '?' in seg:  # Non-greedy
                            wrange = wrange[::-1]
                        if '**' in seg:  # Extended
                            for wpos in wrange:
                                # wpos is always valid if wildcard is extended
                                stack.append((wpos, ix+istep))
                        else:
                            for wpos in wrange:
                                # otherwise wpos is valid if not matching '#'
                                if '#' not in self[pos:wpos:step]:
                                    stack.append((wpos, ix+istep))
                    elif seg == '"':  # Ditto mark
                        matched = self[pos] == self[pos-1]
                    else:  # Grapheme
                        matched = self[pos] == seg
                elif isinstance(seg, Cat):  # Category
                    if self[pos] in seg:  # This may change if categories are allowed to contain sequences
                        matched = True
                        catixes.append(seg.index(self[pos]))
                elif isinstance(seg, list):  # Optional sequence
                    mode = 'g'
                    jump = len(seg)
                    ilength = 0
                    if seg[-1] == '?':  # Non-greedy
                        seg = seg[:-1]
                        mode = 'ng'
                    if (istep == 1) ^ (mode == 'g'):
                        jump = 0
                        ilength = len(seg)
                    if istep == -1:
                        jump += istep
                        ilength += istep
                    stack.append((pos, ix+jump))
                    seq = seq[:ix] + seg + seq[ix+1:]
                    matched = True
                    length = 0
                elif isinstance(seg, tuple) and seg[0].startswith('*'):  # Presently only wildcard repetitions; slight problem with rtl
                    stack.append((pos, ix-istep))
                    matched = True
                    length = 0
            if matched:
                ix += ilength
                pos += length
            elif stack:  # This segment failed to match, so we jump back to the next branch
                pos, ix = stack.pop()
            else:  # Total match failure
                return False, 0, []
        spos = start if step > 0 else end
        return True, (pos-spos)*step, catixes
        
    def match_env(self, env, pos=0, length=0):  # Test if the env matches the word
        '''Match a sound change environment to the word.
        
        Arguments:
            env    -- the environment to be matched (list)
            pos    -- the index of the left edge of the target (int)
            length -- the length of the target (int)
        
        Returns a bool
        '''
        if isinstance(env, tuple):
            return all(self.match_env(e, pos, length) for e in env)
        env = env.copy()
        tar = self[pos:pos+length]
        for j in range(len(env)):
            env[j] = resolve_target_reference(env[j], tar)
        if len(env) == 0:  # Blank environment
            return True
        elif len(env) == 1:  # Global environment
            return env[0] in self
        else:  # Local environment
            if pos:
                # Hacky thing for now to make wildcard repetitions actually work in the left env
                for i in range(len(env[0])):
                    if isinstance(env[0][i], tuple) and env[0][i][0].startswith('*'):
                        env[0][i-1:i+1] = reversed(env[0][i-1:i+1])
                    elif isinstance(env[0][i], list):
                        for j in range(len(env[0][i])):
                            if isinstance(env[0][i][j], tuple) and env[0][i][j][0].startswith('*'):
                                env[0][i][j-1:j+1] = reversed(env[0][i][j-1:j+1])
                matchleft = self.match_pattern(env[0], 0, pos, -1)[0]
            else:  # At the left edge, which can only be matched by a null env
                matchleft = False if env[0] else True
            matchright = self.match_pattern(env[1], pos+length)[0]
            return matchleft and matchright
    
    def apply_match(self, match, rep):
        '''Apply a replacement to a word
        
        Arguments:
            match -- the match to be used
            rep   -- the replacement to be used
            word  -- the word to be changed
        
        Returns a Word.
        '''
        pos, length, catixes = match[:3]
        tar = self[pos:pos+length]
        if isinstance(rep, list):  # Replacement
            rep = rep.copy()
            # Deal with categories and ditto marks
            ix = 0
            for i in range(len(rep)):
                if isinstance(rep[i], Cat):
                    rep[i] = rep[i][catixes[ix] % len(rep[i])]
                    ix = (ix + 1) % len(catixes)
                elif rep[i] == '"':
                    rep[i] = rep[i-1]
            # Deal with target references
            rep = resolve_target_reference(rep, tar)
            word = Word(list(self[:pos]) + rep + list(self[pos+length:]), self.graphs, self.syllabifier)
        else:  # Movement
            if isinstance(rep[1], list):  # Environment
                mode, envs = rep
                matches = []
                for wpos in range(1, len(self)):  # Find all matches
                    if any(self.match_env(env, wpos) for env in envs):
                        if mode == 'move' and wpos >= pos + length:  # We'll need to adjust the matches down
                            wpos -= length
                        matches.append(wpos)
            else:  # Indices
                mode, matches = rep[0:2]
            if mode == 'move':  # Move - delete original tar
                word = self[:pos] + self[pos+length:]
            else:
                word = self[:]
            for match in sorted(matches, reverse=True):
                word = word[:match] + tar + word[match:]
        return word

class Syllabifier:
    __slots__ = ('peaks',)
    
    def __init__(self, rules, peak_cats, cats):
        self.peaks = Cat(peak_cats, cats)
    
    def __call__(self, word):
        pass

# == Functions == #
def resolve_target_reference(seq, tar):
    seq = seq.copy()
    for i in reversed(range(len(seq))):
        if seq[i] == '%':  # Target copying
            seq[i:i+1] = tar
        elif seq[i] == '<':  # Target reversal/metathesis
            seq[i:i+1] = reversed(tar)
    return seq

def slice_indices(iter, start=None, end=None):
    '''Calculate absolute indices from slice indices on an iterable.
    
    Arguments:
        iter  -- the iterable being sliced
        start -- the index of the start of the slice
        end   -- the index of the end of the slice
    
    Returns a tuple of 2 ints.
    '''
    if start is None:
        start = 0
    elif start < 0:
        start += len(iter)
    if end is None:
        end = len(iter)
    elif end < 0:
        end += len(iter)
    return start, end

def_syllabifier = lambda s: None  # Temporary

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
            syms[i] = parse_syms(syms[i][1:-1], cats)
        elif syms[i][0] == '[':  # Category - parse to Cat
            syms[i] = syms[i][1:-1]
            if ',' in syms[i]:  # Nonce cat
                syms[i] = Cat(syms[i], cats)
            else:  # Named cat
                syms[i] = cats[syms[i]]
        elif syms[i][0] == '{':  # Numbers - a few types of this
            syms[i] = syms[i][1:-1]
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

def parse_cats(cats):
    '''Parses a set of categories.
    
    Arguments:
        cats -- the set of categories to be parsed (str)
    
    Returns a dict.
    '''
    if isinstance(cats, str):
        cats = cats.splitlines()
    _cats = {}
    if isinstance(cats, list):
        for cat in cats:
            if '=' in cat:
                name, values = cat.split('=')
                name, values = name.strip(), values.strip()
                if name != '' and values != '':
                    _cats[name] = Cat(values, cats)
    elif isinstance(cats, dict):
        for cat in cats:
            if cat == '' or not cats[cat]:
                continue
            elif isinstance(cats[cat], Cat):
                _cats[cat] = cats[cat]
            else:
                _cats[cat] = Cat(cats[cat])  # meow
    for cat in list(_cats):  # Discard blank categories
        if not _cats[cat]:
            del _cats[cat]
    return _cats

def parse_word(word, graphs=None):
    '''Parse a string of graphemes.
    
    Arguments:
        word   -- the word to be parsed (str)
        graphs -- category of graphemes (Cat)
    
    Returns a list.
    '''
    # Black magic
    # While test isn't a single character and doesn't begin any polygraph
    #     From i=len(test) to i=1
    #         Does test begin with a valid graph? Single characters are always valid
    #             Add this valid graph to the output
    #             Remove the graph from test, and remove leading instances of separator
    test = ''
    if graphs is None:
        graphs = ["'"]
    separator = graphs[0]
    polygraphs = [graph for graph in graphs if len(graph) > 1]
    graphemes = []
    for char in '#'.join(f'.{word}.'.split()).strip('.')+separator:  # Convert all whitespace to a single #
        test += char
        while len(test) > 1 and not any(graph.startswith(test) for graph in polygraphs):
            for i in reversed(range(1, len(test)+1)):
                if i == 1 or test[:i] in polygraphs:
                    graphemes.append(test[:i])
                    test = test[i:].lstrip(separator)
                    break
    return graphemes

def split(string, sep=None, nesting=None, minimal=False):
    '''Nesting-aware string splitting.
    
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

