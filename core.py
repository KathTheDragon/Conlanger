'''Base classes and functions

Exceptions:
    LangException -- Base exception
    FormatError   -- Error for incorrect formatting

Classes:
    Cat              -- represents a category of phonemes
    Word             -- represents a run of text
    RulesSyllabifier -- syllabifies a word based on a set of rules
    PhonoSyllabifier -- syllabifies a word based on phonotactics

Functions:
    resolve_target_reference -- substitutes a target into a pattern list
    slice_indices  -- returns absolute indices for slice indices on an iterable
    parse_patterns -- parses a set of patterns using pattern notation
    parse_pattern  -- parses a string using pattern notation
    parse_cats     -- parses a set of categories
    parse_word     -- parses a string of graphemes
    split          -- splits a string
''''''
==================================== To-do ====================================
=== Bug-fixes ===
catixes in Word.match_pattern should be redone to cope with non-linearity

=== Implementation ===
Perhaps adjust Cat.__init__ to allow sequences of graphemes to be stored
Replace super-disgusting hacky wildcard repetition workaround in Word.match_pattern with something better
- How though

=== Features ===
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

class RuleError(LangException):
    '''Exception raised for errors when running rules.'''

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
    
    __slots__ = ('name',)
    
    def __init__(self, values=None, cats=None, name=None):
        '''Constructor for Cat.
        
        Arguments:
            values -- the values in the category (str, list)
            cats   -- dictionary of categories (dict)
            name   -- optional name for the category
        '''
        self.name = name
        _values = []
        if values is None:
            values = []
        elif isinstance(values, str):  # We want an iterable with each value as an element
            values = split(values, ',', minimal=True)
        for value in values:
            value = value.strip()
            if isinstance(value, Cat):  # Another category
                _values.extend(value)
            elif value[0] == '[':
                value = value.strip('[]')
                if cats is not None and value in cats:
                    _values.extend(cats[value])
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
    
    def __iadd__(self, cat):
        return NotImplemented
    
    def __sub__(self, cat):
        return Cat(value for value in self if value not in cat)
    
    def __le__(self, cat):
        return all(value in cat for value in self)

class Word(list):
    '''Represents a word as a list of graphemes.
    
    Instance variables:
        graphs      -- a category of graphemes (Cat)
        syllabifier -- a function that syllabifies the input word (RulesSyllabifier)
    
    Methods:
        find          -- find a match of a list using pattern notation to the word
        match_pattern -- match a list using pattern notation to the word
        match_env     -- match a sound change environment to the word
        apply_match   -- apply a single match to the word
        strip         -- remove leading and trailing graphemes
    '''
    
    __slots__ = ('graphs', 'syllabifier', '_syllables')
    
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
        self.syllabifier = syllabifier
        self._syllables = None
    
    @property
    def syllables(self):
        if self._syllables is None and self.syllabifier is not None:
            self._syllables = self.syllabifier(self)
        return self._syllables
    
    def __repr__(self):
        word = str(self)
        if "'" in word:
            return f'Word("{word}")'
        else:
            return f"Word('{word}')"
    
    def __str__(self):
        word = unparse_word(self, self.graphs)
        return word.strip(self.graphs[0]+'#').replace('#', ' ')
    
    def __contains__(self, item):
        if isinstance(item, (list, Word)):
            return self.find(item) != -1
        elif isinstance(item, tuple) and isinstance(item[0], (list, Word)):
            return any(self.match_pattern(item[0], index)[0] for index in item[1])
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
        graphs = self.graphs
        if isinstance(other, Word):
            graphs += other.graphs[1:]
            other = list(other)
        elif isinstance(other, str):
            other = parse_word(other, graphs)
        return Word(list(self) + other, graphs, self.syllabifier)
    
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
        start = end = None
        for i, char in enumerate(self):
            if char not in chars:
                if start is not None:
                    start = i
                if self[i+1] in chars:
                    end = i+1
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
        if sub and isinstance(sub[-1], tuple):  # Counting
            matches = 0
            op, count = sub[-1]
            for pos in range(start, end):
                match = self.match_pattern(sub[:-1], pos, end)[0]
                if match:
                    matches += 1
            if eval(f'matches {op} count'):
                return 1
        else:
            for pos in range(start, end):
                match = self.match_pattern(sub, pos, end)[0]
                if match:
                    return pos
        return -1
    
    def match_pattern(self, pattern, start=None, end=None, step=1):
        '''Match a pattern sequence to the word.
        
        Return if the sequence matches the end of the given slice of the word, the far end of the match, and category indexes.
        
        Arguments:
            pattern -- the sequence being matched
            start, end, step -- determine the slice of the word to match within
        
        Returns a tuple.
        '''
        start, end = slice_indices(self, start, end)
        end = end-1
        pos = start if step > 0 else end
        ix = 0 if step > 0 else (len(pattern)-1)
        istep = 1 if step > 0 else -1
        stack = []  # This stores the positions in the word and sequence that we branched at
        catixes = []  # This records the index of each category match. This needs to be redone to cope with non-linearity
        # Hacky thing for now to make wildcard repetitions actually work in rtl
        pattern = pattern.copy()
        if step < 0:
            for i, token in enumerate(pattern):
                if isinstance(token, tuple) and token[0].startswith('*'):
                    pattern[i-1:i+1] = reversed(pattern[i-1:i+1])
        while 0 <= ix < len(pattern):
            matched = False
            length = step
            ilength = istep
            if start <= pos <= end:  # Still in the slice
                token = pattern[ix]
                if isinstance(token, str):
                    if token.startswith('*'):  # Wildcard
                        wrange = range(pos+step, (end if step > 0 else start)+step, step)
                        if '?' in token:  # Non-greedy
                            wrange = wrange[::-1]
                        if '**' in token:  # Extended
                            for wpos in wrange:
                                # wpos is always valid if wildcard is extended
                                stack.append((wpos, ix+istep))
                        else:
                            for wpos in wrange:
                                # otherwise wpos is valid if not matching '#'
                                if '#' not in self[pos:wpos:step]:
                                    stack.append((wpos, ix+istep))
                    elif token == '"':  # Ditto mark
                        matched = self[pos] == self[pos-1]
                    elif token == '$':  # Syllable break
                        matched = (pos in self.syllables)
                        length = 0
                    elif token == '_':  # Null
                        if 0 < ix < len(pattern)-1:
                            pass
                        elif (step > 0)^(ix == 0):  # If _ is the last token
                            matched = self[pos] != '#'
                        else:  # If _ is the first token
                            matched = (0 <= pos-step < len(self)) and self[pos-step] != '#'
                        length = 0
                    else:  # Grapheme
                        matched = self[pos] == token
                elif isinstance(token, Cat):  # Category
                    if self[pos] in token:  # This may change if categories are allowed to contain sequences
                        matched = True
                        catixes.append(token.index(self[pos]))
                elif isinstance(token, list):  # Optional sequence
                    if token[-1] == '?':  # Non-greedy
                        stack.append((pos, ix))
                        pattern[ix] = token[:-1]
                        matched = True
                        length = 0
                    else:
                        stack.append((pos, ix+istep))
                        _start, _end = (pos, None)[::istep]
                        matched, rpos, _catixes = self.match_pattern(token, _start, _end, step)
                        length = rpos-pos
                        if matched:
                            catixes.extend(_catixes)
                elif isinstance(token, tuple) and token[0].startswith('*'):  # Presently only wildcard repetitions
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
        if step < 0:
            pos -= step
        return True, pos, catixes
    
    def match_env(self, env, pos=0, rpos=0):  # Test if the env matches the word
        '''Match a sound change environment to the word.
        
        Arguments:
            env  -- the environment to be matched (list)
            pos  -- the index of the left edge of the target (int)
            rpos -- the index past the right edge of the target (int)
        
        Returns a bool
        '''
        if isinstance(env, tuple):
            return all(self.match_env(e, pos, rpos) for e in env)
        env = env.copy()
        tar = self[pos:rpos]
        if len(env) == 0:  # Blank environment
            return True
        elif len(env) == 1:  # Global environment
            env = env[0]
            if isinstance(env, tuple):
                env = (resolve_target_reference(env[0], tar), env[1])
            else:
                env = resolve_target_reference(env, tar)
            return env in self
        # Local environment
        left, right = resolve_target_reference(env[0], tar), resolve_target_reference(env[1], tar)
        if pos:
            matchleft = self.match_pattern(left, 0, pos, -1)[0]
        else:  # At the left edge, which can only be matched by a null env
            matchleft = False if left else True
        matchright = self.match_pattern(right, rpos)[0]
        return matchleft and matchright
    
    def apply_match(self, match, rep):
        '''Apply a replacement to a word
        
        Arguments:
            match -- the match to be used
            rep   -- the replacement to be used
            word  -- the word to be changed
        
        Returns a Word.
        '''
        pos, rpos, catixes = match[:3]
        tar = self[pos:rpos]
        if isinstance(rep, list):  # Replacement
            rep = rep.copy()
            # Deal with categories and ditto marks
            ix = 0
            for i, token in enumerate(rep):
                if isinstance(token, Cat):
                    if not catixes:
                        raise RuleError('replacement contains a category but target did not')
                    rep[i] = token[catixes[ix] % len(token)]
                    ix = (ix + 1) % len(catixes)
                elif token == '"':
                    rep[i] = rep[i-1] if i != 0 else self[pos-1]
            # Deal with target references
            rep = resolve_target_reference(rep, tar)
            word = Word(list(self[:pos]) + rep + list(self[rpos:]), self.graphs, self.syllabifier)
        else:  # Movement
            if isinstance(rep[1], list):  # Environment
                mode, envs = rep
                matches = []
                for wpos in range(1, len(self)):  # Find all matches
                    if any(self.match_env(env, wpos) for env in envs):
                        if mode == 'move' and wpos >= rpos:  # We'll need to adjust the matches down
                            wpos -= rpos-pos
                        matches.append(wpos)
            else:  # Indices
                mode, matches = rep[0:2]
            if mode == 'move':  # Move - delete original tar
                word = self[:pos] + self[rpos:]
            else:
                word = self[:]
            for match in sorted(matches, reverse=True):
                word = word[:match] + tar + word[match:]
        return word

class Syllabifier:
    slots = ('rules',)
    
    def __init__(self, cats, onsets=(), nuclei=(), codas=(), margins=(), constraints=()):
        onsets = parse_patterns(onsets)
        nuclei = parse_patterns(nuclei)
        codas = parse_patterns(codas)
        margins = parse_patterns(margins)
        constraints = parse_patterns(constraints)
        self.rules = []
        # Generate medial rules - coda + onset + nucleus
        rules = self.get_non_finals(onsets, nuclei, codas)
        self.rules.extend(r[:2] for r in sorted(rules, key=lambda r: r[2]))
        # Generate final rules - coda + right margin
        rules = self.get_finals(codas, margins)
        self.rules.extend(r[:2] for r in sorted(rules, key=lambda r: r[2]))
        # Generate initial rules - left margin + onset + nucleus
        rules = self.get_non_finals(onsets, nuclei, margins)
        self.rules.extend(r[:2] for r in sorted(rules, key=lambda r: r[2]))
        self.rules = [rule for rule in self.rules if self.check_valid(rule[0], constraints)]
    
    @staticmethod
    def get_non_finals(onsets, nuclei, codas):
        rules = []
        for crank, coda in enumerate(codas):
            if coda[-1] == '#':
                continue
            elif coda[-1] == '_':
                coda = coda[:-1]
            for orank, onset in enumerate(onsets):
                if onset[0] == '#':
                    if coda == ['#']:
                        onset = onset[1:]
                    else:
                        continue
                if onset == ['_']:
                    onset = []
                for nrank, nucleus in enumerate(nuclei):
                    if nucleus[0] == '#':
                        if coda == ['#'] and onset == []:
                            nucleus = nucleus[1:]
                        else:
                            continue
                    pattern = coda + onset + nucleus
                    breaks = [len(coda)]
                    if pattern[-1] == '#':
                        breaks.append(len(pattern)-1)
                    rank = crank + orank + nrank
                    rules.append((pattern, breaks, rank))
        return rules
    
    @staticmethod
    def get_finals(codas, margins):
        rules = []
        for mrank, margin in enumerate([margin for margin in margins if margin[-1] == '#']):
            if margin == ['_', '#']:
                margin = ['#']
            for crank, coda in enumerate(codas):
                if coda[-1] == '#':
                    if margin == ['#']:
                        coda = coda[:-1]
                    else:
                        continue
                pattern = coda + margin
                breaks = [0 if coda == ['_'] else len(coda)]
                rank = crank + mrank
                rules.append((pattern, breaks, rank))
        return rules
    
    @staticmethod
    def check_valid(rule, constraints):
        for constraint in constraints:
            for rpos in range(len(rule)-len(constraint)):
                for cpos, ctoken in enumerate(constraint):
                    rtoken = rule[rpos+cpos]
                    if isinstance(rtoken, str) and isinstance(ctoken, str):
                        if rtoken == ctoken:
                            continue
                    elif isinstance(rtoken, str) and isinstance(ctoken, Cat):
                        if rtoken in ctoken:
                            continue
                    elif isinstance(rtoken, Cat) and isinstance(ctoken, Cat):
                        if rtoken <= ctoken:
                            continue
                    break
                else:
                    return False
        return True
    
    def __call__(self, word):
        breaks = []
        # Step through the word
        pos = 0
        while pos < len(word):
            for rule, _breaks in self.rules:
                if rule == ['_', '#'] and pos in breaks:
                    continue
                match, rpos = word.match_pattern(rule, pos)[:2]
                if match:
                    # Compute and add breaks for this pattern
                    for ix in _breaks:
                        # Syllable breaks must be within the word and unique
                        if 0 < pos+ix < len(word) and pos+ix not in breaks:
                            breaks.append(pos+ix)
                    # Step past this match
                    pos = rpos
                    if rule[-1] == '#':
                        pos -= 1
                    break
            else:  # No matches here
                pos += 1
        return tuple(breaks)

# == Functions == #
def resolve_target_reference(seq, tar):
    seq = seq.copy()
    for i, token in reversed(list(enumerate(seq))):
        if token == '%':  # Target copying
            seq[i:i+1] = tar
        elif token == '<':  # Target reversal/metathesis
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

def parse_pattern(pattern, cats=None):
    '''Parse a string using pattern notation.
    
    Arguments:
        pattern -- the input string using pattern notation (str)
        cats    -- a list of cats to use for interpreting categories (list)
    
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
        pattern = pattern.replace(char, f' {char} ')
    pattern = split(pattern, ' ', nesting=(0, '([{', '}])'), minimal=True)
    for i, token in reversed(list(enumerate(pattern))):
        token = token.replace(' ', '')
        if not token:
            del pattern[i]
        elif token[0] == '(':  # Optional - parse to list
            token = parse_pattern(token[1:-1], cats)
            if all(isinstance(sub, list) and not isinstance(sub, Cat) for sub in token):
                pattern[i:i+1] = token
            else:
                pattern[i] = token
        elif token[0] == '[':  # Category - parse to Cat
            token = token[1:-1]
            if ',' in token:  # Nonce cat
                pattern[i] = Cat(token, cats)
            else:  # Named cat
                pattern[i] = cats[token]
        elif token[0] == '{':  # Numbers - a few types of this
            token = token[1:-1]
            if token[0] in '=<>':  # Comparison - parse to tuple
                op = token[0]
                if token[1] == '=' or op == '=':
                    op += '='
                pattern[i] = (op, int(token.strip('=<>')))
            elif token[0] == '*':  # Wildcard repetition - parse to tuple
                pattern[i] = (token,)
            else:  # Repetitions - parse to int
                pattern[i] = int(token)
        else:  # Text - parse as word
            pattern[i:i+1] = parse_word(token, cats['graphs'])
    for i, token in reversed(list(enumerate(pattern))):  # Second pass to evaluate repetitions and ?
        if isinstance(token, int):
            pattern[i-1:i+1] = [pattern[i-1]]*token
        elif token == '?':
            if isinstance(pattern[i-1], list) and not isinstance(pattern[i-1], Cat):  # Optional
                pattern[i-1].append('?')
                del pattern[i]
    return pattern

def parse_cats(cats, initial_cats=None):
    '''Parses a set of categories.
    
    Arguments:
        cats -- the set of categories to be parsed (str)
        initial_cats -- prior categories (dict)
    
    Returns a dict.
    '''
    if initial_cats is None:
        _cats = {}
    else:
        _cats = initial_cats.copy()
    if isinstance(cats, str):
        cats = cats.splitlines()
    if isinstance(cats, list):
        for cat in cats:
            if '=' in cat:
                cop = cat.index('=')
                op = (cat[cop-1] if cat[cop-1] in '+-' else '') + '='
                name, values = cat.split(op)
                name, values = name.strip(), values.strip()
                if name != '' and values != '':
                    exec(f'_cats[name] {op} Cat(values, _cats, name)')
                    if not _cats[name]:
                        del _cats[name]
    elif isinstance(cats, dict):
        for cat in cats:
            if cat == '' or not cats[cat]:
                continue
            elif isinstance(cats[cat], Cat):
                _cats[cat] = cats[cat]
            else:
                _cats[cat] = Cat(cats[cat], _cats, cat)  # meow
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
        return list(word)
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

def unparse_word(wordin, graphs=None):
    word = test = ''
    if graphs is None:
        graphs = ("'",)
    separator = graphs[0]
    polygraphs = [graph for graph in graphs if len(graph) > 1]
    for graph in wordin:
        if not any(graph in poly for poly in polygraphs):
            test = ''  # Can't ever be ambiguous
        elif not test:
            test = graph  # Nothing earlier to be ambiguous with
        else:
            test += graph
            if any(poly in test for poly in polygraphs):
                word += separator  # Ambiguous, so add the separator
                test = graph
            elif not any(test in poly for poly in polygraphs):
                test = test[1:]  # Could still be ambiguous with something later
        word += graph
    return word

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
        for i, char in enumerate(string):
            if char in sep and (nesting is None or depth == nesting[0]):
                result.append(string[:i])
                string = string[i+1:]
                break
            elif nesting is not None and char in nesting[1]:
                depth += 1
            elif nesting is not None and char in nesting[2]:
                depth -= 1
        else:
            if not minimal or string != '':
                result.append(string)
            break
    return result
