'''Base classes and functions

Exceptions:
    LangException -- Base exception
    FormatError   -- Error for incorrect formatting

Classes:
    Cat         -- represents a category of phonemes
    Word        -- represents a run of text
    Syllabifier -- syllabifies a word based on phonotactics

Functions:
    resolve_target_reference -- substitutes a target into a pattern list
    slice_indices -- returns absolute indices for slice indices on an iterable
    parse_cats    -- parses a set of categories
    parse_word    -- parses a string of graphemes
    split         -- splits a string
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Perhaps adjust Cat.__init__ to allow sequences of graphemes to be stored
Regex parser

=== Features ===
Something something punctuation
Escape sequences

=== Style ===
Consider where to raise/handle exceptions
Go over docstrings
'''

import re
from dataclasses import dataclass, field, InitVar

# == Exceptions == #
class LangException(Exception):
    '''Base class for exceptions in this package'''

class FormatError(LangException):
    '''Exception raised for errors in formatting objects.'''

class RuleError(LangException):
    '''Exception raised for errors when running rules.'''

class CompilerError(LangException):
    '''Exception raised for errors when compiling rules.'''
    def __init__(self, error, value, linenum, column):
        super().__init__(f'{error}: `{value}` @ {linenum}:{column}')

class TokenError(CompilerError):
    '''Exception raised for errors involving tokens.'''
    def __init__(self, error, token):
        super().__init__(error, token.value, token.linenum, token.column)

# == Decorators == #
# Implements a decorator we can use as a variation on @property, where the value is calculated once and then stored
class memoisedproperty(object):
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
@dataclass
class Token:
    type: str
    value: str
    linenum: int
    column: int

    def __iter__(self):
        return iter((self.type, self.value))

@dataclass
class Cat:
    '''Represents a category of graphemes.'''
    values: list
    name: str = field(default=None, compare=False)

    def __str__(self):
        return f'[{", ".join(self)}]'

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __contains__(self, item):
        return item in self.values

    def __and__(self, cat):
        return Cat([value for value in self if value in cat])

    def __add__(self, cat):
        return Cat(list.__add__(self, cat))

    def __iadd__(self, cat):
        return NotImplemented

    def __sub__(self, cat):
        return Cat([value for value in self if value not in cat])

    def __le__(self, cat):
        return all(value in cat for value in self)

    def __lt__(self, cat):
        return self <= cat and self != cat

    @staticmethod
    def make(string, cats=None, name=None):
        if not (string.startswith('[') and string.endswith(']')):
            raise FormatError(f'invalid category: {string}')
        cat = string[1:-1]
        if ',' in cat:  # Nonce category
            if cat.endswith(','):
                cat = cat[:-1]
            values = []
            for value in re.split(r', ?', cat):
                if not value:
                    raise FormatError(f'invalid category values: {cat}')
                elif value.startswith('[') and value.endswith(']'):
                    values.extend(Cat.make(value))
                else:
                    if ' ' in value or '[' in value or ']' in value:
                        raise FormatError(f'invalid category value: {value}')
                    values.append(value)
            return Cat(values, name)
        else:  # Named category
            if cats is not None and cat in cats:
                return cats[cat]
            else:
                raise FormatError(f'invalid category name: {cat}')

@dataclass
class Word:
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
    phones: list = field(init=False)
    lexeme: InitVar[str] = ''
    graphs: Cat = None
    syllabifier: 'Syllabifier' = None

    def __post_init__(self, lexeme):
        if isinstance(lexeme, str):
            self.phones = parse_word(f' {lexeme} ', graphs)
        else:
            phones = []
            for i, phone in enumerate(lexeme):
                if not phone:
                    continue
                elif not (phone == '#' and phones and phones[-1] == '#'):
                    phones.append(phone)

    @memoisedproperty
    def syllables(self):
        return self.syllabifier(self)

    def __repr__(self):
        return f'Word({str(self)!r})'

    def __str__(self):
        return unparse_word(self, self.graphs)

    def __len__(self):
        return len(self.phones)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Word(list.__getitem__(self, item), self.graphs, self.syllabifier)
        else:
            return list.__getitem__(self, item)

    def __iter__(self):
        return iter(self.phones)

    def __contains__(self, item):
        if isinstance(item, (list, Word)):
            return self.find(item) != -1
        elif isinstance(item, tuple) and isinstance(item[0], (list, Word)):
            return any(self.match_pattern(item[0], index)[0] for index in item[1])
        else:
            return list.__contains__(self, item)

    def __add__(self, other):
        graphs = self.graphs
        if isinstance(other, Word):
            if graphs is None:
                graphs = other.graphs
            elif other.graphs is not None:
                graphs += other.graphs[1:]
            other = other.phones
        elif isinstance(other, str):
            other = parse_word(other, graphs)
        return Word(self.phones + other, graphs, self.syllabifier)

    def __mul__(self, other):
        return Word(self.phones * other, self.graphs, self.syllabifier)

    def __rmul__(self, other):
        return Word(self.phones * other, self.graphs, self.syllabifier)

    def __iadd__(*args):
        return NotImplemented

    def __imul__(*args):
        return NotImplemented

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
        from ._pattern import parse_pattern
        start, end = slice_indices(self, start, end)
        if isinstance(sub, Word):
            sub = parse_pattern(sub)
        if sub and sub[-1].type == 'Comparison':  # Counting
            matches = 0
            op, count = sub[-1].operation, sub[-1].value
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
            stack -- used to pass stack references into an optional segment

        Returns a tuple.
        '''
        from ._pattern import match_pattern
        start, end = slice_indices(self, start, end)
        return match_pattern(self, pattern, start, end, step)

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
            rep = resolve_target_reference(rep, tar)
            # Resolve tokens
            ix = 0
            for i, token in enumerate(rep):
                if token.type == 'Grapheme':
                    rep[i] = token.grapheme
                elif token.type == 'Category':
                    if not catixes:
                        raise RuleError('replacement contains a category but target did not')
                    cat = token.cat
                    rep[i] = cat[catixes[ix] % len(cat)]
                    ix = (ix + 1) % len(catixes)
                elif token.type == 'Ditto':
                    rep[i] = rep[i-1] if i != 0 else self[pos-1]
                else:
                    rep[i] = ''
            word = Word(list(self[:pos]) + rep + list(self[rpos:]), self.graphs, self.syllabifier)
        else:  # Movement
            if isinstance(rep[1], list):  # Environment
                mode, envs = rep
                matches = []
                for wpos in range(1, len(self)):  # Find all matches
                    if any(self.match_env(env, wpos, wpos) for env in envs):
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

@dataclass
class Syllabifier:
    rules: list

    def __init__(self, cats, onsets=(), nuclei=(), codas=(), margins=(), constraints=()):
        from ._pattern import parse_patterns
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
def resolve_target_reference(pattern, tar):
    pattern = pattern.copy()
    for i, token in reversed(list(enumerate(pattern))):
        if token.type == 'TargetRef':
            pattern[i:i+1] = token.resolve_target(tar)
    return pattern

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
                    cat = Cat.make(f'[{values}]', _cats, name)
                    exec(f'_cats[name] {op} cat')
                    if not _cats[name]:
                        del _cats[name]
    elif isinstance(cats, dict):
        for key, value in cats.items():
            if key == '' or not value:
                continue
            elif isinstance(value, Cat):
                _cats[key] = value
            elif isinstance(value, list):
                _cats[key] = Cat(value, key)
            else:
                _cats[key] = Cat.make(value, _cats, key)  # meow
    for cat in list(_cats):  # Discard blank categories
        if not _cats[cat]:
            del _cats[cat]
    return _cats

WHITESPACE_REGEX = re.compile(r'\s+')

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
    word = WHITESPACE_REGEX.sub('#')
    if graphs is None:
        return list(word)
    separator = graphs[0]
    polygraphs = [graph for graph in graphs if len(graph) > 1]
    if not polygraphs:
        return list(word.replace(separator, ''))
    graphemes = []
    test = ''
    for char in word+separator:  # Convert all whitespace to a single #
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
        separator = ''
        word = ''.join(wordin)
        wordin = []
    separator = graphs[0]
    polygraphs = [graph for graph in graphs if len(graph) > 1]
    if not polygraphs:
        word = ''.join(wordin)
        wordin = []
    for graph in wordin:
        if not any(graph in poly and graph != poly for poly in polygraphs if graph != poly):  # If not a strict substring of any polygraph
            test = ''  # Can't ever be ambiguous
        elif not test:
            test = graph  # Nothing earlier to be ambiguous with
        else:
            test += graph
            if any(poly in test and graph != poly or poly in test[:-1] for poly in polygraphs):  # If test contains a polygraph
                word += separator  # Ambiguous, so add the separator
                test = graph
            elif not any(test in poly for poly in polygraphs):
                test = test[1:]  # Could still be ambiguous with something later
        word += graph
    return word.strip(separator+'#').replace('#', ' ')

def split(string, sep=None, nesting=None, minimal=False):
    '''Nesting-aware string splitting.

    Arguments:
        string  -- the string to be split (str)
        sep     -- the character(s) to split on (str)
        nesting -- a tuple of the form (depth, open, close) containing the nesting depth, and opening and closing nesting characters (tuple)
        minimal -- whether or not to perform the minimal number of splits, similar to str.split() with no arguments

    Returns a list.
    '''
    from string import whitespace
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
