'''Base classes and functions

Exceptions:
    LangException -- Base exception
    FormatError   -- Error for incorrect formatting

Classes:
    Cat         -- represents a category of phonemes
    Word        -- represents a run of text
    Syllabifier -- syllabifies a word based on phonotactics

Functions:
    resolveTargetRef -- substitutes a target into a pattern list
    sliceIndices -- returns absolute indices for slice indices on an iterable
    parseCats    -- parses a set of categories
    parseWord    -- parses a string of graphemes
    split         -- splits a string
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Perhaps adjust Cat.__init__ to allow sequences of graphemes to be stored

=== Features ===
Something something punctuation
Hijack global environments with no pattern to test for position in word

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
        self.funcname = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj, self.funcname, value)
        return value

# == Classes == #
@dataclass
class Token:
    type: str
    value: str
    linenum: int
    column: int

    def __iter__(self):
        yield self.type
        yield self.value

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
        yield from self.values

    def __contains__(self, item):
        return item in self.values

    def __and__(self, cat):
        return Cat([value for value in self if value in cat])

    def __add__(self, cat):
        return Cat(self.values + list(cat))

    def __iadd__(self, cat):
        return NotImplemented

    def __sub__(self, cat):
        return Cat([value for value in self if value not in cat])

    def __le__(self, cat):
        return all(value in cat for value in self)

    def __lt__(self, cat):
        return self <= cat and not (self >= cat)

    def __ge__(self, cat):
        return all(value in self for value in cat)

    def __gt__(self, cat):
        return self >= cat and not (self <= cat)

    def index(self, item):
        return self.values.index(item)

    @staticmethod
    def make(string, cats=None, name=None):
        if not (string.startswith('[') and string.endswith(']')):
            raise FormatError(f'invalid category: {string}')
        cat = string[1:-1]
        if ',' in cat:  # Nonce category
            if cat.endswith(','):
                if cat.count(',') == 1:
                    cat = cat[:-1]
                else:
                    raise FormatError(f'invalid category values: {cat}')
            values = []
            for value in re.split(r', ?', cat):
                if not value:
                    raise FormatError(f'invalid category values: {cat}')
                elif value.startswith('[') and value.endswith(']'):
                    values.extend(Cat.make(value, cats))
                elif ' ' in value or '[' in value or ']' in value:
                    raise FormatError(f'invalid category value: {value}')
                else:
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
        matchPattern -- match a list using pattern notation to the word
        matchEnv     -- match a sound change environment to the word
        applyMatch   -- apply a single match to the word
        strip         -- remove leading and trailing graphemes
    '''
    phones: list = field(init=False)
    lexeme: InitVar[str] = ''
    graphs: Cat = None
    separator: str = ''
    syllabifier: 'Syllabifier' = None

    def __post_init__(self, lexeme):
        if isinstance(lexeme, str):
            self.phones = parseWord(f' {lexeme} ', self.graphs, self.separator)
        else:
            phones = []
            for i, phone in enumerate(lexeme):
                if not phone:
                    continue
                elif not (phone == '#' and phones and phones[-1] == '#'):
                    phones.append(phone)
            self.phones = phones

    @memoisedproperty
    def syllables(self):
        return self.syllabifier(self)

    def __repr__(self):
        return f'Word({str(self)!r})'

    def __str__(self):
        return unparseWord(self, self.graphs, self.separator)

    def __len__(self):
        return len(self.phones)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Word(self.phones[item], self.graphs, self.separator, self.syllabifier)
        else:
            return self.phones[item]

    def __iter__(self):
        yield from self.phones

    def __contains__(self, item):
        from .sce import GlobalEnvironment
        if isinstance(item, (list, Word)):
            return self.find(item) != -1
        elif isinstance(item, GlobalEnvironment):
            pattern, indices = item
            if indices is None:
                return self.find(pattern) != -1
            return any(self.matchPattern(pattern, index)[0] for index in indices)
        else:
            return item in self.phones

    def __add__(self, other):
        graphs = self.graphs
        if isinstance(other, Word):
            if graphs == other.graphs:
                pass
            elif graphs is None:
                graphs = other.graphs
            elif other.graphs is not None:
                graphs = graphs + other.graphs[1:]
            other = other.phones
            separator = self.separator or other.separator
        elif isinstance(other, str):
            other = parseWord(other, graphs)
            separator = self.separator
        return Word(self.phones + other, graphs, separator, self.syllabifier)

    def __radd__(self, other):
        graphs = self.graphs
        if isinstance(other, Word):
            if graphs == other.graphs:
                pass
            elif graphs is None:
                graphs = other.graphs
            elif other.graphs is not None:
                graphs = graphs + other.graphs[1:]
            other = other.phones
            separator = self.separator or other.separator
        elif isinstance(other, str):
            other = parseWord(other, graphs)
            separator = self.separator
        return Word(other + self.phones, graphs, separator, self.syllabifier)

    def __mul__(self, other):
        return Word(self.phones * other, self.graphs, self.separator, self.syllabifier)

    def __rmul__(self, other):
        return Word(self.phones * other, self.graphs, self.separator, self.syllabifier)

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
                if start is None:
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
        from ._pattern import parsePattern
        start, end = sliceIndices(self, start, end)
        if isinstance(sub, Word):
            sub = parsePattern(sub)
        if sub and sub[-1].type == 'Comparison':  # Counting
            matches = 0
            op, count = sub[-1].operation, sub[-1].value
            for pos in range(start, end):
                match = self.matchPattern(sub[:-1], pos, end)[0]
                if match:
                    matches += 1
            if eval(f'matches {op} count'):
                return 1
        else:
            for pos in range(start, end):
                match = self.matchPattern(sub, pos, end)[0]
                if match:
                    return pos
        return -1

    def matchPattern(self, pattern, start=None, end=None, step=1):
        '''Match a pattern sequence to the word.

        Return if the sequence matches the end of the given slice of the word, the far end of the match, and category indexes.

        Arguments:
            pattern -- the sequence being matched
            start, end, step -- determine the slice of the word to match within
            stack -- used to pass stack references into an optional segment

        Returns a tuple.
        '''
        from ._pattern import matchPattern
        start, end = sliceIndices(self, start, end)
        return matchPattern(self, pattern, start, end, step)

    def matchEnv(self, environment, pos=0, rpos=0):  # Test if the env matches the word
        '''Match a sound change environment to the word.

        Arguments:
            environment -- the environment to be matched (list)
            pos, rpos   -- the slice of the word giving the target (int, int)

        Returns a bool
        '''
        for env in environment:
            if env is None:  # Blank environment
                continue
            env = env.resolveTargetRef(self[pos:rpos])
            if not env.match(self, pos, rpos):
                return False
        return True

    def applyMatch(self, match, rep):
        '''Apply a replacement to a word

        Arguments:
            match -- the match to be used
            rep   -- the replacement to be used
            word  -- the word to be changed

        Returns a Word.
        '''
        from .sce import Replacement, LocalEnvironment, GlobalEnvironment
        pos, rpos, catixes = match[:3]
        if not rep:
            return self[:pos] + self[rpos:]
        target = self[pos:rpos]
        if isinstance(rep, Replacement):
            _rep = []
            ix = 0
            for element in rep.resolveTargetRef(target).pattern:
                if element.type == 'Grapheme':
                    _rep.append(element.grapheme)
                elif element.type == 'Category':
                    if not catixes:
                        raise RuleError('replacement contains a category but target did not')
                    cat = element.cat
                    _rep.append(cat[catixes[ix] % len(cat)])
                    ix = (ix + 1) % len(catixes)
                elif element.type == 'Ditto':
                    _rep.append(rep[-1] if _rep else self[pos-1])
                else:
                    _rep.append('')
            return self[:pos] + _rep + self[rpos:]
        elif isinstance(rep, tuple):  # Copy/Move
            mode, envs = rep
            matches = []
            for env in envs:  # Each anded environment contributes destinations
                if isinstance(env, LocalEnvironment):
                    env = env.resolveTargetRef(target)
                    for wpos in range(1, len(self)):  # Find all matches
                        if env.match(self, wpos, wpos):
                            if mode == 'move' and wpos >= rpos:  # We'll need to adjust the matches down
                                wpos -= rpos-pos
                            matches.append(wpos)
                elif isinstance(env, GlobalEnvironment):  # Indices
                    if env.pattern:
                        raise RuleError(f'global environment as destination must have no pattern: {rep}')
                    matches.extend(env.indices)
                else:
                    raise RuleError(f'unknown environment: {rep}')
            if mode == 'move':  # Move - delete original target
                word = self[:pos] + self[rpos:]
            for match in sorted(matches, reverse=True):
                word = word[:match] + target + word[match:]
            return word
        else:
            raise RuleError(f'invalid replacement: {rep}')

@dataclass
class Syllabifier:
    rules: list

    def __init__(self, cats, onsets=(), nuclei=(), codas=(), margins=(), constraints=()):
        from ._pattern import parsePatterns
        onsets = parsePatterns(onsets)
        nuclei = parsePatterns(nuclei)
        codas = parsePatterns(codas)
        margins = parsePatterns(margins)
        constraints = parsePatterns(constraints)
        self.rules = []
        # Generate medial rules - coda + onset + nucleus
        rules = self.getNonFinals(onsets, nuclei, codas)
        self.rules.extend(r[:2] for r in sorted(rules, key=lambda r: r[2]))
        # Generate final rules - coda + right margin
        rules = self.getFinals(codas, margins)
        self.rules.extend(r[:2] for r in sorted(rules, key=lambda r: r[2]))
        # Generate initial rules - left margin + onset + nucleus
        rules = self.getNonFinals(onsets, nuclei, margins)
        self.rules.extend(r[:2] for r in sorted(rules, key=lambda r: r[2]))
        self.rules = [rule for rule in self.rules if self.checkValid(rule[0], constraints)]

    @staticmethod
    def getNonFinals(onsets, nuclei, codas):
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
    def getFinals(codas, margins):
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
    def checkValid(rule, constraints):
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
                match, rpos = word.matchPattern(rule, pos)[:2]
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
def resolveTargetRef(pattern, target):
    _pattern = []
    for token in pattern:
        if token.type == 'TargetRef':
            _pattern.extend(token.resolveTarget(target))
        else:
            _pattern.append(token)
    return _pattern

def sliceIndices(iter, start=None, end=None):
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

def parseCats(cats, initialcats=None):
    '''Parses a set of categories.

    Arguments:
        cats -- the set of categories to be parsed (str)
        initialcats -- prior categories (dict)

    Returns a dict.
    '''
    if initialcats is None:
        _cats = {}
    else:
        _cats = initialcats.copy()
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

def parseWord(word, graphs=None):
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
    word = WHITESPACE_REGEX.sub('#', word)
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

def unparseWord(wordin, graphs=None):
    word = test = ''
    if graphs is None:
        separator = ''
        polygraphs = []
    else:
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

def partition(sequence, *, sep=None, sepfunc=None, yieldsep=False):
    if sep is None == sepfunc is None:
        raise ValueError('exactly one of sep and sepfunc must be given')
    if sep is not None:
        sepfunc = lambda item: item == sep
    i = 0
    for j, item in enumerate(sequence):
        if sepfunc(item):
            if yieldsep:
                yield (sequence[i:j], sequence[j])
            else:
                yield sequence[i:j]
            i = j+1
    if yieldsep:
        yield sequence[i:], None
    else:
        yield sequence[i:]

def partitionTokens(tokens, sep=None, yieldsep=True):
    yield from partition(tokens, sepfunc=(lambda token: token.type == sep), yieldsep=yieldsep)
