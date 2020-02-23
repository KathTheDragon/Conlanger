'''Apply sound changes to a lexicon

Exceptions:
    RuleFailed    -- exception to mark that a rule failed

Classes:
    Rule -- represents a sound change rule

Functions:
    compileRuleset -- compiles a sound change ruleset
    compileRule    -- compiles a sound change rule
    run             -- applies a set of sound change rules to a set of words
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Maybe change >^ and >^? to >> and >>?

=== Features ===
Is it possible to implement a>b>c as notation for a chain shift?
Think about expanding the options for grapheme handling
- diacritics
Allow ~ in tar and rep
Implement more category operations
- intersection
-- feature-style? [+A +B -C] == [A] && [B] && ~[C]
More format conversion metarules?
- !sca2

=== Style ===
Consider where to raise/handle exceptions
Go over docstrings
'''

import logging
import logging.config
import os.path
import re
from contextlib import suppress
from dataclasses import dataclass, InitVar
from .core import LangException, FormatError, RuleError, CompilerError, TokenError, Token, Cat, Word, resolveTargetRef, parseCats, partitionTokens
from ._pattern import tokenise as tokenisePattern, compile as compilePattern

# == Constants == #
MAX_RUNS = 10**3  # Maximum number of times a rule may be repeated
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), 'logging.conf'))
RULE_TOKENS = {
    'EPENTHESIS': r'^\+ ?',
    'DELETION': r'^\- ?',
    'MOVE': r'>\^\?| +>\^\? ',
    'COPY': r'>\^| +>\^ ',
    'REPLACEMENT': r'>| +> ',
    'ENVIRONMENT': r'/| +/ ',
    'EXCEPTION': r'!| +! ',
    'OR': r', ?',
    'AND': r'&| & ',
    'PLACEHOLDER': r'_',
    # 'ADJACENCY': r'~',
    'INDICES': r'@\-?\d+(?:\|\-?\d+)*',
    'SPACE': r' ',
    'UNKNOWN': r'.'
}
RULE_REGEX = re.compile('|'.join(f'(?P<{type}>{regex})' for type, regex in RULE_TOKENS.items()))
METARULES = [
    'block',
    'def',
    'rule',
]
METARULE_TOKENS = {
    'METARULE': fr'^!(?:{"|".join(METARULES)})',
    'COLON': r': ?',
    'NUMBER': r'\d+',
    'IDENTIFIER': r'[a-z_]+',
    'SPACE': r' ',
    'UNKNOWN': r'.',
}
METARULE_REGEX = re.compile('|'.join(f'(?P<{type}>{regex})' for type, regex in METARULE_TOKENS.items()))
FLAGS = [
    'ignore',
    'rtl',
    'ditto',
    'stop',
    'repeat',
    'persist',
    'chance',
]
FLAG_TOKENS = {
    'FLAG': '|'.join(FLAGS),
    'COLON': r': ?',
    'ARGUMENT': r'\d+',
    'NEGATION': r'!',
    'SEPARATOR': r'; ?',
    'UNKNOWN': r'.',
}
FLAG_REGEX = re.compile('|'.join(f'(?P<{type}>{regex})' for type, regex in FLAG_TOKENS.items()))
CATEGORY_TOKENS = {
    'CATEGORY': r'^\w+',
    'OP': r'(?:\+|\-)?=| +(?:\+|\-)?= ',
    'VALUES': r'.+$',  # Might make this part more precise
    'UNKNOWN': r'.',
}
CATEGORY_REGEX = re.compile('|'.join(f'(?P<{type}>{regex})' for type, regex in CATEGORY_TOKENS.items()))

# == Globals == #
logger = None

# == Exceptions == #
class RuleFailed(LangException):
    '''Used to indicate that the rule failed to be applied.'''

# == Classes == #
@dataclass
class IndexedPattern:
    pattern: list
    indices: list = None

    def __str__(self):
        if self.indices is None:
            return str(self.pattern)
        elif not self.pattern:
            return f'@{self.indices}'
        else:
            return f'{self.pattern}@{self.indices}'

    def __iter__(self):
        yield self.pattern
        yield self.indices

    def copy(self):
        cls = self.__class__
        if self.indices is not None:
            return cls(self.pattern.copy(), self.indices.copy())
        else:
            return cls(self.pattern.copy())

@dataclass
class Target(IndexedPattern):
    pass

@dataclass
class Replacement:
    pattern: list

    def __str__(self):
        return str(self.pattern)

    def __iter__(self):
        yield self.pattern

    def copy(self):
        return Replacement(self.pattern.copy())

    def resolveTargetRef(self, target):
        return Replacement(resolveTargetRef(self.pattern, target))

@dataclass
class LocalEnvironment:
    left: list
    right: list

    def __str__(self):
        if self.left and self.right:
            return f'{self.left}_{self.right}'
        elif self.left:
            return f'{self.left}_'
        elif self.right:
            return f'_{self.right}'
        else:
            return '_'

    def __bool__(self):
        return bool(self.left or self.right)

    def __iter__(self):
        yield self.left
        yield self.right

    def copy(self):
        return LocalEnvironment(self.left.copy(), self.right.copy())

    def resolveTargetRef(self, target):
        return LocalEnvironment(resolveTargetRef(self.left, target), resolveTargetRef(self.right, target))

    def match(self, word, pos=0, rpos=0):
        left, right = self
        if pos:
            matchleft = word.matchPattern(left, 0, pos, -1)[0]
        else:  # At the left edge, which can only be matched by a null env
            matchleft = False if left else True
        matchright = word.matchPattern(right, rpos)[0]
        return matchleft and matchright

@dataclass
class GlobalEnvironment(IndexedPattern):
    def __bool__(self):
        return bool(self.pattern or self.indices)

    def resolveTargetRef(self, target):
        if self.indices is not None:
            return GlobalEnvironment(resolveTargetRef(self.pattern, target), self.indices.copy())
        else:
            return GlobalEnvironment(resolveTargetRef(self.pattern, target))

    def match(self, word, pos=0, rpos=0):
        pattern, indices = self
        if indices is None:
            return word.find(pattern) != -1
        else:
            return any(word.matchPattern(pattern, index)[0] for index in indices)

@dataclass
class Flags:
    ignore: int = 0
    ditto: int = 0
    stop: int = 0
    rtl: int = 0
    repeat: int = 1
    persist: int = 1
    chance: int = 100

@dataclass
class Rule:
    '''Class for representing a sound change rule.

    Instance variables:
        rule      -- the rule as a string (str)
        tars      -- target segments (list)
        reps      -- replacement segments (list)
        envs      -- application environments (list)
        excs      -- exception environments (list)
        otherwise -- the rule to apply if an exception is satisfied (Rule)
        flags     -- flags for altering execution (Flags)

    Methods:
        apply       -- apply the rule to a word
        checkMatch -- check if the match is valid
    '''
    tars: list
    reps: list
    envs: list
    excs: list
    otherwise: 'Rule'
    flags: Flags
    rule: str = ''

    def __repr__(self):
        return f"Rule('{self!s}')"

    def __str__(self):
        return self.rule

    def __eq__(self, other):
        return self[1:] == other[1:]

    def __iter__(self):
        yield self.tars
        yield self.reps
        yield self.envs
        yield self.excs
        yield self.otherwise
        yield self.flags
        yield self.rule

    def apply(self, word):
        '''Apply the sound change rule to a single word.

        Arguments:
            word -- the word to which the rule is to be applied (Word)

        Raises RuleFailed if the rule did not apply to the word.
        '''
        logger.debug(f'This rule: `{self}`')
        # Get all target matches, filtered by given indices
        logger.debug('Begin matching targets')
        matches = []
        for i, target in enumerate(self.tars):
            logger.debug(f'> Matching `{target}`')
            if target:
                pattern, indices = target
            else:
                pattern, indices = [], None
            if not pattern:  # All pos's
                logger.debug(f'>> Null target matched all positions in range 1..{len(word)}')
                _matches = [(pos, pos, [], i) for pos in range(1, len(word))]
            else:
                _matches = []
                for pos in range(1, len(word)):  # Find all matches
                    match, rpos, catixes = word.matchPattern(pattern, pos)
                    if match:  # pattern matches at pos
                        logger.debug(f'>> Target matched `{word[pos:rpos]}` at {pos}')
                        _matches.append((pos, rpos, catixes, i))
                if not _matches:
                    logger.debug('>> No matches for this target')
            # Filter only those matches selected by the given indices
            if indices is None:
                matches += _matches
            elif _matches:
                matches += [_matches[ix] for ix in indices if -len(_matches) <= ix < len(_matches)]
        matches.sort()
        logger.debug(f'> Final matches at positions {[match[0] for match in matches]}')
        if not matches:
            logger.debug('No matches')
            raise RuleFailed
        # Filter only those matches that fit the environment - also record the corresponding replacement
        logger.debug('Check matches against environments and exceptions')
        reps = []
        for i in reversed(range(len(matches))):
            logger.debug(f'> Checking match at {matches[i][0]}')
            check = self.checkMatch(matches[i], word)
            if not check:
                logger.debug(f'>> Match at {matches[i][0]} failed')
                del matches[i]
            else:
                # Find the correct replacement
                logger.debug('>> Get replacement for this match')
                rule = self
                for j in range(check-1):
                    rule = rule.otherwise
                _reps = rule.reps
                match = matches[i][3]
                if isinstance(_reps, tuple):  # Copy/move
                    reps.append((_reps[0], _reps[1][match%len(_reps[1])]))
                else:
                    reps.append(_reps[match%len(_reps)])
                logger.debug(f'>>> Found {reps[-1]}')
        if not reps:
            logger.debug('No matches matched environment')
            raise RuleFailed
        reps.reverse()
        matches = sorted(zip(matches, reps), reverse=True)
        # Filter overlaps
        logger.debug('Filter out overlapping matches')
        if self.flags.rtl:
            logger.debug('> Proceeding right-to-left')
            i = 1
            while i < len(matches):
                if matches[i][0][1] > matches[i-1][0][0]:  # Overlap
                    logger.debug(f'>> Match at {matches[i][0][0]} overlaps match at {matches[i-1][0][0]}')
                    del matches[i]
                else:
                    i += 1
        else:
            logger.debug('> Proceeding left-to-right')
            for i in reversed(range(len(matches)-1)):
                if matches[i][0][0] < matches[i+1][0][1]:  # Overlap
                    logger.debug(f'>> Match at {matches[i][0][0]} overlaps match at {matches[i+1][0][0]}')
                    del matches[i]
        logger.debug(f'Applying matches to `{word}`')
        for match, rep in matches:
            logger.debug(f'> Changing `{list(word[match[0]:match[1]])}` to `{rep}` at {match[0]}')
            word = word.applyMatch(match, rep)
        return word

    def checkMatch(self, match, word):
        pos, rpos = match[:2]
        if any(word.matchEnv(exc, pos, rpos) for exc in self.excs):  # If there are exceptions, does any match?
            logger.debug('>> Matched an exception, check the "else" rule')
        elif any(word.matchEnv(env, pos, rpos) for env in self.envs):  # Does any environment match?
            logger.debug('>> Matched an environment, check succeeded')
            return 1
        elif self.excs:  # Are there exceptions?
            logger.debug('>> Environments and exceptions both don\'t match, check failed')
            return 0
        else:
            logger.debug('>> Environment doesn\'t match, check "else" rule')
        if self.otherwise is not None:  # Try checking otherwise
            check = self.otherwise.checkMatch(match, word)
            return check + (1 if check else 0)
        else:
            logger.debug('>> No "else" rule, check failed')
        return 0

@dataclass
class RuleBlock(list):
    '''Groups a block of sound changes together.

    Instance variables:
        flags -- flags for altering execution (Flags)
    '''
    ruleset: InitVar[list]
    flags: Flags = Flags()

    def __post_init__(self, ruleset):
        list.__init__(self, ruleset)

    def apply(self, word):
        from random import randint
        applied = False
        rules = []  # We use a list to store rules, since they may be applied multiple times
        values = []  # We have a parallel list for storing the values of the 'for' flag per rule
        for _rule in self:
            # We want _rule to run before the stored rules, but to be placed at the end instead
            rules.append(_rule)
            values.append(_rule.flags.persist)
            for rule in [_rule]+rules[:-1]:
                flags = rule.flags
                if not flags.ditto or (flags.ditto != 1) ^ applied:
                    for j in range(flags.repeat):
                        if randint(1, 100) <= flags.chance:
                            applied = True
                            wordin = word
                            try:
                                word = rule.apply(word)
                            except RuleFailed:
                                applied = False
                                logger.info(f'`{rule}` does not apply to `{word}`')
                                break
                            except RuleError as e:
                                logger.warning(f'`{rule}` execution suffered an error: {e}')
                                break
                            if wordin == word:
                                logger.info(f'`{rule}` does not change `{word}`')
                                break
                            else:
                                logger.info(f'`{wordin}` -> `{rule}` -> `{word}`')
                        else:
                            applied = False
                            logger.info(f'`{rule}` was randomly not run on `{word}`')
                    if flags.stop and (flags.stop != 1) ^ applied:
                        return word
            for i in reversed(range(len(rules))):
                values[i] -= 1
                if values[i] == 0:  # If the rule has 'expired', discard it
                    del rules[i]
                    del values[i]
        return word

@dataclass
class Line:
    word: Word = None
    comment: str = None

    def __str__(self):
        components = []
        if self.word is not None:
            components.append(str(self.word))
        if self.comment is not None:
            components.append(f'//{self.comment}')
        return ' '.join(components)

# == Functions == #
def parseWordset(wordset, graphs=(), separator='', syllabifier=None):
    '''Parses a wordlist.

    Arguments:
        wordset -- the words to be parsed (str)
        graphs  -- list of graphemes used to parse the words (list)

    Returns a list.
    '''
    if isinstance(wordset, str):
        wordset = wordset.splitlines()
    _wordset = []
    for word in wordset:
        if isinstance(word, Word):
            line = Line(word=word)
        elif isinstance(word, Line):
            line = word
        elif word.startswith('//'):  # Is a comment
            line = Line(comment=word[2:])
        elif '//' in word:  # Contains a comment
            word, comment = word.split('//', 1)
            line = Line(word=Word(word, graphs, separator, syllabifier), comment=comment)
        elif word:
            line = Line(word=Word(word, graphs, separator, syllabifier))
        else:
            line = Line()
        _wordset.append(line)
    return _wordset

def tokeniseCategory(line, linenum=0):
    for match in CATEGORY_REGEX.finditer(line):
        type = match.lastgroup
        value = match.group()
        column = match.start()
        if type == 'OP':
            value = value.strip()
        elif type == 'UNKNOWN':
            raise CompilerError(f'unexpected character', value, linenum, column)
        yield Token(type, value, linenum, column)

def compileCategory(line, linenum=0, cats=None):
    tokens = list(tokeniseCategory(line, linenum))
    if [token.type for token in tokens] != ['CATEGORY', 'OP', 'VALUES']:
        raise FormatError(f'{line!r} is not a category definition')
    name, op, values = [token.value for token in tokens]
    if ',' not in values:
        values += ','
    cat = Cat.make(f'[{values}]', cats, name)
    if op == '=':
        return {name: cat}
    else:
        if cats is None or name not in cats:
            raise TokenError(f'category {name!r} is not defined', tokens[1])
        if op == '+=':
            return {name: cats[name]+cat}
        elif op == '-=':
            return {name: cats[name]-cat}
        else:
            raise TokenError('invalid category operation', tokens[1])

def tokeniseFlags(line, linenum=0, colstart=None):
    for match in FLAG_REGEX.finditer(line, colstart):
        type = match.lastgroup
        value = match.group()
        column = match.start()
        if type == 'UNKNOWN':
            raise CompilerError(f'unexpected character', value, linenum, column)
        yield Token(type, value, linenum, column)

def compileFlags(tokens):
    tokens = list(tokens)
    binaryflags = ('ignore', 'rtl')
    ternaryflags = ('ditto', 'stop')
    numericflags = {'repeat': MAX_RUNS, 'persist': MAX_RUNS, 'chance': 100}  # Maximum values
    flags = {}
    for flag, token in partitionTokens(tokens, 'SEPARATOR'):
        if not flag:
            raise TokenError('expected flag', token)
        elif flag[0].type == 'NEGATION':
            name = flag[-1].value
            if len(flag) == 1:
                raise TokenError('expected flag name', token)
            elif flag[1].type != 'FLAG':
                raise TokenError('expected flag name', flag[1])
            elif name not in ternaryflags:
                raise TokenError('invalid ternary flag name', flag[1])
            elif len(flag) == 2:
                flags[name] = -1
            else:
                raise TokenError('expected semicolon', flag[2])
        elif flag[0].type == 'FLAG':
            name = flag[0].value
            arg = flag[-1].value
            if name not in FLAGS:
                raise TokenError('invalid flag name', flag[0])
            elif len(flag) == 1:
                if name in numericflags:
                    flags[name] = numericflags[name]  # Set to maximum value
                else:
                    flags[name] = 1
            elif flag[1].type != 'COLON':
                raise TokenError('expected colon or semicolon', flag[1])
            elif name not in numericflags:
                raise TokenError('invalid numeric flag name', flag[1])
            elif len(flag) == 2:
                raise TokenError('expected integer argument', token)
            elif flag[2].type != 'ARGUMENT':
                raise TokenError('expected integer argument', flag[2])
            elif not (1 <= int(arg) <= numericflags[name]):
                raise TokenError('argument out of range', flag[2])
            elif len(flag) == 3:
                flags[name] = int(arg)
            else:
                raise TokenError('expected semicolon', flag[3])
        else:
            raise TokenError('invalid flag', flag[0])
    return Flags(**flags)

def tokeniseMetarule(line, linenum=0):
    for match in METARULE_REGEX.finditer(line):
        type = match.lastgroup
        value = match.group()
        column = match.start()
        if type == 'METARULE':
            value = value[1:]
        elif type == 'SPACE':
            yield Token(type, value, linenum, column)
            yield from tokeniseFlags(line, linenum, match.end())
            break
        elif type == 'UNKNOWN':
            raise CompilerError('unexpected character', value, linenum, column)
        yield Token(type, value, linenum, column)

def compileMetarule(line, linenum=0):
    tokens = list(tokeniseMetarule(line, linenum))
    if not tokens:
        raise ValueError('tokens cannot be empty')
    name = tokens[0].value
    for ix, token in enumerate(tokens):
        if token.type == 'SPACE':  # Found flags
            flags = compileFlags(tokens[ix+1:])
            break
    else:
        ix = len(tokens)
        if name == 'block':
            flags = Flags()
        else:
            flags = None
    arg = tokens[ix-1].value
    if tokens[0].type != 'METARULE':
        raise TokenError('expected metarule name', tokens[0])
    elif name not in METARULES:
        raise TokenError('invalid metarule name', tokens[0])
    elif name in ('def', 'rule') and flags:
        raise TokenError(f'metarule !{name} cannot take flags', tokens[ix])
    elif ix == 1:
        if name == 'block':
            arg = None
        else:
            if ix < len(tokens):
                token = tokens[ix]
            else:
                token = Token('', '', linenum, tokens[-1].column+len(tokens[-1].value))
            raise TokenError(f'metarule !{name} requires an argument', token)
    elif tokens[1].type != 'COLON':
        raise TokenError('expected colon', tokens[1])
    elif ix == 2:
        raise TokenError('colon must be followed by an argument', tokens[1])
    elif tokens[2].type != 'NUMBER' and name == 'block':
        raise TokenError('metarule !block requires an integer argument', tokens[2])
    elif tokens[2].type != 'IDENTIFIER' and name in ('def', 'rule'):
        raise TokenError(f'metarule !{name} requires an alphabetic argument', tokens[2])
    elif ix == 3:
        if name == 'block':
            arg = int(arg)
    else:
        raise TokenError('expected space or newline', tokens[3])
    return name, arg, flags

def tokeniseRule(line, linenum=0):
    colstart = 0
    while colstart < len(line):
        match = RULE_REGEX.match(line, colstart)
        type = match.lastgroup
        value = match.group()
        column = match.start()
        colstart = match.end()
        if type == 'INDICES':
            yield Token(type, value[1:], linenum, column)
            continue
        elif type == 'SPACE':
            yield Token(type, value, linenum, column)
            yield from tokeniseFlags(line, linenum, colstart)
            break
        elif type == 'UNKNOWN':
            if column == 0:
                type = 'TARGET'
                value = ''
                colstart = 0
            else:
                raise CompilerError(f'unexpected character', value, linenum, column)
        yield Token(type, value, linenum, column)
        colstart = yield from tokenisePattern(line, colstart, linenum)
    else:
        yield Token('END', '', linenum, colstart)

def compileIndexedPattern(pattern, cats=None, reduceindices=True):
    if pattern[-1].type == 'INDICES':
        indices = [int(index) for index in pattern[-1].value.split('|')]
        if reduceindices:
            indices = [index-(1 if index>0 else 0) for index in indices]
        pattern = pattern[:-1]
    else:
        indices = None
    return compilePattern(pattern, cats), indices

def compileTarget(pattern, cats=None):
    return Target(*compileIndexedPattern(pattern, cats))

def compileEpenthesis(pattern, cats=None):
    pattern, indices = compileIndexedPattern(pattern, cats, False)
    return Target([], indices), Replacement(pattern)

def compileReplacement(pattern, cats=None):
    pattern, indices = compileIndexedPattern(pattern, cats, False)
    if indices is not None:
        raise FormatError('replacement field cannot contain indices')
    else:
        return Replacement(pattern)

def compileEnvironment(pattern, cats=None, reduceindices=True):
    patterns = []
    for pattern, sep in partitionTokens(pattern, 'PLACEHOLDER'):
        if sep is not None and patterns:  # Only one placeholder is allowed, which then follows the first pattern
            raise TokenError('invalid placeholder', sep)
        patterns.append(pattern)
    if len(patterns) == 2:
        left, right = patterns
        env = LocalEnvironment(compilePattern(left, cats), compilePattern(right, cats))
    elif len(patterns) == 1:
        pattern = patterns[0]
        env = GlobalEnvironment(*compileIndexedPattern(pattern, cats, reduceindices))
    return env or None

COMPILERS = {
    'EPENTHESIS': compileEpenthesis,
    'DELETION': compileTarget,
    'TARGET': compileTarget,
    'MOVE': lambda pattern, cats: compileField(pattern, cats, 'AND', False),
    'COPY': lambda pattern, cats: compileField(pattern, cats, 'AND', False),
    'REPLACEMENT': compileReplacement,
    'ENVIRONMENT': lambda pattern, cats: compileField(pattern, cats, 'AND'),
    'EXCEPTION': lambda pattern, cats: compileField(pattern, cats, 'AND'),
}

def compileField(tokens, cats=None, delimiter='OR', reduceindices=True):
    if not tokens:
        return []
    if tokens[-1].type == delimiter:
        raise TokenError('invalid delimiter', tokens[-1])
    fieldmarker = tokens[0].type
    _compile = COMPILERS.get(fieldmarker, lambda pattern, cats: compileEnvironment(pattern, cats, reduceindices))
    if fieldmarker in COMPILERS:
        tokens = tokens[1:]
    field = []
    for pattern, sep in partitionTokens(tokens, delimiter):
        if not pattern:
            raise TokenError('unexpected delimiter', sep)
        field.append(_compile(pattern, cats))
    # Final replacements field handling
    if fieldmarker in ('MOVE', 'COPY'):
        return fieldmarker.lower(), field
    elif fieldmarker == 'EPENTHESIS':
        return map(list, zip(*field))
    return field

FIELD_MARKERS = {
    'EPENTHESIS': 'reps',
    'DELETION': 'tars',
    'TARGET': 'tars',
    'MOVE': 'reps',
    'COPY': 'reps',
    'REPLACEMENT': 'reps',
    'ENVIRONMENT': 'envs',
    'EXCEPTION': 'excs',
}

def compileRule(line, linenum=0, cats=None):
    from math import ceil
    if isinstance(line, str):
        tokens = list(tokeniseRule(line, linenum))
    else:
        tokens = line
        line = ''
    if tokens[0].type == 'END':
        tokens = []
    elif tokens[0].type not in FIELD_MARKERS:
        raise TokenError('unexpected token', tokens[0])
    fields = {
        'otherwise': None,
        'flags': Flags(),
        'rule': line
    }
    # Extract flags
    for ix, token in enumerate(tokens):
        if token.type == 'SPACE':
            fields['flags'] = compileFlags(tokens[ix+1:])
            tokens[ix].type = 'END'
            break
    # Extract remainder of fields
    i = None
    for j, token in enumerate(tokens):
        type, value = token
        if type in FIELD_MARKERS or type == 'END':
            if i is not None:
                field = FIELD_MARKERS[tokens[i].type]
                if field in fields:
                    raise TokenError('unexpected field marker', tokens[i])
                fields[field] = tokens[i:j]
            i = j
            if type in ('MOVE', 'COPY', 'REPLACEMENT'):
                if 'reps' in fields:  # Detected an otherwise
                    fields['otherwise'] = compileRule(fields.get('tars', []) + tokens[j:ix+1], cats=cats)
                    break
            elif type == 'END':
                break
    # Check for restricted field combinations
    if 'tars' in fields and 'reps' in fields:
        if fields['tars'][0].type == 'DELETION':
            raise TokenError('replacement field not allowed with deletion', fields['reps'][0])
        if fields['reps'][0].type == 'EPENTHESIS':
            raise TokenError('target field not allowed with epenthesis', fields['tars'][0])
    # Compile fields
    fields['tars'] = compileField(fields.get('tars', []), cats) or [[]]
    fields['reps'] = compileField(fields.get('reps', []), cats) or [[]]
    fields['envs'] = compileField(fields.get('envs', []), cats) or [[]]
    fields['excs'] = compileField(fields.get('excs', []), cats)
    # Handle indexed epenthesis
    if isinstance(fields['reps'], map):  # Epenthesis
        fields['tars'], fields['reps'] = fields['reps']
    return Rule(**fields)

def compileLine(line, linenum=0, cats=None):
    if not line:
        return None
    # Attempt to tokenise as category
    with suppress(CompilerError, FormatError):
        return compileCategory(line, linenum, cats)
    # Attempt to tokenise as metarule
    with suppress(CompilerError, FormatError):
        return compileMetarule(line, linenum)
    # Attempt to tokenise as rule
    return compileRule(line, linenum, cats)

def makeBlock(ruleset, start=None, num=None, defs=None):
    if defs is None:
        defs = {}
    else:
        defs = defs.copy()
    cats = []
    block = []
    if start is None:
        i = 0
    else:
        i = start
    while len(block) != num and i < len(ruleset):
        rule = ruleset[i]
        i += 1
        if isinstance(rule, Rule):  # Rule
            block.append(rule)
        elif isinstance(rule, dict):  # Category
            cats += rule.items()
        elif isinstance(rule, tuple):  # Metarule
            name, arg, flags = rule
            if name == 'block':
                if arg is not None:
                    _block, _cats, i, defs = makeBlock(ruleset, i, arg, defs)
                else:
                    _block, _cats = makeBlock(ruleset, i, arg, defs)
                    i = len(ruleset)
                block.append(RuleBlock(_block, flags))
                cats += _cats
            elif name == 'def':
                _block, _cats, i, defs = makeBlock(ruleset, i, 1, defs)
                defs[arg] = _block
                cats += _cats
            elif name == 'rule':
                block.extend(defs[arg])
    if start is None:
        return block, cats
    else:
        return block, cats, i, defs

def compileRuleset(ruleset, cats=None):
    if isinstance(ruleset, str):
        ruleset = ruleset.splitlines()
    if cats is None:
        cats = {}
    else:
        cats = cats.copy()
    _ruleset = []
    for linenum, line in enumerate(ruleset):
        # Remove comments
        line = line.split('//')[0].strip()
        # Compile
        try:
            rule = compileLine(line, linenum, cats)
        except CompilerError as e:
            logger.warning(f'{line!r} failed to compile due to bad formatting: {e}')
        except Exception as e:
            logger.warning(f'{line!r} failed to compile due to an unexpected error: {e}')
        else:
            if isinstance(rule, dict):  # Category
                cats.update(rule)
            _ruleset.append(rule)
    # Evaluate meta-rules
    ruleset, _cats = makeBlock(_ruleset)
    return RuleBlock(ruleset), _cats

def setupLogging(filename=__location__, loggername='sce'):
    global logger
    if filename is not None:
        logging.config.fileConfig(filename)
    logger = logging.getLogger(loggername)

def run(wordset, ruleset, cats=None, syllabifier=None, output='list'):
    '''Applies a set of sound change rules to a set of words.

    Arguments:
        wordset     -- the words to which the rules are to be applied (list)
        ruleset     -- the rules which are to be applied to the words (RuleBlock)
        cats        -- the initial categories to be used in ruleset compiling (dict)
        syllabifier -- the syllabifier function to use for syllabifying words (RulesSyllabifier)
        output      -- what form to provide the output in - one of 'list', 'as-is', 'str' (str)

    Returns a str or list.
    '''
    if not ruleset or not wordset:  # One of these is blank so do nothing
        return wordset
    cats = parseCats(cats or {})
    ruleset, _cats = compileRuleset(ruleset, cats)  # Compile ruleset first so we can use the graphs it contains
    # Try to get graphs and separator from the initial categories
    graphs = cats.get('graphs', ())
    separator = cats.get('separator', [''])[0]
    # Ruleset overrides externally-supplied categories
    for name, cat in _cats:
        if name == 'graphs':
            graphs = cat
        elif name == 'separator':
            separator = cat[0]
        else:
            break
    wordset = parseWordset(wordset, graphs, separator, syllabifier)
    for line in wordset:
        if line.word is not None:  # There's a word
            logger.info(f'This word: {line.word}')
            logger.debug(f'Segments: {line.word.phones}')
            line.word = ruleset.apply(line.word)
    if output != 'as-is':
        wordset = [str(line) for line in wordset]
    if output == 'str':
        wordset = '\n'.join(wordset)
    return wordset

apply_ruleset = run

# Setup logging
setupLogging()
