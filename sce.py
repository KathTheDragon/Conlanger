'''Apply sound changes to a lexicon

Exceptions:
    RuleFailed    -- exception to mark that a rule failed
    WordUnchanged -- exception to break out of repeated rule application

Classes:
    Rule -- represents a sound change rule

Functions:
    compile_ruleset -- compiles a sound change ruleset
    compile_rule    -- compiles a sound change rule
    parse_tars      -- parse the targets of a rule
    parse_reps      -- parse the replacements of a rule
    parse_envs      -- parse the environments of a rule
    parse_flags     -- parse the flags of a rule
    run             -- applies a set of sound change rules to a set of words
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===

=== Features ===
Is it possible to implement a>b>c as notation for a chain shift?
Think about expanding the options for grapheme handling
- diacritics
Allow ~ in tar and rep
Look into *not* removing category definitions, to aid extracting graph definitions
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
from dataclasses import dataclass, InitVar
from math import ceil
from random import randint
from .core import LangException, FormatError, RuleError, CompilerError, TokenError, Token, Cat, Word, resolveTargetRef, parse_cats, split, partition
from ._pattern import parse_pattern, escape, tokenise as tokenisePattern, compile_tokens as compilePattern

# == Constants == #
MAX_RUNS = 10**3  # Maximum number of times a rule may be repeated
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), 'logging.conf'))
RULE_TOKENS = {
    'EPENTHESIS': r'^\+ ?',
    'DELETION': r'^\- ?',
    'TARGET': r'^',
    'MOVE': r'>\^\?| >\^\? ',
    'COPY': r'>\^| >\^ ',
    'REPLACEMENT': r'>| > ',
    'ENVIRONMENT': r'/| / ',
    'EXCEPTION': r'!| ! ',
    'OR': r', ?',
    'AND': r'&| & ',
    'PLACEHOLDER': r'_',
    # 'ADJACENCY': r'~',
    'INDICES': r'@\-?\d+(?:\|\-?\d+)*',
    'SPACE': r' ',
    'UNKNOWN': r'.',
    'END': ''
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
    'OP': r'(?:\+|\-)?=| (?:\+|\-)?= ',
    'VALUES': r'.+$',  # Might make this part more precise
    'UNKNOWN': r'.',
}
CATEGORY_REGEX = re.compile('|'.join(f'(?P<{type}>{regex})' for type, regex in CATEGORY_TOKENS.items()))

# == Globals == #
logger = None

# == Exceptions == #
class RuleFailed(LangException):
    '''Used to indicate that the rule failed to be applied.'''

class WordUnchanged(LangException):
    '''Used to indicate that the word was not changed by the rule.'''

# == Classes == #
@dataclass
class Target:
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
        return Target(self.pattern.copy(), self.indices.copy())

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

    def __iter__(self):
        yield self.left
        yield self.right

    def copy(self):
        return LocalEnvironment(self.left.copy(), self.right.copy())

    def resolveTargetRef(self, target):
        return LocalEnvironment(resolveTargetRef(self.left, target), resolveTargetRef(self.right, target))

@dataclass
class GlobalEnvironment:
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
        if self.indices is not None:
            return GlobalEnvironment(self.pattern.copy(), self.indices.copy())
        else:
            return GlobalEnvironment(self.pattern.copy())

    def resolveTargetRef(self, target):
        if self.indices is not None:
            return GlobalEnvironment(resolveTargetRef(self.pattern, target), self.indices.copy())
        else:
            return GlobalEnvironment(resolveTargetRef(self.pattern, target))

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
        check_match -- check if the match is valid
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
        Raises WordUnchanged if the word was not changed by the rule.
        '''
        logger.debug(f'This rule: `{self}`')
        phones = tuple(word)
        # Get all target matches, filtered by given indices
        logger.debug('Begin matching targets')
        matches = []
        for i, tar in enumerate(self.tars):
            if isinstance(tar, tuple):
                tar, indices = tar
            elif isinstance(tar, list):
                tar, indices = tar, ()
            else:
                tar, indices = [], ()
            logger.debug(f'> Matching `{tar}@{indices}`')
            if not tar:  # All pos's
                logger.debug(f'>> Null target matched all positions in range 1..{len(word)}')
                _matches = [(pos, pos, [], i) for pos in range(1, len(word))]
            else:
                _matches = []
                for pos in range(1, len(word)):  # Find all matches
                    match, rpos, catixes = word.match_pattern(tar, pos)
                    if match:  # tar matches at pos
                        logger.debug(f'>> Target matched `{word[pos:rpos]}` at {pos}')
                        _matches.append((pos, rpos, catixes, i))
                if not _matches:
                    logger.debug('>> No matches for this target')
            # Filter only those matches selected by the given indices
            if not indices:
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
            check = self.check_match(matches[i], word)
            if not check:
                del matches[i]
            else:
                # Find the correct replacement
                logger.debug('>> Get replacement for this match')
                if check == 1:
                    reps.append(self.reps[matches[i][3]])
                    logger.debug(f'>>> Found {self.reps[matches[i][3]]}')
                else:
                    otherwise = self.otherwise
                    for j in range(check-2):
                        otherwise = otherwise.otherwise
                    reps.append(otherwise.reps[matches[i][3]])
                    logger.debug(f'>>> Found {otherwise.reps[matches[i][3]]}')
        reps.reverse()
        if not reps:
            logger.debug('No matches matched environment')
            raise RuleFailed
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
            word = word.apply_match(match, rep)
        if phones == tuple(word):
            raise WordUnchanged
        return word

    def check_match(self, match, word):
        pos, rpos = match[:2]
        if any(word.match_env(exc, pos, rpos) for exc in self.excs):  # If there are exceptions, does any match?
            logger.debug('>> Matched an exception, check the "else" rule')
        elif any(word.match_env(env, pos, rpos) for env in self.envs):  # Does any environment match?
            logger.debug('>> Matched an environment, check succeeded')
            return 1
        elif self.excs:  # Are there exceptions?
            logger.debug('>> Environments and exceptions don\'t match, check failed')
            return 0
        else:
            logger.debug('>> Environment doesn\'t match, check "else" rule')
        if self.otherwise is not None:  # Try checking otherwise
            check = self.otherwise.check_match(match, word)
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
                            try:
                                wordin = word
                                word = rule.apply(word)
                                logger.info(f'`{wordin}` -> `{rule}` -> `{word}`')
                                continue
                            except RuleFailed:  # The rule didn't apply, make note of this
                                applied = False
                                logger.info(f'`{rule}` does not apply to `{word}`')
                            except WordUnchanged:  # If the word didn't change, stop applying
                                logger.info(f'`{rule}` does not change `{word}`')
                            except RuleError as e:  # Some other problem occurred
                                logger.warning(f'{rule} execution suffered an error: {e}')
                            break
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

# == Functions == #
def parse_wordset(wordset, cats=None, syllabifier=None):
    '''Parses a wordlist.

    Arguments:
        wordset -- the words to be parsed (str)
        graphs  -- list of graphemes used to parse the words (list)

    Returns a list.
    '''
    if isinstance(wordset, str):
        wordset = wordset.splitlines()
    if cats is not None and 'graphs' in cats:
        graphs = cats['graphs']
    else:
        graphs = Cat(["'"])
    _wordset = []
    for word in wordset:
        if isinstance(word, Word):
            line = [word]
        elif isinstance(word, list):
            line = word
        elif word.startswith('//'):  # Is a comment
            line = [word[2:]]
        elif '//' in word:  # Contains a comment
            word, comment = word.split('//', 1)
            line = [Word(word, graphs, syllabifier), comment]
        elif word:
            line = [Word(word, graphs, syllabifier)]
        else:
            line = []
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
    for flag, token in partition(tokens, sep_func=(lambda token: token.type == 'SEPARATOR'), yield_sep=True):
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
            raise CompilerError(f'unexpected character', value, linenum, column)
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
            raise CompilerError(f'unexpected character', value, linenum, column)
        yield Token(type, value, linenum, column)
        for token in tokenisePattern(line, colstart, linenum):
            if token.type != 'END':
                yield token
            else:
                colstart = token.column
                break
    else:
        yield Token('END', '', linenum, colstart)

def compileTargets(tokens, cats=None):
    targets = []
    if not tokens:
        return []
    if tokens[-1].type == 'OR':
        raise TokenError('invalid comma', tokens[-1])
    type = tokens[0].type
    for pattern, sep in partition(tokens[1:], sep_func=(lambda t: t.type == 'OR'), yield_sep=True):
        if not pattern:
            raise TokenError('unexpected comma', sep)
        if pattern[-1].type == 'INDICES':
            if type == 'REPLACEMENT':
                raise TokenError('indices not allowed in replacement field', pattern[-1])
            pattern, indices = pattern[:-1], [int(index) for index in pattern[-1].value.split('|')]
        else:
            indices = None
        targets.append(Target(compilePattern(pattern, cats), indices))
    return targets

def compileReplacements(tokens, cats=None):
    replacements = []
    if not tokens:
        return []
    if tokens[-1].type == 'OR':
        raise TokenError('invalid comma', tokens[-1])
    type = tokens[0].type.lower()
    if type in ('move', 'copy'):
        replacements = compileEnvironments(tokens, cats)
        # Space for sanity-checking the environments - in particular, global envs must have no pattern
        return (type, replacements)
    else:
        replacements = compileTargets(tokens, cats)  # Necessary because of indexed epenthesis
        if type == 'epenthesis':
            targets = []
            for i, replacement in enumerate(replacements):
                pattern, indices = replacement
                targets.append(Target([], indices))
                replacements[i] = Replacement(pattern)
            return (indices, replacements)
        else:
            return [Replacement(r.pattern) for r in replacements]

def compileEnvironments(tokens, cats=None):
    environments = []
    if not tokens:
        return []
    if tokens[-1].type == 'OR':
        raise TokenError('invalid comma', tokens[-1])
    for environment, sep in partition(tokens[1:], sep_func=(lambda t: t.type == 'OR'), yield_sep=True):
        if not environment:
            raise TokenError('unexpected comma', sep)
        _environment = []
        if tokens[-1].type == 'AND':
            raise TokenError('invalid and', tokens[-1])
        for pattern, sep in partition(environment, sep_func=(lambda t: t.type == 'AND'), yield_sep=True):
            if not pattern:
                raise TokenError('unexpected comma', sep)
            _env = list(partition(pattern, sep_func=(lambda t: t.type == 'PLACEHOLDER')))
            if len(_env) == 2:
                left, right = _env
                _environment.append(LocalEnvironment(compilePattern(left, cats), compilePattern(right, cats)))
            else:
                if pattern[-1].type == 'INDICES':
                    pattern, indices = pattern[:-1], [int(index) for index in pattern[-1].value.split('|')]
                else:
                    indices = None
                _environment.append(GlobalEnvironment(compilePattern(pattern, cats), indices))
        environments.append(_environment)
    return environments

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
    if isinstance(line, str):
        tokens = list(tokeniseRule(line, linenum))
    else:
        tokens = line
        line = ''
    if not tokens:
        return None
    elif tokens[0].type not in ('EPENTHESIS', 'DELETION', 'TARGET'):
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
        if type in RULE_TOKENS and type not in ('OR', 'AND', 'PLACEHOLDER', 'INDICES'):
            if i is not None:
                field = FIELD_MARKERS[tokens[i].type]
                if field in fields:
                    raise TokenError('unexpected field marker', tokens[i])
                fields[field] = tokens[i:j]
            i = j
            if type in ('MOVE', 'COPY', 'REPLACEMENT'):
                if 'reps' in fields:  # Detected an otherwise
                    fields['otherwise'] = compileRule(fields.get('tars', []) + tokens[j:ix+1], cats)
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
    fields['tars'] = compileTargets(fields.get('tars', []), cats) or [[]]
    fields['reps'] = compileReplacements(fields.get('reps', []), cats) or [[]]
    fields['envs'] = compileEnvironments(fields.get('envs', []), cats) or [[]]
    fields['excs'] = compileEnvironments(fields.get('excs', []), cats)
    # Handle indexed epenthesis
    if isinstance(fields['reps'], tuple) and isinstance(fields['reps'][0], list):  # Indexed epenthesis
        fields['tars'], fields['reps'] = fields['reps']
    return Rule(**fields)

def compileLine(line, linenum=0, cats=None):
    if not line:
        return None
    # Attempt to tokenise as category
    try:
        return compileCategory(line, linenum, cats)
    except TokenError:
        raise
    except:
        pass
    # Attempt to tokenise as metarule
    try:
        return compileMetarule(line, linenum)
    except TokenError:
        raise
    except:
        pass
    # Attempt to tokenise as rule
    return compileRule(line, linenum, cats)

def makeBlock(ruleset, start=None, num=None, defs=None):
    if defs is None:
        defs = {}
    else:
        defs = defs.copy()
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
        elif isinstance(rule, tuple):  # Metarule
            name, arg, flags = rule
            if name == 'block':
                _block, i, defs = makeBlock(ruleset, i, arg, defs)
                block.append(RuleBlock(_block, flags))
            elif name == 'def':
                _block, i, defs = makeBlock(ruleset, i, 1, defs)
                defs[arg] = _block
            elif name == 'rule':
                block.extend(defs[arg])
    if start is None:
        return block
    else:
        return block, i, defs

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
        else:
            if isinstance(rule, dict):  # Category
                cats.update(rule)
            else:
                _ruleset.append(rule)
    # Evaluate meta-rules
    ruleset = makeBlock(_ruleset)
    return RuleBlock(ruleset)

def compile_ruleset(ruleset, cats=None):
    '''Compile a sound change ruleset.

    Arguments:
        ruleset -- the set of rules to be compiled (str)
        cats    -- the initial categories to be used to compile the rules (dict)

    Returns a list.
    '''
    if isinstance(ruleset, str):
        ruleset = ruleset.splitlines()
    if cats is None:
        cats = {}
    _ruleset = []
    for rule in ruleset:
        # Escape characters
        rule = escape(rule)
        # Remove comments
        if isinstance(rule, str):
            rule = rule.split('//')[0].strip()
        # Compile
        if rule == '':
            continue
        elif isinstance(rule, Rule):
            _ruleset.append(rule)
        elif '>' in rule or rule[0] in '+-':  # Rule is a sound change
            try:
                _ruleset.append(compile_rule(rule, cats))
            except FormatError as e:
                logger.warning(f'Rule `{rule}` failed to compile due to bad formatting: {e}')
        elif '=' in rule:  # Rule is a cat definition
            if rule.count('=') > 1:
                logger.warning(f'Category `{rule}` failed to compile due to bad formatting: category definitions can only have one "="')
                continue
            cats = parse_cats(rule, cats)
        elif rule.startswith('!'):  # Meta-rule
            _ruleset.append(rule.strip('!'))
    # Evaluate meta-rules
    ruleset = make_block(_ruleset)[0]
    return RuleBlock(ruleset)

def make_block(ruleset, n=None, defs=None):
    if defs is None:
        defs = {}
    else:
        defs = defs.copy()
    block = []
    while len(block) != n and ruleset:
        rule, ruleset = ruleset[0], ruleset[1:]
        if isinstance(rule, Rule):
            block.append(rule)
        else:
            _rule = rule  # Save for error messages
            try:
                if ' ' in rule:
                    rule, flags = rule.split()
                else:
                    flags = ''
                flags = parse_flags(flags)
                rule, arg = validate_arg(rule)
                if rule == 'block':
                    if arg is not None:
                        try:
                            arg = int(arg)
                        except ValueError:
                            raise FormatError('`block` must have an integer argument')
                    _block, ruleset, defs = make_block(ruleset, arg, defs)
                    block.append(RuleBlock(_block, flags))
                elif rule == 'def':
                    if arg is None:
                        raise FormatError('`def` must have an argument')
                    if not ruleset:
                        raise FormatError('`def` requires a following rule')
                    [rule], ruleset, defs = make_block(ruleset, 1, defs)
                    defs[arg] = rule
                elif rule == 'rule':
                    if arg is None:
                        raise FormatError('`rule` must have an argument')
                    if arg not in defs:
                        raise FormatError(f'rule `{arg}` has not been defined')
                    block.append(defs[arg])
                else:
                    raise FormatError('unknown meta-rule')
            except FormatError as e:
                logger.warning(f'Meta-rule `!{_rule}` failed to compile due to bad formatting: {e}')
    return block, ruleset, defs

def validate_arg(rule):
    if ':' not in rule:
        return rule, None
    rule, arg = rule.split(':', 1)
    if arg == '':
        raise FormatError(f'arguments must not be blank')
    if ':' in arg:
        raise FormatError(f'more than one argument may not be given')
    return rule, arg

regexCat = re.compile(r'\[(?!\])')

regexes = (
    re.compile(r' (> [/!] )'),  # Used to delete whitespace before > / or > !
    re.compile(r'(?:^| )(>\^\??|[>/!|&@])(?: |$)'),  # Used to delete whitespace around >^ , >^? , or any of >/!|&@
    re.compile(r'^([+-]) '),  # Used to delete whitespace after either of initial +-
    re.compile(r'([:;,^]) '),  # Used to delete whitespace after any of :;,^
    re.compile(r'(?<!{)([>/!])')  # Used to insert whitespace before field markers
)

def compile_rule(rule, cats=None):
    '''Factory function for Rule objects

    Arguments:
        rule -- the rule as a string (str)
        cats -- dictionary of categories used to interpret the rule (dict)
    '''
    _rule = rule
    for regex in regexes[:-1]:  # Various whitespace manipulations
        rule = regex.sub(r'\1', rule)
    if ' ' in rule:  # Flags are separated by whitespace from the rest of the rule
        rule, flags = rule.rsplit(maxsplit=1)
    else:
        flags = ''
    if '>' not in rule:
        standardise = True
    elif '/' in rule and rule.find('/') < rule.find('>'):
        standardise = True
    elif '!' in rule and rule.find('!') < rule.find('>'):
        standardise = True
    else:
        standardise = False
    if standardise:
        if rule.startswith('+'):  # Put epenthesis/deletion operators into standard form
            rule = '>' + rule[1:]
        elif rule.startswith('-'):
            rule = rule[1:]
    # Identify the field operators and place a space before them - if there are any, tars comes before
    # the first operator, else tars is the whole rule
    if '>' in rule or '/' in rule or '!' in rule:
        tars, rule = re.sub(regexes[-1], r' \1', rule).split(' ', maxsplit=1)
    else:
        tars, rule = rule, ''
    # If there is a > field, it will begin the rule, and there must always be a field before otherwise,
    # so otherwise begins at the first non-initial >. otherwise is None if not present
    pos = rule.find(' >', 1)
    if pos != -1:
        otherwise = rule[pos:].replace(' ', '')
        rule = rule[:pos].split()
    else:
        otherwise = None
        rule = rule.split()
    for i in range(3):  # Fill in missing fields
        if len(rule) == i or rule[i][0] != ['>', '/', '!'][i]:
            rule.insert(i, ['>', '/', '!'][i])
    if not tars.strip(',') and '@' in rule[0]:  # Indexed epenthesis
        tars = ''
        _reps = split(rule[0], ',', nesting=(0, '([{', '}])'), minimal=True)
        rule[0] = ''
        for _rep in _reps:  # Maybe optimise this so I can use rule[0] = ','.join(reps) and tars = ','.join(indiceses)
            if '@' in _rep:  # Index
                if _rep.count('@') > 1:
                    raise FormatError(f'indexed epenthesis must have one `@` per replacement: {_rep}')
                _rep, indices = _rep.split('@')
                tars += '@'+indices+','
            else:
                tars += '@,'
            rule[0] += _rep+','
    if otherwise is not None:  # We need to add the tars to otherwise to make a valid rule, then compile
        otherwise = tars.strip(',') + otherwise
        otherwise = compile_rule(otherwise, cats)
    # Check for invalid formatting where possible here
    if regexCat.match(rule[0]) is not None and regexCat.match(tars) is None:
        raise FormatError(f'a replacement contains a category while the targets do not')
    # Parse the fields
    tars = parse_tars(tars, cats) or [[]]
    reps = parse_reps(rule[0].strip('>'), cats) or [[]]
    envs = parse_envs(rule[1].strip('/'), cats) or [[]]
    excs = parse_envs(rule[2].strip('!'), cats)
    flags = parse_flags(flags)
    if len(reps) < len(tars):  # If reps is shorter than tars, repeat reps until it isn't
        reps *= ceil(len(tars)/len(reps))
    return Rule(tars, reps, envs, excs, otherwise, flags, _rule)

def parse_tars(tars, cats=None):
    '''Parse the targets of a sound change rule.

    Arguments:
        tars -- the targets to be parsed (str)
        cats -- dictionary of categories (dict)

    Returns a list
    '''
    _tars = []
    for tar in split(tars, ',', nesting=(0, '([{', '}])'), minimal=True):
        _tar = tar  # Record the original for error messages
        tar = tar.rstrip('@|')
        if '@' in tar:
            if tar.count('@') > 1:
                raise FormatError(f'indexed targets must have exactly one `@`: {_tar}')
            tar, indices = tar.split('@')
            try:
                indices = tuple(int(index)-(1 if int(index) > 0 else 0) for index in split(indices, '|', minimal=True))
            except ValueError:
                raise FormatError(f'indices must be a pipe-separated (`|`) list of numbers: {_tar}')
            tar = (parse_pattern(tar, cats), indices)
        else:
            tar = parse_pattern(tar, cats)
        _tars.append(tar)
    return _tars

def parse_reps(reps, cats=None):
    '''Parse the replacements of a sound change rule.

    Arguments:
        reps -- the replacements to be parsed (str)
        cats  -- dictionary of categories (dict)

    Returns a list
    '''
    _reps = []
    for rep in split(reps, ',', nesting=(0, '([{', '}])'), minimal=True):
        _rep = rep  # Record the original for error messages
        if rep.startswith('^'):  # Movement rule
            if rep.count('^') > 1:
                raise FormatError(f'replacement fields must have exactly one `^` in movement mode: {_rep}')
            if rep.startswith('^?'):
                mode = 'move'
            else:
                mode = 'copy'
            rep = rep.strip('^?')
            if rep.startswith('@'):  # Indices
                try:
                    rep = (mode, tuple(int(index) for index in split(rep.strip('@'), '|', minimal=True)))
                except ValueError:
                    raise FormatError(f'indices must be a pipe-separated (`|`) list of numbers: {_rep}')
            else:  # Environment
                rep = (mode, parse_envs(rep, cats))
        else:  # Replace rule
            rep = parse_pattern(rep, cats)
        _reps.append(rep)
    return _reps

def parse_envs(envs, cats=None):
    '''Parse the environments of a sound change rule.

    Arguments:
        envs -- the environments to be parsed (str)
        cats  -- dictionary of categories (dict)

    Returns a list
    '''
    _envs = []
    for env in split(envs, ',', minimal=True):
        _env = env  # Record the original for error messages
        env = env.strip('@|')
        if '&' in env:
            env = tuple(parse_envs(env.replace('&',','), cats))
        elif '~' in env:  # A~B is equivalent to AB,BA - ~B is equivalent to _~B
            if env == '~':
                env = []
            else:
                env = env.split('~')
                env[0] = env[0] or '_'
                env[-1] = env[-1] or '_'
                _envs.extend(parse_envs('{},{}'.format(''.join(env), ''.join(reversed(env))), cats))
                continue
        elif '_' in env:  # Local environment
            if env.count('_') > 1:
                raise FormatError(f'local environments must have exactly one `_`: {_env}')
            env = env.split('_')
            env = [parse_pattern(env[0], cats), parse_pattern(env[1], cats)]
        elif '@' in env:  # Indexed global environment
            if env.count('@') > 1:
                raise FormatError(f'indexed global environments must have exactly one `@`: {_env}')
            env, indices = env.split('@')
            try:
                indices = tuple(int(index)-(1 if int(index) < 0 else 0) for index in split(indices, '|', minimal=True))
            except ValueError:
                raise FormatError(f'indices must be a pipe-separated list of numbers: {_env}')
            env = [(parse_pattern(env, cats), indices)]
        else:  # Global environment
            env = [parse_pattern(env, cats)]
        if env in ([[]], [[],[]]):
            env = []
        _envs.append(env)
    return _envs

def parse_flags(flags):
    '''Parse the flags of a sound change rule.

    Arguments:
        flags -- the flags to be parsed (str)

    Returns a namedtuple.
    '''
    binaryflags = ('ignore', 'rtl')
    ternaryflags = ('ditto', 'stop')
    numericflags = {'repeat': MAX_RUNS, 'persist': MAX_RUNS, 'chance': 100}  # Max values
    _flags = {'ignore': 0, 'ditto': 0, 'stop': 0, 'rtl': 0, 'repeat': 1, 'persist': 1, 'chance': 100}  # Default values
    for flag in split(flags, ';', minimal=True):
        _flag = flag  # Record the original for error messages
        if ':' in flag:
            flag, arg = flag.split(':', 1)
            if ':' in arg:
                raise FormatError(f'flags must have at most one argument: {_flag}')
            if flag not in numericflags:
                raise FormatError(f'invalid numeric flag: {_flag}')
            try:
                arg = int(arg)
            except ValueError:
                raise FormatError(f'flags must have integer arguments: {_flag}')
            if 1 <= arg <= numericflags[flag]:
                _flags[flag] = arg
            else:
                raise FormatError(f'flag argument out of range: {_flag}')
        elif flag.startswith('!'):
            flag = flag.strip('!')
            if flag in ternaryflags:
                _flags[flag] = -1
            else:
                raise FormatError(f'invalid ternary flag: {_flag}')
        else:
            if flag in numericflags:
                _flags[flag] = numericflags[flag]  # Set to maximum value
            elif flag in _flags:
                _flags[flag] = 1  # All other flags, set to 1
            else:
                raise FormatError(f'invalid flag: {_flag}')
    return Flags(**_flags)

def setup_logging(filename=__location__, logger_name='sce'):
    global logger
    if filename is not None:
        logging.config.fileConfig(filename)
    logger = logging.getLogger(logger_name)

def run(wordset, ruleset, cats='', syllabifier=None, output='list'):
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
    cats = parse_cats(cats)
    # If we didn't get passed a graphs category, check if we can get it from the ruleset
    if 'graphs' not in cats:
        if isinstance(ruleset, str):
            rule = ruleset.splitlines()[0]
        else:
            rule = ruleset[0]
        if isinstance(rule, str) and '>' not in rule and '=' in rule and rule.startswith('graphs'):
            cats['graphs'] = Cat.make(rule.split('=')[1].strip(), cats)
    wordset = parse_wordset(wordset, cats, syllabifier)
    ruleset = compile_ruleset(ruleset, cats)
    for line in wordset:
        if len(line) == 2 or len(line) == 1 and isinstance(line[0], Word):  # There's a word
            logger.info(f'This word: {line[0]}')
            line[0] = ruleset.apply(line[0])
    if output != 'as-is':
        for i, line in enumerate(wordset):
            if len(line) == 2 or len(line) == 1 and isinstance(line[0], str):
                line[-1] = '//'+line[-1]
            if len(line) == 2 or len(line) == 1 and isinstance(line[0], Word):
                line[0] = str(line[0])
            wordset[i] = ' '.join(line)
    if output == 'str':
        wordset = '\n'.join(wordset)
    return wordset

apply_ruleset = run

# Setup logging
setup_logging()
