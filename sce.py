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
Maybe change >^ and >^? to >> and >>?
Reserve category `separator`, remove separator from `graphs`

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
                    match, rpos, catixes = word.match_pattern(pattern, pos)
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
            check = self.check_match(matches[i], word)
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
        phones = tuple(word)
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
            logger.debug('>> Environments and exceptions both don\'t match, check failed')
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
            raise CompilerError(f'unexpected character', value, linenum, column)
        yield Token(type, value, linenum, column)
        colstart = yield from tokenisePattern(line, colstart, linenum)
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
            indices = [int(index)-(1 if int(index)>0 else 0) for index in pattern[-1].value.split('|')]
            pattern = pattern[:-1]
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
        replacements = compileEnvironments(tokens, cats, reduceindices=False)
        # Space for sanity-checking the environments - in particular, global envs must have no pattern
        return (type, replacements)
    else:
        replacements = compileTargets(tokens, cats)  # Necessary because of indexed epenthesis
        if type == 'epenthesis':
            targets = []
            for i, replacement in enumerate(replacements):
                pattern, indices = replacement
                if indices is not None:
                    indices = [int(index)+(1 if int(index)>=0 else 0) for index in indices]
                targets.append(Target([], indices))
                replacements[i] = Replacement(pattern)
            return targets, replacements
        else:
            return [Replacement(r.pattern) for r in replacements]

def compileEnvironments(tokens, cats=None, reduceindices=True):
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
            patterns = list(partition(pattern, sep_func=(lambda t: t.type == 'PLACEHOLDER')))
            if len(patterns) == 2:
                left, right = patterns
                env = LocalEnvironment(compilePattern(left, cats), compilePattern(right, cats))
                if not env.left and not env.right:
                    env = None
            elif len(patterns) == 1:
                if pattern[-1].type == 'INDICES':
                    if reduceindices:
                        indices = [int(index)-(1 if int(index)>0 else 0) for index in pattern[-1].value.split('|')]
                    else:
                        indices = [int(index) for index in pattern[-1].value.split('|')]
                    pattern = pattern[:-1]
                else:
                    indices = None
                env = GlobalEnvironment(compilePattern(pattern, cats), indices)
                if not env.pattern and env.indices is None:
                    env = None
            _environment.append(env)
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
    from math import ceil
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
        if type in FIELD_MARKERS or type == 'END':
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
    if isinstance(fields['reps'], tuple) and not isinstance(fields['reps'][0], str):  # Indexed epenthesis
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
        except Exception as e:
            logger.warning(f'{line!r} failed to compile due to an unexpected error: {e}')
        else:
            if isinstance(rule, dict):  # Category
                cats.update(rule)
            else:
                _ruleset.append(rule)
    # Evaluate meta-rules
    ruleset = makeBlock(_ruleset)
    return RuleBlock(ruleset)

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
    ruleset = compileRuleset(ruleset, cats)
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
