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
Write rule validator

=== Features ===
Is it possible to implement a>b>c as notation for a chain shift?
Think about expanding the options for grapheme handling
- diacritics?

=== Style ===
Consider where to raise/handle exceptions
Go over docstrings
'''

import logging
import logging.config
import os.path
import re
from collections import namedtuple
from math import ceil
from random import randint
from .core import LangException, Cat, Word, parse_syms, parse_cats, split
from .phomo import translate

# == Constants == #
MAX_RUNS = 10**3  # Maximum number of times a rule may be repeated
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__), 'logging.conf'))

# == Globals == #
logger = None

# == Exceptions == #
class RuleFailed(LangException):
    '''Used to indicate that the rule failed to be applied.'''

class WordUnchanged(LangException):
    '''Used to indicate that the word was not changed by the rule.'''

# == Classes == #
class Rule(namedtuple('Rule', 'rule tars reps envs excs otherwise flags')):
    '''Class for representing a sound change rule.
    
    Instance variables:
        rule      -- the rule as a string (str)
        tars      -- target segments (list)
        reps      -- replacement segments (list)
        envs      -- application environments (list)
        excs      -- exception environments (list)
        otherwise -- the rule to apply if an exception is satisfied (Rule)
        flags     -- flags for altering execution (dict)
    
    Methods:
        apply       -- apply the rule to a word
        check_match -- check if the match is valid
    '''
        
    __slots__ = ()
    
    def __repr__(self):
        return f"Rule('{self!s}')"
    
    def __str__(self):
        return self.rule
    
    def __eq__(self, other):
        return self[1:] == other[1:]
        
    def apply(self, word, debug=False):
        '''Apply the sound change rule to a single word.
        
        Arguments:
            word -- the word to which the rule is to be applied (Word)
        
        Raises RuleFailed if the rule did not apply to the word.
        Raises WordUnchanged if the word was not changed by the rule.
        '''
        phones = tuple(word)
        matches = []
        for i in range(len(self.tars)):
            if isinstance(self.tars[i], tuple):
                tar, indices = self.tars[i]
            elif isinstance(self.tars[i], list):
                tar, indices = self.tars[i], ()
            else:
                tar, indices = [], ()
            _matches = []
            for pos in range(1, len(word)):  # Find all matches
                match, length, catixes = word.match_pattern(tar, pos)
                if match:  # tar matches at pos
                    _matches.append((pos, length, catixes, i))
            # Filter only those matches selected by the given indices
            if not indices:
                matches += _matches
            else:
                matches += [_matches[ix] for ix in indices if ix < len(_matches)]
        # Filter only those matches that fit the environment - also record the corresponding replacement
        reps = []
        for i in reversed(range(len(matches))):
            check = self.check_match(matches[i], word)
            if not check:
                del matches[i]
            else:
                # Find the correct match
                if check == 1:
                    reps.append(self.reps[matches[i][3]])
                else:
                    otherwise = self.otherwise
                    for j in range(check-2):
                        otherwise = otherwise.otherwise
                    reps.append(otherwise.reps[matches[i][3]])
        reps.reverse()
        matches = sorted(zip(matches, reps), reverse=True)
        # Filter overlaps
        if self.flags.rtl:
            i = 1
            while i < len(matches):
                if matches[i][0][0] + matches[i][0][1] > matches[i-1][0][0]:  # Overlap
                    del matches[i]
                else:
                    i += 1
        else:
            for i in reversed(range(len(matches)-1)):
                if matches[i][0][0] < matches[i+1][0][0] + matches[i+1][0][1]:  # Overlap
                    del matches[i]
        for match, rep in matches:
            word = word.apply_match(match, rep)
        if not reps:
            raise RuleFailed
        if phones == tuple(word):
            raise WordUnchanged
        return word
    
    def check_match(self, match, word):
        pos, length = match[:2]
        if any(word.match_env(exc, pos, length) for exc in self.excs):  # If there are exceptions, does any match?
            pass
        elif any(word.match_env(env, pos, length) for env in self.envs):  # Does any environment match?
            return 1
        elif self.excs:  # Are there exceptions?
            return 0
        if self.otherwise is not None:  # Try checking otherwise
            check = self.otherwise.check_match(match, word)
            return check + (1 if check else 0)
        return 0

class RuleBlock(list):
    '''Groups a block of sound changes together.
    
    Instance variables:
        flags -- flags for altering execution (namedtuple)
    '''
    
    __slots__ = ('flags',)
    
    def __init__(self, ruleset, flags=None):
        self.flags = flags
        list.__init__(self, ruleset)
    
    def apply(self, word):
        applied = False
        rules = []  # We use a list to store rules, since they may be applied multiple times
        values = []  # We have a parallel list for storing the values of the 'for' flag per rule
        for _rule in self:
            # We want _rule to run before the stored rules, but to be placed at the end instead
            rules.append(_rule)
            values.append(_rule.flags.for_)
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
                            except RuleFailed:  # The rule didn't apply, make note of this
                                applied = False
                                logger.info(f'`{rule}` did not apply to `{word}`')
                                break
                            except WordUnchanged:  # If the word didn't change, stop applying
                                logger.info(f'`{word}` is not changed by `{rule}`')
                                break
                        else:
                            applied = False
                    if flags.stop and (flags.stop != 1) ^ applied:
                        return word
            for i in reversed(range(len(rules))):
                values[i] -= 1
                if values[i] == 0:  # If the rule has 'expired', discard it
                    del rules[i]
                    del values[i]
        return word

Flags = namedtuple('Flags', 'ignore, ditto, stop, rtl, repeat, for_, chance')

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
        graphs = Cat("'")
    _wordset = []
    for word in wordset:
        # Remove comments
        if isinstance(word, str):
            word = word.split('//')[0].strip()
        # Parse
        if word == '':
            continue
        elif isinstance(word, Word):
            _wordset.append(word)
        else:
            _wordset.append(Word(word, graphs, syllabifier))
    return _wordset

regexes = re.compile(r'\s+(>\s+[/!]\s+)'), re.compile(r'\s+([>/!|&@])\s+'), re.compile(r'^([+-])\s+'), re.compile(r'([:;,])\s+')

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
    phomo = False  # Tells us if we need to translate from PhoMo to SCE syntax first
    for rule in ruleset:
        # Check PhoMo
        if phomo:
            rule = translate(rule)
            if isinstance(phomo, int):
                phomo -= 1
                if phomo == 0:
                    phomo = False
        # Remove comments
        if isinstance(rule, str):
            rule = rule.split('//')[0].strip()
        # Compile
        if rule == '':
            continue
        elif isinstance(rule, Rule):
            _ruleset.append(rule)
        elif '>' in rule or rule[0] in '+-':  # Rule is a sound change
            _ruleset.append(compile_rule(rule, cats))
        elif '=' in rule:  # Rule is a cat definition
            cop = rule.index('=')
            op = (rule[cop-1] if rule[cop-1] in '+-' else '') + '='
            name, vals = rule.split(op)
            name, vals = name.strip(), vals.strip()
            if name != '':
                exec(f'cats[name] {op} Cat(vals, cats)')
                if not cats[name]:
                    del cats[name]
        elif rule.startswith('!'):  # Meta-rule
            if rule.startswith('!phomo'):  # Enable PhoMo rules
                if ':' in rule:
                    phomo = int(rule.split(':')[1])
                else:
                    phomo = True
                continue
            _ruleset.append(rule.strip('!'))
    # Second pass to create blocks
    for i in reversed(range(len(_ruleset))):
        if isinstance(_ruleset[i], str):
            rule = _ruleset[i]
            rule = regexes[-1].sub(r'\1', rule)  # Clear extra whitespace
            if ' ' in rule:
                rule, flags = rule.split()
            else:
                flags = ''
            flags = parse_flags(flags)
            if ':' in rule:
                rule, arg = rule.split(':')
                arg = int(arg)
            else:
                arg = 0
            if rule == 'block':
                if arg:
                    _ruleset[i:i+arg+1] = [RuleBlock(_ruleset[i+1:i+arg+1], flags)]
                else:
                    _ruleset[i:] = RuleBlock(_ruleset[i+1:], flags)
    return RuleBlock(_ruleset)

def compile_rule(rule, cats=None):
    '''Factory function for Rule objects
    
    Arguments:
        rule -- the rule as a string (str)
        cats -- dictionary of categories used to interpret the rule (dict)
    '''
    _rule = rule
    for regex in regexes:  # Various whitespace manipulations
        rule = regex.sub(r'\1', rule)
    if ' ' in rule:  # Flags are separated by whitespace from the rest of the rule
        rule, flags = rule.rsplit(maxsplit=1)
    else:
        flags = ''
    if rule.startswith('+'):  # Put epenthesis/deletion operators into standard form
        rule = '>' + rule.strip('+')
    elif rule.startswith('-'):
        rule = rule.strip('-')
    # Identify the field operators and place a space before them - if there are any, tars comes before
    # the first operator, else tars is the whole rule
    if '>' in rule or '/' in rule or '!' in rule:
        tars, rule = re.sub(r'(?<!{)([>/!])', r' \1', rule).split(' ', maxsplit=1)
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
        for _rep in _reps:
            if '@' in _rep:  # Index
                _rep, indices = _rep.split('@')
                tars += '@'+indices+','
            else:
                tars += '@,'
            rule[0] += _rep+','
    if otherwise is not None:  # We need to add the tars to otherwise to make a valid rule, then compile
        otherwise = tars.strip(',') + otherwise
        otherwise = compile_rule(otherwise, cats)
    # Parse the fields
    tars = parse_tars(tars, cats) or [[]]
    reps = parse_reps(rule[0].strip('>'), cats) or [[]]
    envs = parse_envs(rule[1].strip('/'), cats) or [[]]
    excs = parse_envs(rule[2].strip('!'), cats)
    flags = parse_flags(flags)
    if len(reps) < len(tars):  # If reps is shorter than tars, repeat reps until it isn't
        reps *= ceil(len(tars)/len(reps))
    return Rule(_rule, tars, reps, envs, excs, otherwise, flags)

def parse_tars(tars, cats=None):
    '''Parse the targets of a sound change rule.
    
    Arguments:
        tars -- the targets to be parsed (str)
        cats -- dictionary of categories (dict)
    
    Returns a list
    '''
    _tars = []
    for tar in split(tars, ',', nesting=(0, '([{', '}])'), minimal=True):
        tar = tar.rstrip('@|')
        if '@' in tar:
            tar, indices = tar.split('@')
            indices = tuple(int(index)-(1 if int(index) > 0 else 0) for index in split(indices, '|', minimal=True))
            tar = (parse_syms(tar, cats), indices)
        else:
            tar = parse_syms(tar, cats)
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
        if rep.startswith('^'):  # Movement rule
            if rep.startswith('^?'):
                mode = 'move'
            else:
                mode = 'copy'
            rep = rep.strip('^?')
            if rep.startswith('@'):  # Indices
                rep = (mode, tuple(int(index) for index in split(rep.strip('@'), '|', minimal=True)))
            else:  # Environment
                rep = (mode, parse_envs(rep, cats))
        else:  # Replace rule
            rep = parse_syms(rep, cats)
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
    for env in split(envs, '|', minimal=True):
        env = env.strip('@,')
        if '&' in env:
            env = tuple(parse_envs(env.replace('&','|'), cats))
        elif env.startswith('~'):  # ~X is equivalent to X_|_X
            _envs.extend(parse_envs('{0}_|_{0}'.format(env.strip('~')), cats))
            continue
        elif '_' in env:  # Local environment
            env = env.split('_')
            env = [parse_syms(env[0], cats), parse_syms(env[1], cats)]
        elif '@' in env:  # Indexed global environment
            env, indices = env.split('@')
            indices = tuple(int(index)-(1 if int(index) > 0 else 0) for index in split(indices, ',', minimal=True))
            env = [(parse_syms(env, cats), indices)]
        else:  # Global environment
            env = [parse_syms(env, cats)]
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
    _flags = {'ignore': 0, 'ditto': 0, 'stop': 0, 'rtl': 0, 'repeat': 1, 'for': 1, 'chance': 100}  # Default values
    for flag in split(flags, ';', minimal=True):
        if ':' in flag:
            flag, arg = flag.split(':')
            if flag in _flags:
                _flags[flag] = int(arg)
        elif flag.startswith('!'):
            flag = flag.strip('!')
            if flag in _flags:
                _flags[flag] = _flags[flag]-1
        else:
            if flag in _flags:
                _flags[flag] = 1-_flags[flag]
    _flags['for_'] = _flags['for']
    del _flags['for']
    # Validate values
    # Binary flags
    for flag in ('ignore', 'rtl'):
        if not 0 <= _flags[flag] <= 1:
            _flags[flag] = 0
    # Ternary flags
    for flag in ('ditto', 'stop'):
        if not -1 <= _flags[flag] <= 1:
            _flags[flag] = 0
    # Unbounded flags
    for flag in ('repeat', 'for_'):
        if not 1 <= _flags[flag] <= MAX_RUNS:
            _flags[flag] = MAX_RUNS
    # Value flags
    if not 0 <= _flags['chance'] <= 100:
        _flags['chance'] = 100
    return Flags(**_flags)

def validate_rule(rule):
    '''Determine if a rule is valid or not.
    
    Arguments:
        rule -- the rule being tested (Rule)
    '''
    pass

def setup_logging(filename=__location__, logger_name='sce'):
    global logger
    if filename is not None:
        logging.config.fileConfig(filename)
    logger = logging.getLogger(logger_name)

def run(wordset, ruleset, cats='', syllabifier=None, to_string=False):
    '''Applies a set of sound change rules to a set of words.
    
    Arguments:
        wordset     -- the words to which the rules are to be applied (list)
        ruleset     -- the rules which are to be applied to the words (RuleBlock)
        cats        -- the initial categories to be used in ruleset compiling (dict)
        syllabifier -- the syllabifier function to use for syllabifying words (Syllabifier)
        to_string   -- whether to give a string or list output
    
    Returns a str or list.
    '''
    cats = parse_cats(cats)
    # If we didn't get passed a graphs category, get it from the ruleset
    if 'graphs' not in cats:
        if isinstance(ruleset, str):
            rule = ruleset.splitlines()[0]
        else:
            rule = ruleset[0]
        if isinstance(rule, str) and '>' not in rule and '=' in rule and rule.startswith('graphs'):
            cats['graphs'] = Cat(rule.split('=')[1].strip(), cats)
    wordset = parse_wordset(wordset, cats, syllabifier)
    ruleset = compile_ruleset(ruleset, cats)
    wordset = [str(ruleset.apply(word)) for word in wordset]
    if to_string:
        wordset = '\n'.join(wordset)
    return wordset

apply_ruleset = run

# Setup logging
setup_logging()

