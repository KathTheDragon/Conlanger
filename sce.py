'''Apply sound changes to a lexicon

Exceptions:
    RuleFailed    -- exception to mark that a rule failed
    WordUnchanged -- exception to break out of repeated rule application

Classes:
    Rule -- represents a sound change rule

Functions:
    compile_ruleset -- compiles a sound change ruleset
    parse_field     -- parse the fields of a rule
    parse_flags     -- parse the flags of a rule
    apply_ruleset   -- applies a set of sound change rules to a set of words
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Check that tar still matches immediately before replacement (difficult)
Move compiling code to own functions
Is there a better name for Rule.else_?

=== Features ===
Implement $ and syllables
Implement " for copying previous segment
Implement additional logic options for environments
Is it possible to implement a>b>c as notation for a chain shift?

=== Style ===
Consider where to raise/handle exceptions
'''

from math import ceil
from random import randint
from core import LangException, Cat, Word, parse_syms, split

# == Constants == #
MAX_RUNS = 10**3  # Maximum number of times a rule may be repeated

# == Exceptions == #
class RuleFailed(LangException):
    '''Used to indicate that the rule failed to be applied.'''

class WordUnchanged(LangException):
    '''Used to indicate that the word was not changed by the rule.'''

# == Classes == #
class Rule():
    '''Class for representing a sound change rule.
    
    Instance variables:
        rule  -- the rule as a string (str)
        tars  -- target segments (list)
        reps  -- replacement segments (list)
        envs  -- application environments (list)
        excs  -- exception environments (list)
        else_ -- the rule to apply if an exception is satisfied (Rule)
        flags -- flags for altering execution (dict)
    
    Methods:
        apply       -- apply the rule to a word
        apply_match -- apply a single match to a word
    '''
        
    __slots__ = ['rule', 'tars', 'reps', 'envs', 'excs', 'else_', 'flags']
    
    def __init__(self, rule='', cats=None):  # Format is tars>reps/envs!excs flag; envs, excs, and flag are all optional
        '''Constructor for Rule
        
        Arguments:
            rule -- the rule as a string (str)
            cats -- dictionary of categories used to interpret the rule (dict)
        '''
        self.rule = rule
        if ' ' in rule:
            rule, flags = rule.split()
        else:
            flags = ''
        if rule[0] == '+':
            rule = '>'+rule[1:]
        elif rule[0] == '-':
            rule = rule[1:]
        rule = rule.replace('>', ' >').replace('{ >', '{>').replace('/', ' /').replace('!', ' !').split(' ')
        tars = rule.pop(0)
        # We want to extract just the first iteration of (reps, envs, excs) and store everything else in else_
        # To do this, we observe that if we fill in missing fields once we reach a later field, then if we hit
        # a repeat (by seeing that the field variable is not None) we are in the second iteration. If there is
        # no second iteration, else_ will be None.
        reps = envs = excs = else_ = None
        for i in range(len(rule)):
            if rule[i][0] == '>' and reps is None:
                reps = rule[i][1:]
                continue
            if rule[i][0] == '/' and envs is None:
                envs = rule[i][1:]
                if reps is None:
                    reps = ''
                continue
            if rule[i][0] == '!' and excs is None:
                excs = rule[i][1:]
                if envs is None:
                    envs = '_'
                if reps is None:
                    reps = ''
                continue
            else_ = rule[i:]
            else_.insert(0, tars)
        if reps is None:
            reps = ''
        if envs is None:
            envs = '_'
        if excs is None:
            excs = ''
        if cats is None:
            cats = {}
        self.tars = parse_field(tars, 'tars', cats)
        self.reps = parse_field(reps, 'reps', cats)
        self.envs = parse_field(envs, 'envs', cats)
        self.excs = parse_field(excs, 'envs', cats)
        self.flags = parse_flags(flags)
        if not self.reps:
            self.reps = [[]]
        if len(self.reps) < len(self.tars):
            self.reps *= ceil(len(self.tars)/len(self.reps))
        if else_ is not None:
            self.else_ = Rule(''.join(else_), cats)
        else:
            self.else_ = None
        if self.flags['rtl']:
            self.reverse()
        return
    
    def __repr__(self):
        return f"Rule('{self!s}')"
    
    def __str__(self):
        return self.rule
    
    def reverse(self):
        for tar in self.tars:
            tar.reverse()
        for rep in self.reps:
            rep.reverse()
        for env in self.envs:
            env.reverse()
            env[0].reverse()
            if len(env) == 2:
                env[1].reverse()
        for exc in self.excs:
            exc.reverse()
            exc[0].reverse()
            if len(exc) == 2:
                exc[1].reverse()
        if self.else_ is not None:
            self.else_.reverse()
    
    def apply(self, word):
        '''Apply the sound change rule to a single word.
        
        Arguments:
            word -- the word to which the rule is to be applied (Word)
        
        Raises RuleFailed if the rule did not apply to the word.
        Raises WordUnchanged if the word was not changed by the rule.
        '''
        phones = tuple(word)
        if self.flags['rtl']:
            word = word[::-1]
        matches = []
        tars = self.tars
        if not tars:
            tars = [([], ())]
        for i in range(len(tars)):
            if tars[i]:
                tar, indices = tars[i]
            else:
                tar, indices = [], ()
            _matches = []
            for pos in range(len(word)):  # Find all matches
                match, length, catixes = word.match_pattern(tar, pos, return_cats=True)
                if match:  # tar matches at pos
                    _matches.append((pos, length, catixes, i))
            # Filter only those matches selected by the given indices
            if not indices:
                matches += _matches
            else:
                matches += [_matches[ix] for ix in indices]
        # Filter only those matches that fit the environment - also record the corresponding replacement
        reps = []
        i = 0
        for i in reversed(range(len(matches))):
            if self.check_match(matches[i], word):
                # Save the appropriate rep
                reps.append(self.reps[matches[i][3]])
            else:
                del matches[i]
        reps.reverse()
        # Filter overlaps
        i = 1  # This marks the match we're testing for overlapping the previous match
        while i < len(matches):
            if matches[i][0] < matches[i-1][0] + matches[i-1][1]:  # Overlap
                del matches[i]
                del reps[i]
            else:
                i += 1
        for match, rep in sorted(zip(matches, reps), reverse=True):
            word = self.apply_match(match, rep, word)
        if self.flags['rtl']:
            word = word[::-1]
        if not reps:
            raise RuleFailed
        if phones == tuple(word):
            raise WordUnchanged
        return word
    
    def check_match(self, match, word):
        pos, length = match[:2]
        tar = word[pos:pos+length]
        if self.excs and any(word.match_env(exc, pos, tar) for exc in self.excs):
            if self.else_ is not None:  # Try checking else_
                return self.else_.check_match(match, word)
        elif any(word.match_env(env, pos, tar) for env in self.envs):
            return True
        elif not self.excs:
            if self.else_ is not None:  # Try checking else_
                return self.else_.check_match(match, word)
        return False
    
    def apply_match(self, match, rep, word):
        '''Apply a replacement to a word
        
        Arguments:
            match -- the match to be checked
            rep   -- the replacement to be used
            word  -- the word to check against
        
        Returns a bool.
        '''
        pos, length, catixes = match[:3]
        tar = word[pos:pos+length]
        if isinstance(rep, list):  # Replacement
            rep = rep.copy()
            # Deal with categories
            ix = 0
            for i in range(len(rep)):
                if isinstance(rep[i], Cat):
                    rep[i] = rep[i][catixes[ix]]
            # Deal with target references
            for i in reversed(range(len(rep))):
                if rep[i] == '%':  # Target copying
                    rep[i:i+1] = tar
                elif rep[i] == '<':  # Target reversal/metathesis
                    rep[i:i+1] = reversed(tar)
            word = word[:pos] + Word(rep) + word[pos+length:]
        else:  # Movement
            if isinstance(rep[1], list):  # Environment
                mode, envs = rep
                matches = []
                for wpos in range(len(word)):  # Find all matches
                    if any(word.match_env(env, wpos) for env in envs):
                        if mode == 'move' and wpos >= pos + length:  # We'll need to adjust the matches down
                            wpos -= length
                        matches.append(wpos)
            else:  # Indices
                mode, matches = rep[0], list(rep[1])
            if mode == 'move':  # Move - delete original tar
                word = word[:pos] + word[pos+length:]
            for match in sorted(matches, reverse=True):
                word = word[:match] + tar + word[match:]
        return word

# == Functions == #
def parse_wordset(wordset, graphs=None):
    '''Parses a wordlist.
    
    Arguments:
        wordset -- the words to be parsed (str)
        graphs  -- list of graphemes used to parse the words (list)
    
    Returns a list.
    '''
    if isinstance(wordset, str):
        wordset = wordset.splitlines()
    else:
        wordset = wordset.copy()
    for i in reversed(range(len(wordset))):
        if wordset[i] == '':
            del wordset[i]
        elif isinstance(wordset[i], Word):
            continue
        else:
            wordset[i] = Word(wordset[i], graphs)
    return wordset

def compile_ruleset(ruleset, cats=None):
    '''Compile a sound change ruleset.
    
    Arguments:
        ruleset -- the set of rules to be compiled (str)
        cats    -- the initial categories to be used to compile the rules (dict)
    
    Returns a list.
    '''
    if isinstance(ruleset, str):
        ruleset = ruleset.splitlines()
    else:
        ruleset = ruleset.copy()
    if cats is None:
        cats = {}
    for i in range(len(ruleset)):
        rule = ruleset[i].strip()
        if rule == '':
            ruleset[i] = None
        elif isinstance(rule, Rule):
            continue
        elif '>' in rule or rule[0] in '+-':  # Rule is a sound change
            ruleset[i] = Rule(rule, cats)
        else:  # Rule is a cat definition
            cop = rule.index('=')
            op = (rule[cop-1] if rule[cop-1] in '+-' else '') + '='
            name, vals = rule.split(op)
            exec(f'cats[name] {op} Cat(vals)')
            for cat in list(cats):  # Discard blank categories
                if not cats[cat]:
                    del cats[cat]
            ruleset[i] = None
    for i in reversed(range(len(ruleset))):
        if ruleset[i] is None or ruleset[i].flags['ignore']:
            del ruleset[i]
    return ruleset
    
def parse_field(field, mode, cats=None):
    '''Parse a field of a sound change rule.
    
    Arguments:
        field -- the field to be parsed (str)
        mode  -- which kind of field it is (str)
        cats  -- dictionary of categories (dict)
    
    Returns a list
    '''
    if cats is None:
        cats = {}
    _field = []
    if mode == 'envs':
        for env in split(field, '|', minimal=True):
            if env.startswith('~'):  # ~X is equivalent to X_,_X
                _field += parse_field('{0}_|_{0}'.format(env.strip('~')), 'envs', cats)
            elif '_' in env:
                env = env.split('_')
                env = [parse_syms(env[0], cats)[::-1], parse_syms(env[1], cats)]
            else:
                env = [parse_syms(env, cats)]
            _field.append(env)
    elif mode == 'tars':
        for tar in split(field, ',', nesting=(0, '([{', '}])'), minimal=True):
            if '@' in tar:
                tar, indices = tar.split('@')
                indices = tuple(int(index) for index in split(indices, '|', minimal=True))
            else:
                indices = ()
            tar = parse_syms(tar, cats)
            tar = (tar, indices)
            _field.append(tar)
    elif mode == 'reps':
        for rep in split(field, ',', nesting=(0, '([{', '}])'), minimal=True):
            if rep.startswith('^'):  # Movement rule
                if rep.startswith('^?'):
                    mode = 'move'
                else:
                    mode = 'copy'
                rep = rep.strip('^?')
                if rep.startswith('@'):  # Indices
                    rep = (mode, tuple(int(index) for index in split(rep.strip('@'), '|', minimal=True)))
                else:  # Environment
                    rep = (mode, parse_field(rep, 'envs', cats))
            else:  # Replace rule
                rep = parse_syms(rep, cats)
            _field.append(rep)
    return _field

def parse_flags(flags):
    '''Parse the flags of a sound change rule.
    
    Arguments:
        flags -- the flags to be parsed (str)
        
    Returns a dictionary.
    '''
    _flags = dict(ignore=0, ditto=0, stop=0, rtl=0, repeat=1, age=1, chance=100)  # Default values
    for flag in split(flags, ';', minimal=True):
        if ':' in flag:
            flag, arg = flag.split(':')
            _flags[flag] = int(arg)
        else:
            _flags[flag] = 1-_flags[flag]
    if not 0 < _flags['repeat'] <= MAX_RUNS:
        _flags['repeat'] = MAX_RUNS
    if not 0 < _flags['age'] <= MAX_RUNS:
        _flags['age'] = MAX_RUNS
    return _flags

def apply_ruleset(wordset, ruleset, graphs=None, cats=None, debug=False):
    '''Applies a set of sound change rules to a set of words.
    
    Arguments:
        wordset -- the words to which the rules are to be applied (list)
        ruleset -- the rules which are to be applied to the words (list)
        cats    -- the initial categories to be used in ruleset compiling (dict)
    
    Returns a list.
    '''
    wordset = parse_wordset(wordset, graphs)
    ruleset = compile_ruleset(ruleset, cats)
    rules = []  # We use a list to store rules, since they may be applied multiple times
    applied = [False]*len(wordset)  # For each word, we store a boolean noting whether a rule got applied or not
    for rule in ruleset:
        rules.append(rule)
        if debug:
            print('Words =', [str(word) for word in wordset])  # For debugging
        for i in range(len(wordset)):
            if applied[i] is not None:  # We stopped execution for this word
                for rule in reversed(rules):
                    # Either the rule isn't marked 'ditto', or it is and the last rule ran
                    if not rule.flags['ditto'] or applied[i]:
                        if debug:
                            print('rule =', rule)  # For debugging
                        applied[i] = None if rule.flags['stop'] else True
                        for j in range(rule.flags['repeat']):
                            try:
                                if randint(1, 100) <= rule.flags['chance']:
                                    wordset[i] = rule.apply(wordset[i])
                                else:
                                    applied[i] = False
                            except RuleFailed:  # The rule didn't apply, make note of this
                                applied[i] = False
                                break
                            except WordUnchanged:  # If the word didn't change, stop applying
                                break
                        if applied[i] is None:
                            break
        for i in reversed(range(len(rules))):
            rules[i].flags['age'] -= 1
            if rules[i].flags['age'] == 0:  # If the rule has 'expired', discard it
                del rules[i]
    return wordset

