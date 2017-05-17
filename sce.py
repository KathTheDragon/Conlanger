'''Apply sound changes to a lexicon

Exceptions:
    WordUnchanged -- exception to break out of repeated rule application

Classes:
    Rule -- represents a sound change rule

Functions:
    parse_ruleset -- parses a sound change ruleset
    parse_field   -- parse the fields of a rule
    parse_flags   -- parse the flags of a rule
    apply_ruleset -- applies a set of sound change rules to a set of words
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Check that tar still matches immediately before replacement (difficult)
Check if a rule is able to run infinitely and raise an exception if it can
- (tar in rep and rule['repeat'] < 1)
Move compiling code to own functions
Is there a better name for Rule.else_?

=== Features ===
Implement $ and syllables
Implement " for copying previous segment
Implement * in the target, with variants **, *?, **?
Implement ^ for range indices
Implement flag 'chance' for non-deterministic rule application
Implement flag 'stop' to terminate execution if the rule succeeds
Implement flag 'ditto' to only run a rule if the previous rule ran
Implement extended category substitution
Implement additional logic options for environments
Implement repetition shorthand

=== Style ===
Write docstrings
Consider where to raise/handle exceptions
'''

from math import ceil
from core import LangException, Cat, parse_syms, split

#== Constants ==#
MAX_RUNS = 10**3 #maximum number of times a rule may be repeated

#== Exceptions ==#
class WordUnchanged(LangException):
    '''Used to indicate that the word was not changed by the rule.'''

#== Classes ==#
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
    '''
    
    rule = ''
    
    def __init__(self, rule='', cats=None): #format is tars>reps/envs!excs flag; envs, excs, and flag are all optional
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
        rule = rule.replace('>', ' >').replace('/', ' /').replace('!', ' !').split(' ')
        tars = rule.pop(0)
        #We want to extract just the first iteration of (reps, envs, excs) and store everything else in else_
        #To do this, we observe that if we fill in missing fields once we reach a later field, then if we hit
        #a repeat (by seeing that the field variable is not None) we are in the second iteration. If there is
        #no second iteration, else_ will be None.
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
            else_.insert(0,tars)
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
        if self.flags['ltr']:
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
        
        Returns a Word
        
        Raises WordUnchanged if the word was not changed by the rule.
        '''
        phones = word.phones.copy()
        if self.flags['ltr']:
            word.reverse()
        matches = []
        tars = self.tars
        if not tars:
            tars = [([],[])]
        for i in range(len(tars)):
            if tars[i]:
                tar, indices = tars[i]
            else:
                tar, indices = [], []
            _matches = []
            index = 1
            while True:
                match, _tar = word.find(tar, index, return_match=True) #find the next place where tar matches
                if match == -1: #no more matches
                    break
                index += match
                _matches.append((index, _tar, i))
                index += 1
            if not indices:
                indices = range(len(_matches))
            matches += [_matches[i] for i in indices]
        matches = sorted(matches, reverse=True)
        for match in matches:
            self.apply_match(match, word)
        if self.flags['ltr']:
            word.reverse()
        if word.phones == phones:
            raise WordUnchanged
        return word
    
    def apply_match(self, match, word):
        '''Apply a replacement if a match meets the rule condition.
        
        Arguments:
            match -- the match to be checked
            word  -- the word to check against
        '''
        reps = self.reps.copy()
        index, tar, i = match
        if self.excs: #might need improvement
            for exc in self.excs: #if any exception matches, try checking else_
                if word.match_env(exc, index, tar):
                    if self.else_ is not None:
                        self.else_.apply_match(match, word)
                    return
            for env in self.envs: #if any environment matches, return the match
                if word.match_env(env, index, tar):
                    rep = reps[i]
                    if len(rep) == 1 and isinstance(rep[0], Cat):
                        rep[0] = rep[0][self.tars[i][0][0].index(tar[0])]
                    word.replace(index, tar, rep)
                    return
            #rule failed
        else:
            for env in self.envs: #if any environment matches, return the match
                if word.match_env(env, index, tar):
                    rep = self.reps[i]
                    if len(rep) == 1 and isinstance(rep[0], Cat):
                        rep[0] = rep[0][self.tars[i][0][0].index(tar[0])]
                    word.replace(index, tar, rep)
                    return
            if self.else_ is not None: #try checking else_
                self.else_.apply_match(match, word)

#== Functions ==#
def parse_ruleset(ruleset, cats=None):
    '''Parse a sound change ruleset.
    
    Arguments:
        ruleset -- the set of rules to be parsed (str)
        cats    -- the initial categories to be used to parse the rules (dict)
    
    Returns a list.
    '''
    if cats is None:
        cats = {}
    if isinstance(ruleset, str):
        ruleset = ruleset.splitlines()
    else:
        ruleset = ruleset.copy()
    for i in range(len(ruleset)):
        rule = ruleset[i]
        if rule == '':
            ruleset[i] = None
        elif isinstance(rule, Rule):
            continue
        elif '>' in rule or rule[0] in '+-': #rule is a sound change
            ruleset[i] = Rule(rule, cats)
        else: #rule is a cat definition
            cop = rule.index('=')
            op = (rule[cop-1] if rule[cop-1] in '+-' else '') + '='
            name, vals = rule.split(op)
            exec(f'cats[name] {op} Cat(vals)')
            for cat in list(cats): #discard blank categories
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
            if '~' in env: #~X is equivalent to X_,_X
                _field += Rule.parse_field('{0}_|_{0}'.format(envs[1:]), 'envs', cats)
            elif '_' in env:
                env = env.split('_')
                env = [parse_syms(env[0], cats)[::-1], parse_syms(env[1], cats)]
            else:
                env = [parse_syms(env, cats)]
            _field.append(env)
    else:
        for tar in split(field, ',', nesting=(0,'([{','}])'), minimal=True):
            if mode == 'tars':
                if '@' in tar:
                    tar, index = tar.split('@')
                    indices = split(index, '|', minimal=True)
                    for i in range(len(indices)):
                        indices[i] = int(indices[i])
                else:
                    indices = []
            tar = parse_syms(tar, cats)
            if mode == 'tars':
                tar = (tar, indices)
            _field.append(tar)
    return _field

def parse_flags(flags):
    '''Parse the flags of a sound change rule.
    
    Arguments:
        flags -- the flags to be parsed (str)
        
    Returns a dictionary.
    '''
    _flags = {'ignore':0, 'ltr':0, 'repeat':1, 'age':1} #default values
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

def apply_ruleset(words, ruleset, cats=None, debug=False):
    '''Applies a set of sound change rules to a set of words.
    
    Arguments:
        words   -- the words to which the rules are to be applied (list)
        ruleset -- the rules which are to be applied to the words (list)
        cats    -- the initial categories to be used in ruleset parsing (dict)
    
    Returns a list.
    '''
    words = words.copy()
    if cats is None:
        cats = {}
    ruleset = parse_ruleset(ruleset, cats)
    rules = [] #we use a list to store rules, since they may be applied multiple times
    for rule in ruleset:
        rules.append(rule)
        if debug:
            print('Words =',[str(word) for word in words]) #for debugging
        for i in range(len(words)):
            for rule in reversed(rules):
                if debug:
                    print('rule =',rule) #for debugging
                for j in range(rule.flags['repeat']):
                    try:
                        words[i] = rule.apply(words[i])
                    except WordUnchanged: #if the word didn't change, stop applying
                        break
        for i in reversed(range(len(rules))):
            rules[i].flags['age'] -= 1
            if rules[i].flags['age'] == 0: #if the rule has 'expired', discard it
                del rules[i]
    return words

