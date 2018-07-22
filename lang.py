'''Create and manipulate languages

Classes:
    Config   -- collection of gen.py configuration data
    Language -- represents a language

Functions:
    unparse_pattern -- unparse a generation pattern back to a string
    load_lang       -- load the data from the named language file
    save_lang       -- save the given language's data to file
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Implement confirming overwriting save data - needs UI
Maybe add different modes for each positional syllable type

=== Features ===
Add generating every possible word/root
Language.apply_ruleset will be replaced by calls to the diachronics module, once that exists
Add constraints to phonotactics
- Primarily for word gen, potentially useful for PhonoSyllabifier

=== Style ===
Consider where to raise/handle exceptions
'''

from collections import namedtuple
import os
import json
from .core import Cat, RulesSyllabifier, PhonoSyllabifier, parse_patterns, parse_cats, unparse_word
from . import gen, sce

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Language files are in conlanger/langs/

# == Classes == #
Config = namedtuple('Config', 'patterns, constraints, sylrange, sylmode, patternmode, graphmode')

class Language:
    '''Class for representing a single language.
    
    Instance variables:
        name        -- language name (str)
        cats        -- grapheme categories (dict)
        wordConfig  -- word configuration data (Config)
        syllabifier -- syllabification function (RulesSyllabifier)
    
    Methods:
        gen_word      -- generate words
        apply_ruleset -- apply a sound change ruleset to a wordset
    '''
    __slots__ = ('name', 'cats', 'configs', 'phonotactics', 'syllabifier')
    
    def __init__(self, name='', cats=None, configs=None, phonotactics=None, syllabifier=None):
        '''Constructor for Language().
        
        Arguments:
            name    -- language name (str)
            cats    -- grapheme categories (dict)
            configs -- configuration data sets (dict)
        '''
        self.name = name
        self.cats = parse_cats(cats)
        if 'graphs' not in self.cats:  # Category 'graphs' must exist
            self.cats['graphs'] = Cat("'")
        self.configs = {}
        if configs is None:
            configs = {}
        for config in configs:
            _config = configs[config].copy()
            _config['patterns'] = parse_patterns(_config['patterns'], self.cats)
            _config['constraints'] = parse_patterns(_config['constraints'], self.cats)
            _config['sylrange'] = range(_config['sylrange'][0], _config['sylrange'][1]+1)
            self.configs[config] = Config(**_config)
        self.phonotactics = parse_patterns(phonotactics, self.cats)
        # Need some default phonotactics instead of empty lists
        if not self.phonotactics['onsets']:
            self.phonotactics['onsets'] = [['_']]
        if not self.phonotactics['codas']:
            self.phonotactics['codas'] = [['_']]
        while '#' in self.phonotactics['margins']:
            ix = self.phonotactics['margins'].index('#')
            del self.phonotactics['margins'][ix]
        if all(m[0] != '#' for m in self.phonotactics['margins']):
            self.phonotactics['margins'].append(['#', '_'])
        if all(m[-1] != '#' for m in self.phonotactics['margins']):
            self.phonotactics['margins'].append(['_', '#'])
        if syllabifier is not None:
            self.syllabifier = RulesSyllabifier(self.cats, parse_patterns(syllabifier, self.cats))
        else:
            self.syllabifier = PhonoSyllabifier(self.cats, **self.phonotactics)
    
    @property
    def data(self):
        data = {}
        if self.name != '':
            data['name'] = self.name
        if self.cats != {}:
            data['cats'] = {name: list(cat) for name, cat in self.cats.items()}
        if self._configs != {}:
            data['configs'] = self._configs
        if isinstance(self.syllabifier, RulesSyllabifier):
            data['syllabifier'] = []
            for rule in self.syllabifier.rules:
            	rule, indices = rule
            	rule = rule.copy()
            	for i in reversed(indices):
            		rule.insert(i, '$')
            	data['syllabifier'].append(unparse_pattern(rule))
        if self.phonotactics is not None:
            data['phonotactics'] = {k: [unparse_pattern(pattern) for pattern in v] for k, v in self.phonotactics.items()}
        return data
    
    def gen(self, config, num=1):
        '''Generates 'num' words using 'config'.
        
        Arguments:
            config -- config data to use
            num    -- number of words to generate, 0 generates every possible word (int)
        
        Returns a list
        '''
        if config not in self.configs:
            return []
        if num == 0:  # Generate every possible word, unimplemented
            return []
        return [gen.gen_word(self.configs[config], self.cats['graphs'], self.syllabifier) for i in range(num)]
    
    def apply_ruleset(self, wordset, ruleset, to_string=False):
        '''Runs the sound change 'ruleset' on the 'wordset'.
        
        Arguments:
            wordset   -- the words to be changed (str, list)
            ruleset   -- the sound changes to apply (str, list)
            to_string -- whether or not to have string output
        
        Returns a str or list
        '''
        return sce.run(wordset, ruleset, self.cats, self.syllabifier, to_string)

# == Functions == #
def load_lang(name):
    '''Loads language data from file.
    
    Arguments:
        name -- the name of the language file to load from
    
    Returns a Language
    '''
    with open('langs/{}.dat'.format(name.lower()), 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Language(**data)

def save_lang(lang):
    '''Saves a language to file.
    
    Arguments:
        lang -- the Language to save
    '''
    data = lang.data
    # Check for existing save data
    with open('langs/{}.dat'.format(name.lower()), 'r+', encoding='utf-8') as f:
        if f.read():
            if True:  # Check if the user wants to overwrite this data - not implemented yet
                f.truncate()
            else:
                return
        json.dump(data)

def unparse_pattern(pattern):
    for i, token in reversed(list(enumerate(pattern))):
        if isinstance(token, list) and not isinstance(token, Cat) and token[-1] == '?':
            del pattern[i][-1]
            pattern.insert(i+1, '?')
        # Add collapsing repeated tokens
    for i, token in reversed(list(enumerate(pattern))):
        if isinstance(token, int):  # Integer repetition
            pattern[i] = f'{{{pattern[i]}}}'
        elif isinstance(token, tuple):  # Wildcard repetition and comparison
            if isinstance(token[-1], int):
                token = list(token)
                token[-1] = str(token[-1])
            pattern[i] = f'{{{"".join(token)}}}'
        elif isinstance(token, Cat):
            if token.name is not None:
                pattern[i] = f'[{token.name}]'
            else:
                pattern[i] = f'[{token}]'
        elif isinstance(token, list):
            pattern[i] = f'({unparse_pattern(token)})'
    return unparse_word(pattern)

def getcwd():
    print(os.getcwd())
