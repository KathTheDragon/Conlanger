'''Create and manipulate languages

Classes:
    Config   -- collection of gen.py configuration data
    Language -- represents a language

Functions:
    parse_patterns -- parse a string of generation patterns
    load_lang      -- load the data from the named language file
    save_lang      -- save the given language's data to file
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Implement confirming overwriting save data - needs UI
Maybe add different modes for each positional syllable type

=== Features ===
Add generating every possible word/root
Language.apply_ruleset will be replaced by calls to the diachronics module, once that exists

=== Style ===
Consider where to raise/handle exceptions
'''

from collections import namedtuple
import os
import json
from .core import Cat, RulesSyllabifier, PhonoSyllabifier, parse_syms, parse_cats, split
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
    __slots__ = ('name', 'cats', '_configs', 'configs', '_syllabifier', 'rulessyllabifier', 'phonosyllabifier')
    
    def __init__(self, name='', cats=None, configs=None, syllabifier=None):
        '''Constructor for Language().
        
        Arguments:
            name    -- language name (str)
            cats    -- grapheme categories (dict)
            configs -- configuration data sets (dict)
        '''
        self.name = name
        self.cats = {}
        if cats is not None:
            for cat in cats:
                self.cats[cat] = Cat(cats[cat], self.cats)
        if 'graphs' not in self.cats:  # Category 'graphs' must exist
            self.cats['graphs'] = Cat("'")
        self.configs = {}
        if configs is None:
            configs = {}
        self._configs = configs  # We need to store the raw input so that we can retrieve it for saving to file
        for config in configs:
            _config = configs[config].copy()
            _config['patterns'] = {k: parse_patterns(v, self.cats) for k,v in _config['patterns'].items()}
            _config['constraints'] = parse_patterns(_config['constraints'], self.cats)
            _config['sylrange'] = range(_config['sylrange'][0], _config['sylrange'][1]+1)
            self.configs[config] = Config(**_config)
        self._syllabifier = syllabifier  # We need to store the raw input so that we can retrieve it for saving to file
        self.rulessyllabifier = RulesSyllabifier(self.cats, syllabifier['rules'])
        self.phonosyllabifier = PhonoSyllabifier(self.cats, syllabifier['onsets'], syllabifier['nuclei'], syllabifier['codas'])
    
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
def parse_patterns(patterns, cats=None):
    '''Parses generation patterns.
    
    Arguments:
        patterns -- set of patterns to parse (str or list)
    
    Returns a list
    '''
    if isinstance(patterns, str):
        patterns = split(patterns, ',', minimal=True)
    return [parse_syms(pattern, cats) for pattern in patterns]

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
    data = {'name': lang.name, 'cats': {k: list(v) for k,v in lang.cats.items()}, 'configs': lang._configs, 'syllabifier': lang._syllabifier}
    # Check for existing save data
    with open('langs/{}.dat'.format(name.lower()), 'r+', encoding='utf-8') as f:
        if f.read():
            if True:  # Check if the user wants to overwrite this data - not implemented yet
                f.truncate()
            else:
                return
        json.dump(data)

def getcwd():
    print(os.getcwd())

