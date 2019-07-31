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

=== Style ===
Consider where to raise/handle exceptions
'''

from collections import namedtuple
import os
import json
from .core import Cat, Syllabifier, parse_patterns, parse_cats, unparse_word
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
        phonotactics = parse_patterns(phonotactics, self.cats)
        self.phonotactics = {'nuclei': phonotactics['nuclei'], 'margins': []}
        # Need some default phonotactics instead of empty lists
        self.phonotactics['onsets'] = phonotactics['onsets'] or parse_patterns('_')
        self.phonotactics['codas'] = phonotactics['codas'] or parse_patterns('_')
        for margin in phonotactics['margins']:
            if (margin[0] == '#') != (margin[-1] == '#'):
                self.phonotactics['margins'].append(margin)
        if not any((margin[0] == '#') for margin in self.phonotactics['margins']):
            self.phonotactics['margins'].extend(parse_patterns('#_'))
        if not any((margin[-1] == '#') for margin in self.phonotactics['margins']):
            self.phonotactics['margins'].extend(parse_patterns('_#'))
        self.syllabifier = Syllabifier(self.cats, **self.phonotactics)

    @property
    def data(self):
        data = {}
        if self.name != '':
            data['name'] = self.name
        if self.cats != {}:
            data['cats'] = {name: list(cat) for name, cat in self.cats.items()}
        if self._configs != {}:
            data['configs'] = self._configs
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

    def apply_ruleset(self, wordset, ruleset, output='list'):
        '''Runs the sound change 'ruleset' on the 'wordset'.

        Arguments:
            wordset   -- the words to be changed (str, list)
            ruleset   -- the sound changes to apply (str, list)
            to_string -- whether or not to have string output

        Returns a str or list
        '''
        return sce.run(wordset, ruleset, self.cats, self.syllabifier, output)

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
    # Add collapsing repeated tokens
    for i, token in reversed(list(enumerate(pattern))):
        if isinstance(token, int):  # Integer repetition
            pattern[i] = f'{{{pattern[i]}}}'
        # This probably should be moved to _pattern
        elif token.type == 'Optional':
            pattern[i] = f'({unparse_pattern(token.pattern)})'
            if not token.greedy:
                pattern[i] = pattern[i] + '?'
        else:
            pattern[i] = str(token)
    return unparse_word(pattern)

def getcwd():
    print(os.getcwd())
