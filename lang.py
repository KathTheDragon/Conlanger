'''Create and manipulate languages

Classes:
    Config   -- collection of gen.py configuration data
    Language -- represents a language

Functions:
    load_lang -- load the data from the named language file
    save_lang -- save the given language's data to file
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===

=== Features ===
Add generating every possible word/root

=== Style ===
Consider where to raise/handle exceptions
'''

from .core import Cat, parse_syms, parse_cats, split
from . import gen, sce

# == Classes == #
Config = namedtuple('Config', 'patterns, counts, constraints, freq, monofreq, graphfreq, patternfreq')

class Language:
    '''Class for representing a single language.
    
    Instance variables:
        name        -- language name (str)
        cats        -- grapheme categories (dict)
        wordConfig  -- word configuration data (Config)
        syllabifier -- syllabification function (Syllabifier)
    
    Methods:
        parse_patterns -- parse a string denoting generation patterns
        gen_word       -- generate words
        apply_ruleset  -- apply a sound change ruleset to a wordset
    '''
    
    def __init__(self, name='', cats=None, word_config=None, syllabifier=None):
        '''Constructor for Language().
        
        Arguments:
            name        -- language name (str)
            cats        -- grapheme categories (dict)
            word_config  -- word configuration data (Config)
            root_config  -- root configuration data (Config)
            pattern_freq -- drop-off frequency for patterns (float)
            graph_freq   -- drop-off frequency for graphemes (float)
        '''
        self.name = name
        self.cats = parse_cats(cats)
        if 'graphs' not in self.cats:  # Category 'graphs' must exist
            self.cats['graphs'] = Cat("'")
        if word_config is None:
            self.word_config = Config([], range(0), [], 0, 0, 0, 0)
        else:
            self.word_config = word_config
        self.syllabifier = syllabifier
    
    def parse_patterns(self, patterns):
        '''Parses generation patterns.
        
        Arguments:
            patterns -- set of patterns to parse (str)
        
        Returns a list
        '''
        patterns = split(patterns, ',', minimal=True)
        for i in range(len(patterns)):
            patterns[i] = parse_syms(patterns[i], self.cats)
        return patterns
    
    def gen_word(self, num):
        '''Generates 'num' words.
        
        Arguments:
            num -- number of words to generate, 0 generates every possible word (int)
        
        Returns a list
        '''
        if num == 0:  # Generate every possible word, unimplemented
            return []
        results = []
        for i in range(num):
            results.append(gen.gen_word(self))
        return results
    
    def apply_ruleset(self, wordset, ruleset, to_string=True):
        '''Runs the sound change 'ruleset' on the 'wordset'.
        
        Arguments:
            wordset   -- the words to be changed (str, list)
            ruleset   -- the sound changes to apply (str, list)
            to_string -- whether or not to have string output
        
        Returns a str or list
        '''
        return sce.apply_ruleset(wordset, ruleset, self.cats, self.syllabifier, False, to_string)

# == Functions == #
def load_lang(name):
    '''Loads language data from file.
    
    Arguments:
        name -- the name of the language file to load from
    
    Returns a Language
    '''
    with open('langs/{}.dat'.format(name.lower()), 'r', encoding='utf-8') as f:
        data = list(f)
    name = data[0].strip()
    cats = eval(data[1].strip())
    word_config = eval(data[2].strip())
    return Language(name, cats, word_config)

def save_lang(lang):
    '''Saves a language to file.
    
    Arguments:
        lang -- the Language to save
    '''
    name = lang.name
    cats = str(lang.cats)
    word_config = str(lang.wordConfig)
    data = '\n'.join([name, cats, word_config])
    with open('langs/{}.dat'.format(name.lower()), 'w', encoding='utf-8') as f:
        f.write(data)

