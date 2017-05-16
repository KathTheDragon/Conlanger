'''Create and manipulate languages

Classes:
    Language -- represents a language

Functions:
    load_lang -- load the data from the named language file
    save_lang -- save the given language's data to file
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Work out where the cat parsing code is going

=== Features ===
Add generating every possible word/root

=== Style ===
Consider where to raise/handle exceptions
'''

from core import Cat, Config, parse_syms, split
import gen

#== Classes ==#
class Language():
    '''Class for representing a single language.
    
    Instance variables:
        name        -- language name (str)
        cats        -- grapheme categories (dict)
        wordConfig  -- word configuration data (Config)
        rootConfig  -- root configuration data (Config)
        patternFreq -- drop-off frequency for patterns (float)
        graphFreq   -- drop-off frequency for graphemes (float)
    
    Methods:
        parse_patterns -- parse a string denoting generation patterns
        gen_word       -- generate words
        gen_root       -- generate roots
    '''
    
    def __init__(self, name='', cats=None, wordConfig=None, rootConfig=None, patternFreq=0, graphFreq=0):
        '''Constructor for Language().
        
        Arguments:
            name        -- language name (str)
            cats        -- grapheme categories (dict)
            wordConfig  -- word configuration data (Config)
            rootConfig  -- root configuration data (Config)
            patternFreq -- drop-off frequency for patterns (float)
            graphFreq   -- drop-off frequency for graphemes (float)
        '''
        self.name = name
        if cats is None:
            self.cats = {}
        else:
            self.cats = cats
        #need to keep this but not sure where exactly it's being moved to
        # cats = cats.replace('|',' ').split()
        # for cat in cats:
            # name, vals = cat.split('=')
            # vals = vals.replace(',',' ').split()
            # if not vals: #this would yeild an empty cat
                # continue
            # for i in range(len(vals)):
                # if '[' in vals[i]: #this is another category
                    # vals[i] = self.cats[vals[i][1:-1]]
            # self.cats[name] = Cat(vals)
        # if 'graphs' not in self.cats or not self.cats['graphs']: #category 'graphs' must exist and contain at least one character
            # self.cats['graphs'] = Cat("'")
        # for cat in self.cats.keys(): #discard blank categories
            # if not self.cats[cat]:
                # del self.cats[cat]
        if wordConfig is None:
            self.wordConfig = Config([],range(0),[],0,0)
        else:
            self.wordConfig = wordConfig
        if rootConfig is None:
            self.rootConfig = Config([],range(0),[],0,0)
        else:
            self.rootConfig = rootConfig
        self.patternFreq = patternFreq
        self.graphFreq = graphFreq
    
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
        if num == 0: #generate every possible word, unimplemented
            return []
        results = []
        for i in range(num):
            results.append(gen.gen_word(self))
        return results
    
    def gen_root(self, num):
        '''Generates 'num' roots.
        
        Arguments:
            num -- number of roots to generate, 0 generates every possible root (int)
        
        Returns a list
        '''
        if num == 0: #generate every possible word, unimplemented
            return []
        results = []
        for i in range(num):
            results.append(gen.gen_root(self))
        return results

#== Functions ==#
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
    wordConfig = eval(data[2].strip())
    rootConfig = eval(data[3].strip())
    patternFreq = eval(data[4].strip())
    graphFreq = eval(data[5].strip())
    return Language(name, cats, wordConfig, rootConfig, patternFreq, graphFreq)

def save_lang(lang):
    '''Saves a language to file.
    
    Arguments:
        lang -- the Language to save
    '''
    name = lang.name
    cats = str(lang.cats)
    wordConfig = str(lang.wordConfig)
    rootConfig = str(lang.rootConfig)
    patternFreq = str(lang.patternFreq)
    graphFreq = str(lang.graphFreq)
    data = '\n'.join([name, cats, wordConfig, rootConfig, patternFreq, graphFreq])
    with open('langs/{}.dat'.format(name.lower()), 'w', encoding='utf-8') as f:
        f.write(data)

