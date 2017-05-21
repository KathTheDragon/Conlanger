'''Base classes and functions

Exceptions:
    LangException -- Base exception
    FormatError   -- Error for incorrect formatting

Classes:
    Cat    -- represents a category of phonemes
    Word   -- represents a run of text
    Config -- collection of gen.py configuration data

Functions:
    parse_syms -- parses a string using pattern notation
    split      -- splits a string
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===
Utilise new implementation of Word as sequence type
- investigate reindexing Word
Consider how to subclass Word from list
Break out format checking into separate functions
I want to change supplying the actual syllable boundaries to Word to giving a syllabifier function - this is obviously language-dependent
Perhaps adjust Cat.__init__ to allow sequences of graphemes to be stored

=== Features ===
Implement cat subsets - maybe?

=== Style ===
Consider where to raise/handle exceptions
'''

from collections import namedtuple
from string import whitespace

#== Exceptions ==#
class LangException(Exception):
    '''Base class for exceptions in this package'''

class FormatError(LangException):
    '''Exception raised for errors in formatting objects.'''

#== Classes ==#
class Cat(list):
    '''Represents a category of graphemes.'''
    
    __slots__ = []
    
    def __init__(self, values=None, cats=None):
        '''Constructor for Cat.
        
        Arguments:
            values -- the values in the category (str, list)
            cats   -- dictionary of categories (dict)
        '''
        _values = []
        if values is None:
            values = []
        elif isinstance(values, str): #we want an iteratible with each value as an element
            values = split(values, ',', minimal=True)
        for value in values:
            if isinstance(value, Cat): #another category
                _values.extend(value.values)
            elif '[' in value:
                if cats is not None and value.strip('[]') in cats:
                    _values.extend(cats[value.strip('[]')])
                else:
                    continue
            else:
                _values.append(value)
        list.__init__(self, _values)
    
    def __repr__(self):
        return f"Cat('{self!s}')"
    
    def __str__(self):
        return ', '.join(self)
    
    def __and__(self, cat):
        return Cat(value for value in self if value in cat)
    
    def __add__(self, cat):
        return Cat(list.__add__(self, cat))
    
    def __sub__(self, cat):
        return Cat(value for value in self if value not in cat)

class Word(list):
    '''Represents a word as a list of graphemes.
    
    Instance variables:
        graphs    -- a list of graphemes (list)
        syllables -- a list of tuples representing syllables (list)
    
    Methods:
        find      -- match a list using pattern notation to the word
        match_env -- match a sound change environment to the word
        strip     -- remove leading and trailing graphemes
    '''
    
    __slots__ = ['graphs', 'syllables']
    
    def __init__(self, lexeme=None, graphs=None, syllables=None):
        '''Constructor for Word
        
        Arguments:
            lexeme    -- the word (str)
            syllables -- list of tuples representing syllables (list)
            graphs    -- list of graphemes (list)
        '''
        if graphs is None:
            graphs = ["'"]
        self.graphs = graphs
        if lexeme is None:
            lexeme = []
        elif isinstance(lexeme, str):
            lexeme = parse_word(f' {lexeme} ', self.graphs)
        list.__init__(self, lexeme)
        self.syllables = syllables #do a bit of sanity checking here
    
    def __repr__(self):
        return f"Word('{self!s}')"
    
    def __str__(self):
        word = test = ''
        separator = self.graphs[0]
        polygraphs = (graph for graph in self.graphs if len(graph) > 1)
        for graph in self:
            if not any(graph in poly for poly in polygraphs):
                test = '' #can't ever be ambiguous
            elif not test:
                test = graph #nothing earlier to be ambiguous with
            else:
                test += graph
                if any(test == poly or poly in test for poly in polygraphs):
                    word += separator #ambiguous, so add the separator
                    test = graph
                elif not any(test in poly for poly in polygraphs):
                    test = test[1:] #could still be ambiguous with something later
            word += graph
        return word.strip(separator+'#').replace('#',' ')
    
    def __contains__(self, item):
        if isinstance(item, (list, Word)):
            return self.find(item) != -1
        else:
            return list.__contains__(self, item)
    
    def __add__(self, other):
        return Word(list.__add__(self, other), self.graphs + other.graphs[1:], self.syllables + other.syllables)
    
    def __mul__(self, other):
        return Word(list.__mul__(self, other), self.graphs, self.syllables * other)
    
    def __rmul__(self, other):
        return Word(list.__rmul__(self, other), self.graphs, self.syllables * other)
    
    def copy(self):
        return Word(self, self.graphs, self.syllables)
    
    def strip(self, chars=None):
        if chars is None:
            chars = '#'
        for i in range(len(self)):
            if self[i] not in chars:
                start = i
                break
        else:
            self[:] = []
            return
        for i in reversed(range(len(self))):
            if self[i] not in chars:
                end = i+1
                break
        self[:] = self[start:end]
    
    def find(self, sub, start=None, end=None, return_match=False):
        '''Match a sequence using pattern notation to the word.
        
        Arguments:
            sub   -- the list to be found (list)
            start -- the index of the beginning of the range to check (int)
            end   -- the index of the end of the range to check (int)
        
        Returns an int
        '''
        if start is None:
            start = 0
        elif start < 0:
            start += len(self)
        if end is None:
            end = len(self)
        elif end < 0:
            end += len(self)
        sub = sub.copy()
        if isinstance(sub, Word):
            sub.strip() #we want to strip out the leading and trailing '#'s so that this works like finding substrings
        for i in range(0, end-start):
            j = i + start #position in the word
            for k, sym in enumerate(sub):
                if j >= end: #we've reached the end of the slice, so the find fails
                    return (-1, []) if return_match else -1
                elif isinstance(sym, tuple): #optional sequence
                    index = self.find(list(sym)+sub[k+1:], j, end, return_match)
                    if return_match:
                        index, match = index
                    if index == 0: #try with the optional sequence
                        return (i, self[i+start:j]+match) if return_match else i
                    j -= 1 #if this fails, we jump back to where we were
                elif isinstance(sym, Cat): #category
                    if not self[j] in sym: #this may change - definitely if categories are allowed to contain sequences
                        break
                elif sym == '*': #wildcard
                    index = self.find(sub[k+1:],j, end, return_match)
                    if return_match:
                        index, match = index
                    if index != -1: #only fails if the rest of the sequence is nowhere present
                        return (i, self[i+start:index+j]+match) if return_match else i
                    break
                elif self[j] != sym: #grapheme
                    break
                j += 1
            else:
                return (i, self[i+start:j]) if return_match else i
        else:
            return (-1, []) if return_match else -1
    
    def match_env(self, env, pos=0, tar=None): #test if the env matches the word
        '''Match a sound change environment to the word.
        
        Arguments:
            env -- the environment to be matched (list)
            pos -- the index of the left edge of the target (int)
            tar -- the target (list)
        
        Returns a bool
        '''
        env = env.copy()
        if tar is None:
            tar = []
        for i in reversed(range(len(env))):
            if env[i] == '%':
                env[i:i+1] = tar
            if env[i] == '<':
                env[i:i+1] = reversed(tar)
        if len(env) == 1:
            return env[0] in self
        else:
            if pos:
                matchLeft = self[::-1].find(env[0],-pos)
            else: #at the left edge, which can only be matched by a null env
                matchLeft = -1 if env[0] else 0
            matchRight = self.find(env[1], pos+len(tar))
            return matchLeft == matchRight == 0

Config = namedtuple('Config', 'patterns, counts, constraints, freq, monofreq')

#== Functions ==#
def parse_syms(syms, cats=None):
    '''Parse a string using pattern notation.
    
    Arguments:
        syms -- the input string using pattern notation (str)
        cats -- a list of cats to use for interpreting categories (list)
    
    Returns a list
    '''
    if cats is None:
        cats = {}
    for char in '([{}])':
        syms = syms.replace(char, f' {char} ')
    syms = split(syms, ' ', nesting=(0, '([{','}])'), minimal=True)
    for i in reversed(range(len(syms))):
        syms[i] = syms[i].replace(' ','')
        if not syms[i]:
            del syms[i]
        elif syms[i][0] == '(': #optional - parse to tuple
            syms[i] = tuple(parse_syms(syms[i].strip('()'), cats))
        elif syms[i][0] == '[': #category - parse to Cat
            syms[i] = syms[i].strip('[]')
            if ',' in syms[i]: #nonce cat
                syms[i] = Cat(syms[i])
            else: #named cat
                syms[i] = cats[syms[i]]
        elif syms[i][0] == '{': #unimplemented - delete
            del syms[i]
        else: #text - parse as word
            syms[i:i+1] = parse_word(syms[i])
    return syms

def parse_word(word, graphs=None):
    '''Parse a string of graphemes.
    
    Arguments:
        word       -- the word to be parsed (str)
        separator  -- disambiguator character (str)
        polygraphs -- list of polygraphs (list)
    
    Returns a list.
    '''
    #black magic
    test = ''
    if graphs is None:
        graphs = ["'"]
    separator = graphs[0]
    polygraphs = (graph for graph in graphs if len(graph) > 1)
    graphemes = []
    for char in '#'.join(f'.{word}.'.split()).strip('.')+separator: #convert all whitespace to a single #
        test += char
        while len(test) > 1 and not any(g.startswith(test) for g in polygraphs): #while test isn't a single character and doesn't begin any polygraph
            for i in reversed(range(1,len(test)+1)): #from i=len(test) to i=1
                if i == 1 or test[:i] in polygraphs: #does test begin with a valid graph? Single characters are always valid
                    graphemes.append(test[:i]) #add this valid graph to the output
                    test = test[i:].lstrip(separator) #remove the graph from test, and remove leading instances of separator
                    break
    return graphemes

def split(string, sep=None, nesting=None, minimal=False):
    '''String splitting.
    
    Arguments:
        string  -- the string to be split (str)
        sep     -- the character(s) to split on (str)
        nesting -- a tuple of the form (depth, open, close) containing the nesting depth, and opening and closing nesting characters (tuple)
        minimal -- whether or not to perform the minimal number of splits, similar to str.split() with no arguments
    
    Returns a list.
    '''
    if sep is None:
        sep = whitespace
        minimal = True
    result = []
    depth = 0
    while True:
        if minimal and (nesting is None or depth == nesting[0]):
            string = string.lstrip(sep)
        for i in range(len(string)):
            if string[i] in sep and (nesting is None or depth == nesting[0]):
                result.append(string[:i])
                string = string[i+1:]
                break
            elif nesting is not None and string[i] in nesting[1]:
                depth += 1
            elif nesting is not None and string[i] in nesting[2]:
                depth -= 1
        else:
            if not minimal or string != '':
                result.append(string)
            break
    return result

