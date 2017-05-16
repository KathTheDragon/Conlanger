'''Generate syllables, words, or roots

Exceptions:
    ExceededMaxRunsError -- used to terminate generation

Functions:
    dist       -- provides a power law distribution
    peaked_dist -- provides a peaked power law distribution
    populate   -- generates a list of graphemes according to a pattern
    gen_word    -- generates a word
    gen_root    -- generates a root
''''''
==================================== To-do ====================================
=== Bug-fixes ===
Doesn't seem to be checking exceptions correctly (not urgent-urgent)

=== Implementation ===
Might be worth having gen_word() create the Word() at the beginning rather than the end
Look into utilising decomposition theorem in syllable generation - might need to add more to pyle.Language

=== Features ===

=== Style ===
Consider where to raise/handle exceptions

''''''
=== Mathematical model ===
r is the number of segments
p is the 'dropoff rate'
f(n) = p**n*(1-p)/(1-p**r) is the frequency of the nth most frequent segment (frequencies sum to 1)

p must be determined by observing that a = (1-p)/(1-p**r) is the frequency of the most frequent segment. From this, we can estimate p ≈ 1-a, and with a first-order correction, p ≈ (1-a)+(a*(1-a)**r)/(1-a*r*(1-a)**(r-1)).

P(n) = (1-p**n)/(1-p**r) is the cumulative frequency of the first n segments, and can be found by summing over f(n)

A probability distribution can then be obtained by finding the inverse of P(n). Let x be a continuous random variable from 0 to 1 (say, random.random()). Then n = floor(log(1-x*(1-p**r),p))

Obtaining a variant with a peak can be done by using two distributions, one reversed, with their modes overlapping. This can be done by taking the range of x corresponding to the reversed section and rescaling it as follows, where a is the frequency of the mode, and c the cumulative frequency of the bins before the mode: x -> 1-x/(a+c). Thus, when x<c, we use a distribution with m+1 bins, mode a/(a+c), and the rescaled random variable. For the remainder, we use a distribution with r-m bins, mode a/(1-c), and a rescaled variable x -> (x-c)/(1-c). Note that the mode belongs to this second distribution.
'''

from core import LangException, Cat, Word
from random import random, choice
from math import log, floor, ceil

#== Constants ==#
MAX_RUNS = 10**5 #maximum number of times something can fail to be generated

#== Exceptions ==#
class ExceededMaxRunsError(LangException):
    '''Exception raised when something has failed to be generated too many times.'''

#== Functions ==#
def dist(bins, a=0, x=None): #first bin has frequency a, random variable x
    '''Returns an element of 'bins' according to a power law distribution.
    
    Arguments:
        bins -- a non-empty ordered collection of elements (str, list, tuple)
        a    -- the frequency that the first bin should be selected (0 for equiprobable distribution) (float)
        x    -- a random variable supplied if the default random.random() is not desired (float)
    '''
    #See the docstring titled 'Mathematical Model' for the maths
    r = len(bins)
    if a <= 0: #use equiprobable distribution instead
        return choice(bins)
    if r == 1 or a >= 1: #only one bin
        return bins[0]
    if x is None: #no random variable supplied
        x = random()
    p = (1-a)+(a*(1-a)**r)/(1-a*r*(1-a)**(r-1))
    return bins[floor(log(1-x*(1-p**r),p))]

def peaked_dist(bins, a=0, m=0, c=0):
    '''Returns an element of 'bins' according to a peaked power law distribution.
    
    Arguments:
        bins -- an ordered collection of elements (str, list, tuple)
        a    -- the frequency that the most frequent bin should be selected (0 for equiprobable distribution) (float)
        m    -- the index of the most frequent bin
        c    -- the cumulative frequency of bins 0 to m-1
    '''
    #See the docstring titled 'Mathematical Model' for the maths
    if m <= 0 or c <= 0: #all bins before the mode are ignored
        return dist(bins[m:], a)
    x = random()
    if x < c: #in the left-hand branch
        return dist(bins[m::-1], a/(a+c), 1-x/(a+c))
    else:
        return dist(bins[m:], a/(1-c), (x-c)/(1-c))

def populate(pattern, frequency, all=False):
    '''Generate a word section according to 'pattern'
    
    Arguments:
        pattern   -- the pattern to generate (list)
        frequency -- grapheme drop-off frequency (float)
        all       -- indicator to generate every possible pattern, or one random pattern (bool)
    '''
    if not all: #one random syllable
        result = []
        for seg in pattern:
            if isinstance(seg, Cat):
                result.append(dist(seg, frequency))
            elif seg == "'":
                result.append(result[-1])
            else:
                result.append(seg)
        return result
    else: #every possible syllable
        results = [[]]
        for seg in pattern:
            if isinstance(seg, Cat):
                temp = []
                for result in results:
                    for sym in seg:
                        temp.append(result+[sym])
                results = temp
            elif seg == "'":
                for i in range(len(results)):
                    results[i].append(results[i][-1])
            else:
                for i in range(len(results)):
                    results[i].append(seg)
        return results

def gen_word(lang):
    '''Generate a single word as specified by 'lang'.
    
    Arguments:
        lang -- the language the word is to be generated for (Language)
    
    Returns a Word
    
    Raises ExceededMaxRunsError when the word repeatedly fails to be valid
    '''
    word = ['#']
    patterns, counts, constraints, frequency, monofreq = lang.wordConfig
    pattFreq, phonFreq = lang.patternFreq, lang.graphFreq
    sylCount = peaked_dist(counts, frequency, 1, monofreq)
    for i in range(sylCount-1): #generate all but the final syllable
        for j in range(MAX_RUNS):
            pattern = dist(patterns, pattFreq)
            syl = populate(pattern, phonFreq)+['$']
            _word = Word(word+syl)
            for env in constraints:
                if env in _word:
                    break
            else:
                word += syl #if so, keep it, else try a new syllable
                break
        else:
            raise ExceededMaxRunsError()
    for j in range(MAX_RUNS):
        pattern = dist(patterns, pattFreq)
        syl = populate(pattern, phonFreq)+['#'] #generate final syllable
        _word = Word(word+syl)
        for env in constraints:
            if env in _word:
                break
        else:
            word += syl #if so, keep it, else try a new syllable
            sylEdges = [1]+[i-word[:i].count('$') for i in range(len(word)) if word[i] == '$']
            while '$' in word:
                word.remove('$')
            return Word(word, sylEdges, lang.cats['graphs'])
    else:
        raise ExceededMaxRunsError()

def gen_root(lang):
    '''Generate a single root as specified by 'lang'.
    
    Arguments:
        lang -- the language the root is to be generated for (Language)
    
    Returns a Word
    
    Raises ExceededMaxRunsError when the root repeatedly fails to be valid
    '''
    #generate a root according to rootPatterns
    root = []
    patterns, counts, constraints, frequency, monofreq = lang.rootConfig
    pattFreq, phonFreq = lang.patternFreq, lang.graphFreq
    sylCount = peaked_dist(counts, frequency, 1, monofreq)
    for i in range(sylCount): #generate all but the final syllable
        for j in range(MAX_RUNS):
            pattern = dist(patterns, pattFreq)
            syl = populate(pattern, phonFreq)
            _root = Word(root+syl)
            for env in constraints:
                if env in _root:
                    break
            else:
                root += syl
                break
        else:
            raise ExceededMaxRunsError()
    return Word(root, None, lang.cats['graphs'])

