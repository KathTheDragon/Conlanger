'''Generate syllables, words, or roots

==================================== To-do ====================================
=== Bug-fixes ===
Doesn't seem to be checking exceptions correctly (not urgent-urgent)

=== Implementation ===
Potentially going to be overhauled in the near future

=== Features ===

=== Style ===
Consider where to raise/handle exceptions

=== Mathematical model ===
r is the number of segments
p is the 'dropoff rate'
f(n) = p**n*(1-p)/(1-p**r) is the frequency of the nth most frequent segment (frequencies sum to 1)

p must be determined by observing that a = (1-p)/(1-p**r) is the frequency of the most frequent segment. From this, we
can estimate p ≈ 1-a, and with a first-order correction, p ≈ (1-a)+(a*(1-a)**r)/(1-a*r*(1-a)**(r-1)).

P(n) = (1-p**n)/(1-p**r) is the cumulative frequency of the first n segments, and can be found by summing over f(n)

A probability distribution can then be obtained by finding the inverse of P(n). Let x be a continuous random variable
from 0 to 1 (say, random.random()). Then n = floor(log(1-x*(1-p**r),p))

Obtaining a variant with a peak can be done by using two distributions, one reversed, with their modes overlapping. This
can be done by taking the range of x corresponding to the reversed section and rescaling it as follows, where a is the
frequency of the mode, and c the cumulative frequency of the bins before the mode: x -> 1-x/(a+c). Thus, when x<c, we
use a distribution with m+1 bins, mode a/(a+c), and the rescaled random variable. For the remainder, we use a
distribution with r-m bins, mode a/(1-c), and a rescaled variable x -> (x-c)/(1-c). Note that the mode belongs to this
second distribution.
'''

from .core import LangException, Cat, Word
from random import random, choice
from math import log, floor, ceil

# == Constants == #
MAX_RUNS = 10**5  # maximum number of times something can fail to be generated

# == Exceptions == #
class ExceededMaxRunsError(LangException):
    '''Exception raised when something has failed to be generated too many times.'''

# == Functions == #
def dist(bins, a=0, x=None):  # First bin has frequency a, random variable x
    '''Returns an element of 'bins' according to a power law distribution.

    Arguments:
        bins -- a non-empty ordered collection of elements (str, list, tuple)
        a    -- the frequency that the first bin should be selected (0 for equiprobable distribution) (float)
        x    -- a random variable supplied if the default random.random() is not desired (float)
    '''
    # See the docstring titled 'Mathematical Model' for the maths
    r = len(bins)
    if a <= 0:  # Use equiprobable distribution instead
        return choice(bins)
    if r == 1 or a >= 1:  # Only one bin
        return bins[0]
    if x is None:  # No random variable supplied
        x = random()
    p = (1-a)+(a*(1-a)**r)/(1-a*r*(1-a)**(r-1))
    return bins[floor(log(1-x*(1-p**r), p))]

def peakedDist(bins, a=0, m=0, c=0):
    '''Returns an element of 'bins' according to a peaked power law distribution.

    Arguments:
        bins -- an ordered collection of elements (str, list, tuple)
        a    -- the frequency that the most frequent bin should be selected (0 for equiprobable distribution) (float)
        m    -- the index of the most frequent bin
        c    -- the cumulative frequency of bins 0 to m-1
    '''
    # See the docstring titled 'Mathematical Model' for the maths
    if m <= 0 or c <= 0:  # All bins before the mode are ignored
        return dist(bins[m:], a)
    x = random()
    if x < c:  # In the left-hand branch
        return dist(bins[m::-1], a/(a+c), 1-x/(a+c))
    else:
        return dist(bins[m:], a/(1-c), (x-c)/(1-c))

def populate(pattern, mode):
    '''Generate a word section according to 'pattern'

    Arguments:
        pattern -- the pattern to generate (list)
        mode    -- representation of the mode of the grapheme distribution (list)
        all     -- indicator to generate every possible pattern, or one random pattern (bool)
    '''
    result = []
    for token in pattern:
        if token.type == 'category':
            result.append(peakedDist(token.cat, *mode))
        elif token == '"':
            result.append(result[-1])
        else:
            result.append(str(token))
    return result

def populateAll(pattern):
    results = [[]]
    for token in pattern:
        if token.type == 'category':
            temp = []
            for result in results:
                for graph in token.cat:
                    temp.append(result+[graph])
            results = temp
        elif token == '"':
            for result in results:
                result.append(result[-1])
        else:
            for result in results:
                result.append(str(token))
    return results

def genFromConfig(config, graphs=None, separator='', syllabifier=None):
    '''Generate a single word as specified by the 'config'.

    Arguments:
        config -- the config data to be used to generate this word
        graphs -- the set of graphemes used for this word

    Returns a Word

    Raises ExceededMaxRunsError when the word repeatedly fails to be valid
    '''
    word = Word(['#'], graphs, separator, syllabifier)
    patterns, constraints, sylrange, sylmode, patternmode, graphmode = config
    sylcount = peakedDist(sylrange, *sylmode)
    for i in range(sylcount):
        if sylcount == 1:  # Monosyllable
            _patterns = patterns['mono'] or patterns['init'] or patterns['term'] or patterns['medi']
        elif i == 0:  # Initial syllable
            _patterns = patterns['init'] or patterns['medi']
        elif i == sylcount-1:  # Final syllable
            _patterns = patterns['term'] or patterns['medi']
        else:  # Medial syllable
            _patterns = patterns['medi']
        for j in range(MAX_RUNS):
            pattern = peakedDist(_patterns, *patternmode)
            syl = populate(pattern, graphmode)
            _word = word + syl
            for constraint in constraints:
                if constraint and constraint in _word:
                    break
            else:
                word = _word
                break
        else:
            raise ExceededMaxRunsError()
    return word + '#'

def genFromPhonotactics(phonotactics, sylrange=(1,), sylmode=(), graphs=None, syllabifier=None):
    '''Generate a single word as specified by the 'phonotactics'.

    Arguments:
        phonotactics  -- the phonotactic data to be used
        graphs      -- the set of graphemes used for this word
        syllabifier -- the syllabifier used for syllabification

    Returns a Word
    '''
    word = Word([], graphs, syllabifier)
    sylcount = peakedDist(sylrange, *sylmode)
    for i in range(sylcount):
        # Generate a syllable
        for _ in range(MAX_RUNS):
            # Pick an onset
            onset = selectPeriphery(phonotactics['onsets'], phonotactics['margins'], 'left', i)
            # Pick a coda
            coda = selectPeriphery(phonotactics['codas'], phonotactics['margins'], 'right', i-sylcount)
            # Pick a nucleus
            nuclei = phonotactics['nuclei']
            if onset != ['#']:
                nuclei = [nucleus for nucleus in nuclei if nucleus[0] != '#']
            if coda != ['#']:
                nuclei = [nucleus for nucleus in nuclei if nucleus[-1] != '#']
            nucleus = choice(nuclei)
            syl = populate(onset+nucleus+coda, ())
            _word = word + syl
            for env in phonotactics['constraints']:
                if env and env in _word:
                    break
            else:
                word = _word
                break
        else:
            raise ExceededMaxRunsError()
    return word

def selectPeriphery(peripheries, margins, edge, i):
    edge = 0 if edge == 'left' else -1
    if i == edge:
        margin = choice([margin for margin in margins if margin[edge] == '#'])
        if margin == (['_', '#'] if edge else ['#', '_']):
            margin = ['#']
        peripheries = [(p+margin if edge else margin+p) if p[edge] != '#' else p for p in peripheries]
    else:
        peripheries = [p for p in peripheries if p[edge] != '#']
    periphery = choice(peripheries)
    if edge and periphery[0] == '_':
        return periphery[1:]
    elif not edge and periphery[-1] == '_':
        return periphery[:-1]
    else:
        return periphery
