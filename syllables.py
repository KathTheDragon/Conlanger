from dataclasses import dataclass

@dataclass
class Syllabifier:
    rules: tuple

    def __init__(self, cats, onsets=(), nuclei=(), codas=(), margins=(), constraints=()):
        from ._pattern import parsePatterns
        onsets = parsePatterns(onsets)
        nuclei = parsePatterns(nuclei)
        codas = parsePatterns(codas)
        margins = parsePatterns(margins)
        constraints = parsePatterns(constraints)
        rules = []
        rules.extend(generateNonFinals(codas, onsets, nuclei))    # Medials
        rules.extend(generateFinals(codas, margins))              # Finals
        rules.extend(generateNonFinals(margins, onsets, nuclei))  # Initials
        self.rules = tuple(rule for rule in self.rules if checkValid(rule[0], constraints))

    def __call__(self, word):
        breaks = []
        # Step through the word
        pos = 0
        while pos < len(word):
            for rule, _breaks in self.rules:
                if rule == ['_', '#'] and pos in breaks:
                    continue
                match, rpos = word.matchPattern(rule, pos)[:2]
                if match:
                    # Compute and add breaks for this pattern
                    for ix in _breaks:
                        # Syllable breaks must be within the word and unique
                        if 0 < pos+ix < len(word) and pos+ix not in breaks:
                            breaks.append(pos+ix)
                    # Step past this match
                    pos = rpos
                    if rule[-1] == '#':
                        pos -= 1
                    break
            else:  # No matches here
                pos += 1
        return tuple(breaks)

def generateNonFinals(codas, onsets, nuclei):
    rules = []
    for crank, coda in enumerate(codas):
        if coda[-1] == '#':
            continue
        elif coda[-1] == '_':
            coda = coda[:-1]
        for orank, onset in enumerate(onsets):
            if onset[0] == '#':
                if coda == ['#']:
                    onset = onset[1:]
                else:
                    continue
            if onset == ['_']:
                onset = []
            for nrank, nucleus in enumerate(nuclei):
                if nucleus[0] == '#':
                    if coda == ['#'] and onset == []:
                        nucleus = nucleus[1:]
                    else:
                        continue
                pattern = coda + onset + nucleus
                breaks = [len(coda)]
                if pattern[-1] == '#':
                    breaks.append(len(pattern)-1)
                rank = crank + orank + nrank
                rules.append((pattern, breaks, rank))
    return (r[:2] for r in sorted(rules, key=lambda r: r[2]))

def generateFinals(codas, margins):
    rules = []
    for mrank, margin in enumerate([margin for margin in margins if margin[-1] == '#']):
        if margin == ['_', '#']:
            margin = ['#']
        for crank, coda in enumerate(codas):
            if coda[-1] == '#':
                if margin == ['#']:
                    coda = coda[:-1]
                else:
                    continue
            pattern = coda + margin
            breaks = [0 if coda == ['_'] else len(coda)]
            rank = crank + mrank
            rules.append((pattern, breaks, rank))
    return (r[:2] for r in sorted(rules, key=lambda r: r[2]))

def checkValid(rule, constraints):
    for constraint in constraints:
        for rpos in range(len(rule)-len(constraint)):
            for cpos, ctoken in enumerate(constraint):
                rtoken = rule[rpos+cpos]
                if isinstance(rtoken, str) and isinstance(ctoken, str):
                    if rtoken == ctoken:
                        continue
                elif isinstance(rtoken, str) and isinstance(ctoken, Cat):
                    if rtoken in ctoken:
                        continue
                elif isinstance(rtoken, Cat) and isinstance(ctoken, Cat):
                    if rtoken <= ctoken:
                        continue
                break
            else:
                return False
    return True
