'''Implementing parsing from PhoMo to SCE

Functions:
    translate         -- translate a PhoMo rule into SCE
    trans_replacement -- translate the replacement fields
    trans_condition   -- translate the condition fields
''''''
==================================== To-do ====================================
=== Bug-fixes ===

=== Implementation ===

=== Features ===

=== Style ===
'''

import re

regex = re.compile(r'([A-Z])', r'[\1]')

def translate(rule):
    # Preliminary conversions
    # Grab flags
    flags = ' '
    if '!stop' in rule:
        flags += 'stop'
        rule = rule.split('!')[0]
    if '/"' in rule:
        if flags != ' ':
            flags += ';'
        flags += 'ditto'
        rule = rule.replace('"', '')
    # Surround categories with [], change ? to <, split rule
    rule = regex.sub(rule.replace('?', '<')).split('/')  # Note that the category stuff doesn't really work - PhoMo categories can be Unicode capitals too
    while len(rule) < 5:
        rule.append('')
    trg, chg, env, exp, els = rule[:5]
    # Translate fields
    tars, reps, mov_env, otherwise = trans_replacement(trg, chg, els)
    envs = trans_condition(env)
    excs = trans_condition(exp)
    if mov_env and envs:
        envs = mov_env + '&' + envs
    elif mov_env:
        envs = mov_env
    # Add field markers
    if tars and not reps:  # Deletion
        tars = '- ' + tars
    elif reps and not tars:  # Epenthesis
        reps = '+ ' + reps
    else:  # Substitution
        reps = ' > ' + reps
    if envs:
        envs = ' / ' + envs
    if excs:
        excs = ' ! ' + excs
    if otherwise:
        otherwise = ' > ' + otherwise
    return tars+reps+envs+excs+otherwise+flags

def trans_replacement(trg, chg, els):
    tars = run = idx = mov_env = None
    otherwise = els
    if '^' in chg:
        chg, run = chg.split('^')
    if '@' in chg:
        chg, idx = chg.split('@')
    if trg == '#':  # Affix rule
        if chg.startswith('>'):  # Move/copy rule
            if chg.startswith('>!'):  # Copy
                reps = '^'
            else:  # Move
                reps = '^?'
            chg = chg.strip('>!')
            if '_' in chg:  # Environments
                mov_env = chg
                chg = None
            else:  # Indices
                idx = '@' + idx
            reps += idx
            idx = chg
        elif chg.startswith('<'):  # Reversal
            idx = chg.strip('<')
            reps = '<'
        elif '[' in chg:  # Affix category
            if chg.startswith('#'):  # Suffix
                reps = '^_#'
            else:  # Prefix
                reps = '^#_'
            tars = chg.strip('#')
        else:
            reps = chg.replace('#', '%').replace('-', '#')
        otherwise = els.replace('#', '%').replace('-', '#')
        if tars is None and run is None and (idx is None or reps is '<'):
            tars = '*'
        else:
            tars = '*?'
        if run is not None:
            tars += '{' + run + '}'
    else:  # Replace rule
        tars = trg
        reps = chg
        otherwise = els
    if idx is not None:
        tars += '@' + idx
    return tars, reps, mov_env, otherwise

def trans_condition(cond):
    if '_' not in cond:  # Global condition
        if cond.startswith('#'):  # Word ends with X
            cond = cond.strip('#') + '#'
        elif cond.endswith('#'):  # Word starts with X
            cond = '#' + cond.strip('#')
        elif '=' in cond:  # Count
            cond, count = cond.split('=')
            if count[0] not in '<>':
                count = '=' + count
            cond += '{' + count + '}'
    return cond

