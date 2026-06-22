"""
Category extractor - parses ChatTS/GPT-4o text responses into category codes.
Handles A-E, G, L, and composite labels (B+L).

The full extract_category function is from Cell 56 of the original notebook.
It supports all 7 anomaly types plus multi-label answers.
"""

import re


LABEL_NAMES = {
    'A':   'Drift',
    'B':   'Spikes',
    'C':   'Frozen',
    'D':   'Phase Change',
    'E':   'None/Clean',
    'G':   'Var Collapse',
    'L':   'Intermittent Fail',
    'B+L': 'Spikes + Intermittent Fail',
    '?':   'Unclear',
}


def extract_category(
    answer: str,
    detected: list | None = None,
) -> tuple[str, str]:
    """
    Extract anomaly category letter(s) from a model text response.

    Parameters
    ----------
    answer   : raw model response string
    detected : pre-screener findings list (used to hard-confirm some categories)

    Returns
    -------
    (code, label) e.g. ('B', 'Spikes') or ('B+L', 'Spikes + Intermittent Fail')
    """
    up = answer[:200].upper()

    # Pattern 0 - hard confirm B+L when pre-screener found both
    if detected and 'intermittent' in detected and 'spikes' in detected:
        return 'B+L', 'Spikes + Intermittent Fail'

    # Pattern 0b - hard confirm L when physical evidence is in answer
    if detected and 'intermittent' in detected:
        if re.search(r'NEGATIVE|IMPOSSIBLE|CANNOT BE|PHYSICAL', up[:200]):
            return 'L', 'Intermittent Fail'

    # Pattern 1 - first line starts with letter (most reliable)
    first_line = answer.strip().split('\n')[0].upper()
    m = re.search(r'^([ABCDEGL])\)', first_line.strip())
    if m:
        c = m.group(1)
        return c, LABEL_NAMES.get(c, '?')

    # Pattern 2 - multi-label "B, L" or "B and L" or "B+L"
    multi = re.findall(r'\b([ABCDEGL])\b', up[:80])
    valid = [c for c in multi if c in 'ABCDEGL']
    if len(valid) >= 2 and len(set(valid)) >= 2:
        cats = list(dict.fromkeys(valid[:3]))
        if detected and any(d in detected for d in ['spikes', 'intermittent']):
            cats = [c for c in cats if c != 'A']
        if not cats:
            cats = valid[:1]
        if len(cats) == 1:
            return cats[0], LABEL_NAMES.get(cats[0], '?')
        code = '+'.join(cats)
        label = ' + '.join(LABEL_NAMES.get(c, '?') for c in cats)
        return code, label

    # Pattern 3 - "A)" or "Answer: A" anywhere in first 200 chars
    for pattern in [
        r'\b([ABCDEGL])\)',
        r'ANSWER[:\s]+([ABCDEGL])',
        r'^([ABCDEGL])[\s\.\,\)]',
    ]:
        m = re.search(pattern, up.strip())
        if m:
            c = m.group(1)
            return c, LABEL_NAMES.get(c, '?')

    # Pattern 4 - stale/frozen textual evidence
    if re.search(r'STD_MID\s*[=<]\s*0\.0|FROZEN SEGMENT|FLATLINE', up):
        return 'C', 'Frozen'

    # Pattern 5 - variance collapse textual evidence
    if re.search(r'VARIANCE COLLAPSE|AMPLITUDE COLLAPSES|ENVELOPE COLLAPSE', up):
        return 'G', 'Var Collapse'

    # Pattern 6 - intermittent failure textual evidence
    if re.search(r'NEGATIVE.*IMPOSSIBLE|PHYSICALLY IMPOSSIBLE|CANNOT BE NEGATIVE', up):
        return 'L', 'Intermittent Fail'

    # Pattern 7 - clean signal description without anomaly keywords
    if re.search(r'BASELINE.{0,50}STABLE|REFERENCE LINE.{0,50}STABLE', up[:150]):
        if not re.search(r'SPIKE|FROZEN|PHASE CHANGE|VARIANCE COLLAPSE', up[:150]):
            return 'E', 'None/Clean'

    return '?', 'Unclear'
