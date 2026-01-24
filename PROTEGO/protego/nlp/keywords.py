"""
keywords.py
============
Domain-specific keyword banks for PROTEGO.

These keywords act as a SAFETY NET to prevent false negatives.
They NEVER replace ML models.

Design goals:
- Phrase-aware matching
- Word-boundary safety
- Explainable keyword hits
- Deterministic behavior
"""

import re
from typing import Iterable, Dict

# -------------------------------------------------
# Emergency / life-threatening
# -------------------------------------------------
EMERGENCY_KEYWORDS = {
    "kill", "killing", "knife", "gun", "weapon", "blood",
    "strangle", "strangling", "choking",
    "attack", "attacking",
    "dying", "death",
    "threatening", "threat",
    "help now", "right now", "immediately"
}

# -------------------------------------------------
# Physical abuse indicators
# -------------------------------------------------
PHYSICAL_ABUSE_KEYWORDS = {
    "hit", "hitting", "beat", "beating",
    "slap", "slapped", "punch", "punched",
    "kick", "kicked", "push", "pushed",
    "bruise", "injury", "hurt"
}

# -------------------------------------------------
# Emotional / psychological abuse
# -------------------------------------------------
EMOTIONAL_ABUSE_KEYWORDS = {
    "insult", "insulting", "shout", "shouting",
    "control", "controlling",
    "humiliate", "humiliated",
    "threaten", "threatening",
    "manipulate", "manipulating"
}

# -------------------------------------------------
# Fear & safety concern
# -------------------------------------------------
FEAR_KEYWORDS = {
    "scared", "afraid", "fear",
    "unsafe", "danger", "terrified",
    "panic", "panicking"
}

# -------------------------------------------------
# Self-harm / hopelessness (PHRASES ONLY)
# -------------------------------------------------
SELF_HARM_KEYWORDS = {
    "i will die",
    "i want to die",
    "i am going to die",
    "kill myself",
    "end my life",
    "no reason to live",
    "i can't go on",
    "i cant go on",
    "i give up",
    "i want everything to stop",
    "i don't want to live",
    "i dont want to live"
}

# -------------------------------------------------
# Keyword matching engine
# -------------------------------------------------
def keyword_hits(text: str, keywords: Iterable[str]) -> int:
    """
    Count keyword/phrase hits in text.

    - Single words use word-boundary matching
    - Phrases use substring matching
    """

    if not isinstance(text, str) or not text.strip():
        return 0

    text_lower = text.lower()
    hits = 0

    for kw in keywords:
        kw = kw.lower().strip()

        # Phrase match (contains space)
        if " " in kw:
            if kw in text_lower:
                hits += 1
        else:
            # Word-boundary safe match
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                hits += 1

    return hits


# -------------------------------------------------
# Explainable keyword scan (optional)
# -------------------------------------------------
def keyword_explain(text: str, keywords: Iterable[str]) -> Dict[str, int]:
    """
    Return which keywords were matched and how many times.
    Useful for debugging and audits.
    """

    if not isinstance(text, str) or not text.strip():
        return {}

    text_lower = text.lower()
    result = {}

    for kw in keywords:
        kw = kw.lower().strip()

        if " " in kw:
            count = text_lower.count(kw)
        else:
            count = len(re.findall(rf"\b{re.escape(kw)}\b", text_lower))

        if count > 0:
            result[kw] = count

    return result
