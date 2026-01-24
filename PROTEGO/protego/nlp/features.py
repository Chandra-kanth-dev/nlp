"""
features.py
============
Linguistic and behavioral feature extraction for PROTEGO.

This module extracts non-ML signals that complement
emotion, sentiment, and risk classification models.

Design goals:
- Safety-oriented signals
- Robust to noisy input
- Explainable features
- Normalized outputs
"""

import re
from typing import Dict

# -------------------------------------------------
# Domain-specific signal vocabularies
# -------------------------------------------------
URGENCY_WORDS = {
    "now", "today", "immediately", "urgent", "asap", "right", "help"
}

FIRST_PERSON_WORDS = {
    "i", "me", "my", "mine", "myself"
}

NEGATION_WORDS = {
    "not", "no", "never", "dont", "can't", "cant", "won't", "wont"
}

DISTRESS_PHRASES = {
    "help me",
    "please help",
    "i can't",
    "i cant",
    "i am scared",
    "i feel unsafe"
}

# -------------------------------------------------
# Feature extraction
# -------------------------------------------------
def extract_features(text: str) -> Dict[str, float]:
    """
    Extract linguistic and psychological features from raw text.

    Returns normalized, safety-oriented signals.
    """

    if not isinstance(text, str) or not text.strip():
        return _empty_features()

    raw_text = text.strip()
    text_lower = raw_text.lower()

    tokens = re.findall(r"\b[a-z']+\b", text_lower)
    word_count = len(tokens)
    unique_word_count = len(set(tokens))

    # -----------------------------
    # Lexical features
    # -----------------------------
    urgency_count = sum(1 for w in tokens if w in URGENCY_WORDS)
    first_person_count = sum(1 for w in tokens if w in FIRST_PERSON_WORDS)
    negation_count = sum(1 for w in tokens if w in NEGATION_WORDS)

    # -----------------------------
    # Phrase-level distress
    # -----------------------------
    distress_phrase_hits = sum(
        1 for phrase in DISTRESS_PHRASES if phrase in text_lower
    )

    # -----------------------------
    # Repetition / rumination
    # -----------------------------
    repetition_score = max(word_count - unique_word_count, 0)

    # -----------------------------
    # Punctuation & casing intensity
    # -----------------------------
    exclamation_count = raw_text.count("!")
    question_count = raw_text.count("?")
    uppercase_ratio = (
        sum(1 for c in raw_text if c.isupper()) / max(len(raw_text), 1)
    )

    # -----------------------------
    # Normalized intensity
    # -----------------------------
    length_intensity = min(word_count / 20.0, 1.0)

    punctuation_intensity = min(
        (exclamation_count + question_count) / 5.0,
        1.0
    )

    # -----------------------------
    # Final feature vector
    # -----------------------------
    return {
        "word_count": float(word_count),
        "urgency_count": float(urgency_count),
        "first_person_ratio": round(first_person_count / max(word_count, 1), 2),
        "negation_count": float(negation_count),
        "repetition_score": float(repetition_score),
        "distress_phrase_hits": float(distress_phrase_hits),
        "uppercase_ratio": round(uppercase_ratio, 2),
        "punctuation_intensity": round(punctuation_intensity, 2),
        "intensity": round(length_intensity, 2)
    }


# -------------------------------------------------
# Empty feature fallback
# -------------------------------------------------
def _empty_features() -> Dict[str, float]:
    """
    Default feature values for empty or invalid input.
    """
    return {
        "word_count": 0.0,
        "urgency_count": 0.0,
        "first_person_ratio": 0.0,
        "negation_count": 0.0,
        "repetition_score": 0.0,
        "distress_phrase_hits": 0.0,
        "uppercase_ratio": 0.0,
        "punctuation_intensity": 0.0,
        "intensity": 0.0
    }
