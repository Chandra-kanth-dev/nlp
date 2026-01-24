"""
preprocess.py
================
Centralized NLP preprocessing module for PROTEGO.

Design goals:
- Deterministic and safe preprocessing
- Preserve emotional and safety signals
- Avoid runtime failures in production
- ML-friendly normalization
"""

import re
from functools import lru_cache
from typing import List

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# -------------------------------------------------
# IMPORTANT:
# NLTK resources MUST be installed at build time.
# Do NOT download during runtime in production.
# -------------------------------------------------

# -------------------------------------------------
# Global NLP tools (load once)
# -------------------------------------------------
_LEMMATIZER = WordNetLemmatizer()

# Load stopwords safely
try:
    _STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    raise RuntimeError(
        "NLTK stopwords not found. "
        "Run: python -m nltk.downloader stopwords wordnet"
    )

# Negations are emotionally important — keep them
NEGATION_WORDS = {
    "not", "no", "never", "dont", "can't", "cant",
    "won't", "wont", "isn't", "isnt", "aren't", "arent"
}

_STOP_WORDS = _STOP_WORDS - NEGATION_WORDS

# -------------------------------------------------
# Regex patterns (compiled once)
# -------------------------------------------------
URL_PATTERN = re.compile(r"http\S+|www\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z!?'\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")
REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{2,}")  # soooo → soo

# -------------------------------------------------
# Core preprocessing function
# -------------------------------------------------
@lru_cache(maxsize=1024)
def clean_text(text: str) -> str:
    """
    Clean and normalize input text while preserving
    emotional and safety-critical signals.
    """

    if not isinstance(text, str):
        return ""

    # Trim & lowercase
    text = text.strip().lower()
    if not text:
        return ""

    # Normalize elongated characters (panic signals)
    text = REPEAT_CHAR_PATTERN.sub(r"\1\1", text)

    # Remove URLs
    text = URL_PATTERN.sub("", text)

    # Remove non-text noise but KEEP ! ? '
    text = NON_ALPHA_PATTERN.sub(" ", text)

    # Normalize whitespace
    text = MULTISPACE_PATTERN.sub(" ", text)

    # Tokenize
    tokens = text.split()

    # Stopword removal + lemmatization
    cleaned_tokens = []
    for token in tokens:
        if token not in _STOP_WORDS:
            lemma = _LEMMATIZER.lemmatize(token)
            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


# -------------------------------------------------
# Batch utility (safe)
# -------------------------------------------------
def preprocess_batch(texts: List[str]) -> List[str]:
    """
    Apply clean_text to a batch of texts.
    """

    if not isinstance(texts, list):
        return []

    return [clean_text(t) for t in texts if isinstance(t, str)]
