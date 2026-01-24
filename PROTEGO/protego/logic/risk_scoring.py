"""
risk_scoring.py
================
Advanced risk decision engine for PROTEGO.

This module fuses:
- ML model predictions
- Linguistic intensity signals
- Keyword-based safety overrides
- Conversation context

into a single, explainable, safety-first risk decision.
"""

from typing import Dict, List, Optional

from protego.nlp.features import extract_features
from protego.nlp.keywords import (
    EMERGENCY_KEYWORDS,
    PHYSICAL_ABUSE_KEYWORDS,
    FEAR_KEYWORDS,
    SELF_HARM_KEYWORDS,
    keyword_hits
)

# -------------------------------------------------
# Risk hierarchy (authoritative)
# -------------------------------------------------
RISK_ORDER = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "emergency": 4
}

# -------------------------------------------------
# Component weights (tuned for safety bias)
# -------------------------------------------------
EMOTION_WEIGHTS = {
    "fear": 3.0,
    "sadness": 2.0,
    "anger": 1.5,
    "neutral": 0.5
}

SENTIMENT_WEIGHTS = {
    "negative": 1.5,
    "neutral": 0.5,
    "positive": 0.0
}

ML_RISK_WEIGHTS = {
    "low": 1.0,
    "medium": 3.0,
    "high": 6.0,
    "emergency": 9.0
}

# -------------------------------------------------
# Risk thresholds (monotonic & interpretable)
# -------------------------------------------------
RISK_THRESHOLDS = {
    "medium": 4.0,
    "high": 8.0,
    "emergency": 12.0
}

# -------------------------------------------------
# Emergency type detection
# -------------------------------------------------
def detect_emergency_type(text: str) -> Optional[str]:
    """
    Detect explicit emergency categories from text.
    """
    text = text.lower()

    if keyword_hits(text, SELF_HARM_KEYWORDS) >= 1:
        return "self_harm"

    if (
        keyword_hits(text, EMERGENCY_KEYWORDS) >= 1
        or keyword_hits(text, PHYSICAL_ABUSE_KEYWORDS) >= 1
    ):
        return "external_threat"

    return None


# -------------------------------------------------
# Core scoring function
# -------------------------------------------------
def compute_risk(
    text: str,
    emotion: str,
    sentiment: str,
    ml_risk: str,
    previous_risks: List[str] | None = None
) -> Dict[str, object]:
    """
    Compute final risk decision with explainability.
    """

    if previous_risks is None:
        previous_risks = []

    score = 0.0
    explanations = []

    # -----------------------------
    # 1️⃣ ML & affective baseline
    # -----------------------------
    score += EMOTION_WEIGHTS.get(emotion, 0.5)
    score += SENTIMENT_WEIGHTS.get(sentiment, 0.5)
    score += ML_RISK_WEIGHTS.get(ml_risk, 1.0)

    explanations.append("baseline_ml_emotion_sentiment")

    # -----------------------------
    # 2️⃣ Linguistic intensity signals
    # -----------------------------
    features = extract_features(text)

    score += min(features.get("urgency_count", 0), 3) * 2.0
    score += min(features.get("intensity", 1.0), 3.0) * 2.0
    score += min(features.get("repetition_score", 0), 4) * 0.5

    explanations.append("linguistic_intensity")

    # -----------------------------
    # 3️⃣ Keyword escalation (capped)
    # -----------------------------
    score += min(keyword_hits(text, PHYSICAL_ABUSE_KEYWORDS), 3) * 3.0
    score += min(keyword_hits(text, FEAR_KEYWORDS), 4) * 1.5
    score += min(keyword_hits(text, SELF_HARM_KEYWORDS), 2) * 4.0

    explanations.append("keyword_escalation")

    # -----------------------------
    # 4️⃣ Context escalation (trend-aware)
    # -----------------------------
    if previous_risks:
        recent = previous_risks[-3:]
        numeric = [RISK_ORDER.get(r, 1) for r in recent]

        if numeric[-1] >= numeric[0] and max(numeric) >= RISK_ORDER["high"]:
            score += 1.5
            explanations.append("context_escalation")

    # -----------------------------
    # 5️⃣ Emergency override (safety-first)
    # -----------------------------
    emergency_type = detect_emergency_type(text)
    emergency_override = emergency_type is not None

    # -----------------------------
    # 6️⃣ Final risk mapping
    # -----------------------------
    final_risk = _map_score_to_risk(score, emergency_override)

    return {
        "risk_score": round(score, 2),
        "final_risk": final_risk,
        "emergency_override": emergency_override,
        "emergency_type": emergency_type,
        "explanations": explanations,
        "debug": {
            "emotion": emotion,
            "sentiment": sentiment,
            "ml_risk": ml_risk,
            "features": features,
            "previous_risks": previous_risks
        }
    }


# -------------------------------------------------
# Score → risk mapping
# -------------------------------------------------
def _map_score_to_risk(score: float, emergency_override: bool) -> str:
    """
    Map numerical score to discrete risk level.
    """

    if emergency_override or score >= RISK_THRESHOLDS["emergency"]:
        return "emergency"
    if score >= RISK_THRESHOLDS["high"]:
        return "high"
    if score >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"
