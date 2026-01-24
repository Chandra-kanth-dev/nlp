"""
safety_rules.py
================
Rule-based safety enforcement module for PROTEGO.

This module is the FINAL authority in risk decisions.
It deterministically overrides ML-based outputs when
user safety could be compromised.

Design principles:
- Safety-first escalation
- Deterministic rule priority
- Explainable overrides
- Conservative emergency handling
"""

from typing import Dict

from protego.nlp.keywords import (
    EMERGENCY_KEYWORDS,
    PHYSICAL_ABUSE_KEYWORDS,
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
# Safety rule engine
# -------------------------------------------------
def apply_safety_rules(
    text: str,
    current_risk: str,
    context_summary: Dict[str, object]
) -> Dict[str, object]:
    """
    Apply deterministic, priority-ordered safety rules.

    Returns:
        Dict with:
        - final_risk (str)
        - rule_triggered (bool)
        - rule_reason (str or None)
        - rule_priority (int or None)
    """

    text_lower = text.lower()
    final_risk = current_risk

    # Default response
    result = {
        "final_risk": current_risk,
        "rule_triggered": False,
        "rule_reason": None,
        "rule_priority": None
    }

    # -------------------------------------------------
    # RULE 1 (Priority 1): Immediate emergency keywords
    # -------------------------------------------------
    if keyword_hits(text_lower, EMERGENCY_KEYWORDS) >= 1:
        result.update({
            "final_risk": "emergency",
            "rule_triggered": True,
            "rule_reason": "Immediate emergency keyword detected",
            "rule_priority": 1
        })
        return result

    # -------------------------------------------------
    # RULE 2 (Priority 2): Repeated high-risk pattern
    # -------------------------------------------------
    if context_summary.get("repeated_high_risk"):
        result.update({
            "final_risk": "emergency",
            "rule_triggered": True,
            "rule_reason": "Repeated high or emergency risk in conversation",
            "rule_priority": 2
        })
        return result

    # -------------------------------------------------
    # RULE 3 (Priority 3): Physical abuse indicators
    # -------------------------------------------------
    if keyword_hits(text_lower, PHYSICAL_ABUSE_KEYWORDS) >= 1:
        if RISK_ORDER.get(current_risk, 1) < RISK_ORDER["high"]:
            result.update({
                "final_risk": "high",
                "rule_triggered": True,
                "rule_reason": "Physical abuse indicator detected",
                "rule_priority": 3
            })
            return result

    # -------------------------------------------------
    # RULE 4 (Priority 4): Escalating risk trend
    # -------------------------------------------------
    if context_summary.get("is_escalating"):
        if current_risk == "medium":
            result.update({
                "final_risk": "high",
                "rule_triggered": True,
                "rule_reason": "Escalating risk trend detected",
                "rule_priority": 4
            })
            return result

    # -------------------------------------------------
    # RULE 5 (Priority 5): Emergency persistence
    # -------------------------------------------------
    if "emergency" in context_summary.get("recent_risks", []):
        result.update({
            "final_risk": "emergency",
            "rule_triggered": True,
            "rule_reason": "Previous emergency detected in conversation",
            "rule_priority": 5
        })
        return result

    # -------------------------------------------------
    # No override
    # -------------------------------------------------
    return result
