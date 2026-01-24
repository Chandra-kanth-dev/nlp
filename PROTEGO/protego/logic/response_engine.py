"""
response_engine.py
==================
Generates safe, empathetic, context-aware responses
for the PROTEGO chatbot based on FINAL risk level.
"""

import random
from typing import Dict, Optional

# -----------------------------
# Message templates (VARIANTS)
# -----------------------------

LOW_RISK_MESSAGES = [
    "I’m here with you 🤍\nYou can talk to me about anything that’s on your mind.",
    "You’re not alone. I’m listening whenever you’re ready to share.",
    "Take your time. Whatever you’re feeling, it’s okay to talk about it here."
]

MEDIUM_RISK_MESSAGES = [
    "Thank you for telling me.\nWhat you’re feeling matters, and you’re not overreacting.\n\nWould you like to share more?",
    "I hear you. That sounds difficult, and your feelings are valid.\n\nDo you want to tell me what’s been happening?",
    "I’m really glad you spoke up.\nWhat you’re experiencing deserves attention and care."
]

HIGH_RISK_MESSAGES = [
    "I’m really concerned about your safety.\nWhat you’re experiencing is serious and not okay.\n\nPlease consider reaching out to someone you trust.",
    "This sounds very serious, and your safety matters.\nYou deserve help and protection.",
    "I’m worried about you.\nNo one has the right to make you feel unsafe."
]

EMERGENCY_MESSAGES = [
    "🚨 **Immediate danger detected** 🚨\n\nPlease try to move to a safer place if you can.\nCall emergency services right now.\n\nYour safety matters. I’m here with you.",
    "🚨 **Your safety is at risk** 🚨\n\nIf possible, contact emergency services or a trusted person immediately."
]

# -----------------------------
# Public API (UNCHANGED)
# -----------------------------
def generate_response(
    emotion: str,
    final_risk: str,
    emergency_contacts: Optional[Dict[str, str]] = None
) -> Dict[str, object]:
    """
    Generate final chatbot response based on FINAL risk.
    """

    if final_risk == "emergency":
        return {
            "message": random.choice(EMERGENCY_MESSAGES),
            "show_emergency": True,
            "tone": "urgent"
        }

    if final_risk == "high":
        return {
            "message": random.choice(HIGH_RISK_MESSAGES),
            "show_emergency": True,
            "tone": "serious"
        }

    if final_risk == "medium":
        return {
            "message": random.choice(MEDIUM_RISK_MESSAGES),
            "show_emergency": False,
            "tone": "concerned"
        }

    return {
        "message": random.choice(LOW_RISK_MESSAGES),
        "show_emergency": False,
        "tone": "gentle"
    }
