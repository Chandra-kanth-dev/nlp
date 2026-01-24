"""
context_memory.py
=================
Conversation context and risk memory manager for PROTEGO.

Advanced capabilities:
- Short-term, privacy-preserving memory
- Risk trend detection with strength scoring
- Escalation awareness
- Explainable summaries for audits & debugging

No raw text is ever stored.
"""

from collections import deque
from typing import List, Dict


class ContextMemory:
    """
    Maintains short-term conversational context for a single user session.
    """

    # Risk hierarchy (authoritative)
    RISK_ORDER = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "emergency": 4
    }

    def __init__(self, max_history: int = 5):
        """
        Args:
            max_history (int): Number of recent interactions to remember
        """
        self.max_history = max_history
        self.risk_history: deque[str] = deque(maxlen=max_history)
        self.emotion_history: deque[str] = deque(maxlen=max_history)

    # -----------------------------
    # Update context
    # -----------------------------
    def update(self, risk: str, emotion: str) -> None:
        """
        Update memory with latest risk and emotion.

        Invalid or unknown values are ignored for safety.
        """

        if risk in self.RISK_ORDER:
            self.risk_history.append(risk)

        if isinstance(emotion, str) and emotion:
            self.emotion_history.append(emotion)

    # -----------------------------
    # Accessors
    # -----------------------------
    def get_recent_risks(self) -> List[str]:
        return list(self.risk_history)

    def get_recent_emotions(self) -> List[str]:
        return list(self.emotion_history)

    # -----------------------------
    # Trend analysis
    # -----------------------------
    def _numeric_risks(self) -> List[int]:
        return [self.RISK_ORDER[r] for r in self.risk_history]

    def is_escalating(self) -> bool:
        """
        Detects whether overall risk trend is increasing.

        Uses trend direction, not just last step.
        """
        nums = self._numeric_risks()
        if len(nums) < 3:
            return False

        return nums[-1] >= nums[-2] >= nums[-3] and nums[-1] > nums[0]

    def escalation_strength(self) -> float:
        """
        Returns a normalized escalation score between 0.0 and 1.0.
        """
        nums = self._numeric_risks()
        if len(nums) < 2:
            return 0.0

        delta = nums[-1] - nums[0]
        max_delta = self.RISK_ORDER["emergency"] - self.RISK_ORDER["low"]

        return round(max(delta / max_delta, 0.0), 2)

    def repeated_high_risk(self) -> bool:
        """
        Detects repeated or sustained high/emergency risk.
        """
        if not self.risk_history:
            return False

        last = self.risk_history[-1]

        if last == "emergency":
            return True

        recent = list(self.risk_history)[-3:]
        return recent.count("high") >= 2

    # -----------------------------
    # Explainable summary
    # -----------------------------
    def summary(self) -> Dict[str, object]:
        """
        Returns an explainable snapshot of context state.
        """

        return {
            "recent_risks": self.get_recent_risks(),
            "recent_emotions": self.get_recent_emotions(),
            "is_escalating": self.is_escalating(),
            "escalation_strength": self.escalation_strength(),
            "repeated_high_risk": self.repeated_high_risk()
        }

    # -----------------------------
    # Reset memory
    # -----------------------------
    def reset(self) -> None:
        """
        Clears all stored context (session end).
        """
        self.risk_history.clear()
        self.emotion_history.clear()
