"""
Guardrail helpers — parse inline refusals from the single LangChain agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from moderation.safety_layer import REFUSAL_MESSAGES

GUARDRAIL_REFUSAL_PATTERN = re.compile(
    r"^\[GUARDRAIL_REFUSAL:([a-z_]+)\]\s*\n?",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class GuardrailDecision:
    """Decision from an inline guardrail refusal."""
    blocked: bool
    reason: Optional[str] = None
    category: Optional[str] = None
    confidence: float = 1.0


GUARDRAIL_CATEGORIES = [
    "hate_speech",
    "fake_scripture_injection",
    "adversarial_rewrite",
    "heresy_promotion",
    "ideology_manipulation",
    "image_violation",
    "off_topic_harmful",
    "any_other_topic_except_bible",
]


def parse_guardrail_refusal(response_text: str) -> tuple[str, Optional[GuardrailDecision]]:
    """
    Strip an inline guardrail marker and return cleaned text plus decision.

    Returns:
        (cleaned_response, decision) — decision is None if not a guardrail refusal.
    """
    match = GUARDRAIL_REFUSAL_PATTERN.search(response_text)
    if not match:
        return response_text, None

    category = match.group(1).lower()
    cleaned = GUARDRAIL_REFUSAL_PATTERN.sub("", response_text, count=1).strip()

    return cleaned, GuardrailDecision(
        blocked=True,
        category=category,
        reason=f"Blocked by integrated guardrail ({category})",
        confidence=1.0,
    )


def build_refusal_response(decision: GuardrailDecision) -> dict:
    """Build API metadata for a guardrail refusal."""
    category = decision.category or "general"

    refusal_map = {
        "hate_speech": REFUSAL_MESSAGES["hate_speech"],
        "fake_scripture_injection": REFUSAL_MESSAGES["fake_scripture"],
        "adversarial_rewrite": REFUSAL_MESSAGES["adversarial_rewrite"],
        "heresy_promotion": REFUSAL_MESSAGES["general"],
        "ideology_manipulation": REFUSAL_MESSAGES["adversarial_rewrite"],
        "image_violation": REFUSAL_MESSAGES["image_violation"],
        "off_topic_harmful": REFUSAL_MESSAGES["general"],
    }

    refusal_text = refusal_map.get(category, REFUSAL_MESSAGES["general"])

    return {
        "response": refusal_text,
        "guardrail_triggered": True,
        "guardrail_category": category,
        "guardrail_reason": decision.reason,
        "guardrail_confidence": decision.confidence,
        "citations": [],
        "rag_used": False,
        "image_url": None,
    }
