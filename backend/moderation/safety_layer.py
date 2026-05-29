"""
Safety Layer — Three-layer content safety system.

Layer 1 (Input):  Guardrail agent (pre-agent)
Layer 2 (Output): Post-agent response scanning
Layer 3 (Image):  Image prompt safety (pre-generation)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Graceful refusal messages for each category
REFUSAL_MESSAGES = {
    "hate_speech": (
        "I'm not able to help with that. Christianity calls us to love all people — "
        "I'd be glad to help you explore what scripture says about love, reconciliation, "
        "or understanding."
    ),
    "fake_scripture": (
        "I want to be careful with scripture references. I wasn't able to verify that verse "
        "in my knowledge base. Would you like me to search for actual passages on that topic?"
    ),
    "adversarial_rewrite": (
        "Rewriting scripture to support a particular ideology isn't something I'm able to do — "
        "it would risk misrepresenting God's Word. I'm happy to explore what different Christian "
        "traditions have said about this topic using their actual teachings."
    ),
    "image_violation": (
        "I'm not able to generate that particular image. I can create respectful Christian-themed "
        "imagery — landscapes, symbols, abstract faith concepts. What theme would you like to explore?"
    ),
    "general": (
        "That's not something I'm able to help with in this context. I'm here to support your "
        "exploration of Christianity, faith, and scripture. What else can I help you with today?"
    ),
}


@dataclass
class SafetyCheckResult:
    """Result from post-processing safety check."""
    passed: bool
    warnings: list[str] = field(default_factory=list)
    modified_response: str | None = None


async def post_process_safety(
    agent_response: str,
    agent_tool_calls: list[dict],
) -> SafetyCheckResult:
    """
    Layer 2: Post-agent output safety check.

    Scans the agent response for:
    1. Unverified verse citations (regex for verse patterns without verification evidence)
    2. Hallucination markers (phrases like "as the Bible says" without citations)
    3. Content that may have slipped past the guardrail

    Args:
        agent_response: The agent's generated response text.
        agent_tool_calls: List of tool calls made during generation.

    Returns:
        SafetyCheckResult with warnings and optional modified response.
    """
    warnings = []
    modified = agent_response

    # Check 1: Scan for verse citations without verification
    verse_pattern = re.compile(r'(\d?\s*\w+)\s+(\d+):(\d+)(?:-\d+)?')
    verse_matches = list(verse_pattern.finditer(agent_response))

    # Citations are considered grounded if the agent either verified the reference
    # or retrieved supporting passages from the knowledge base via rag_tool.
    grounded = any(
        tc.get("name") in ("scripture_verify_tool", "rag_tool")
        for tc in agent_tool_calls
    )

    if verse_matches and not grounded:
        warnings.append(
            "Response contains scripture references but neither scripture_verify_tool "
            "nor rag_tool was called. Citations may be unverified."
        )

    # Check 2: Hallucination markers without citations
    hallucination_phrases = [
        "as the bible says",
        "the bible tells us",
        "scripture tells us",
        "the word of god says",
        "according to the bible",
        "the bible clearly states",
    ]

    response_lower = agent_response.lower()
    for phrase in hallucination_phrases:
        if phrase in response_lower:
            # Check if there's a citation nearby
            phrase_pos = response_lower.find(phrase)
            surrounding = agent_response[max(0, phrase_pos - 50):phrase_pos + len(phrase) + 100]
            if not verse_pattern.search(surrounding):
                warnings.append(
                    f"Response contains '{phrase}' without a specific citation nearby. "
                    "This may indicate unsupported claims."
                )

    # Check 3: Check for potentially fabricated book names
    known_fake_books = [
        "hesitations", "opinions", "chronicles of narnia",
        "book of mormon", "suggestions",
    ]
    for fake in known_fake_books:
        if fake in response_lower:
            # Only flag if it's being cited as scripture
            context_start = max(0, response_lower.find(fake) - 30)
            context = response_lower[context_start:context_start + 80]
            if any(indicator in context for indicator in ["verse", "chapter", ":", "says"]):
                warnings.append(
                    f"Response may reference a non-canonical text ('{fake}'). "
                    "Verify this is not being presented as Biblical scripture."
                )

    passed = len(warnings) == 0

    return SafetyCheckResult(
        passed=passed,
        warnings=warnings,
        modified_response=modified if modified != agent_response else None,
    )


def get_refusal_message(category: str) -> str:
    """Get the appropriate refusal message for a category."""
    return REFUSAL_MESSAGES.get(category, REFUSAL_MESSAGES["general"])
