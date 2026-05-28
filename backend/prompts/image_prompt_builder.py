"""
Image Prompt Builder — Safe image prompt construction with content filtering.
Prevents generation of theologically inappropriate or harmful images.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Concepts that are BLOCKED for image generation
BLOCKED_IMAGE_CONCEPTS = [
    "face of jesus", "face of god", "face of allah", "prophet",
    "crucifixion gore", "religious violence", "burning church",
    "antichrist", "666", "satanic", "occult ritual",
    "political figure as jesus", "jesus as",
    "rewrite bible", "bible supports", "scripture says",
    "jesus holding", "god holding", "jesus with gun",
    "jesus fighting", "god fighting", "holy war",
    "destroy mosque", "destroy synagogue", "destroy temple",
    "religious figure", "portrait of jesus", "portrait of god",
    "face of christ", "realistic jesus", "realistic god",
    "jesus face", "god face", "christ face",
]

# Safe style prefixes for different art styles
SAFE_STYLE_PREFIXES = {
    "realistic": "Photorealistic, cinematic lighting, high detail",
    "painterly": "Oil painting style, warm tones, classical religious art inspiration",
    "symbolic": "Symbolic, abstract, stained-glass inspired, no human faces",
    "minimalist": "Minimalist vector illustration, soft colors, clean lines",
}

# Denomination-specific image notes
DENOMINATION_NOTES = {
    "catholic": "Include appropriate Catholic symbols (crucifix, rosary, saints' icons if relevant).",
    "orthodox": "Include Orthodox Christian elements (icons, gold leaf style, Byzantine inspiration).",
    "protestant": "Simple, scripture-focused imagery. Avoid saints' iconography.",
    "evangelical": "Contemporary Christian art style. Modern and inviting.",
    "nondenominational": "Universal Christian themes. Avoid denomination-specific symbols.",
}


@dataclass
class ImagePromptResult:
    """Result from building a safe image prompt."""
    prompt: str
    blocked: bool
    reason: str | None = None


def _fuzzy_match_blocked(concept: str) -> tuple[bool, str | None]:
    """
    Check if a concept matches any blocked image concept using fuzzy matching.

    Args:
        concept: The image concept to check.

    Returns:
        Tuple of (is_blocked, matched_concept).
    """
    concept_lower = concept.lower().strip()

    for blocked in BLOCKED_IMAGE_CONCEPTS:
        # Exact substring match
        if blocked in concept_lower:
            return True, blocked

        # Word-level fuzzy match
        blocked_words = set(blocked.split())
        concept_words = set(concept_lower.split())
        overlap = blocked_words & concept_words

        # If most of the blocked phrase's words are present, it's a match
        if len(blocked_words) > 1 and len(overlap) >= len(blocked_words) * 0.7:
            return True, blocked

    return False, None


def build_safe_image_prompt(
    concept: str,
    style: str = "painterly",
    denomination_sensitivity: str | None = None,
) -> ImagePromptResult:
    """
    Build a safe, filtered image generation prompt.

    Steps:
    1. Check concept against BLOCKED_IMAGE_CONCEPTS (fuzzy match)
    2. If blocked, return blocked result with reason
    3. Build safe prompt with style prefix and safety guardrails
    4. Add denomination-specific notes if provided

    Args:
        concept: The image concept description.
        style: Art style ("realistic", "painterly", "symbolic", "minimalist").
        denomination_sensitivity: Optional denomination for style adaptation.

    Returns:
        ImagePromptResult with the safe prompt or blocked status.
    """
    # Step 1: Check against blocked concepts
    is_blocked, matched = _fuzzy_match_blocked(concept)
    if is_blocked:
        return ImagePromptResult(
            prompt="",
            blocked=True,
            reason=(
                f"Image concept matches blocked category: '{matched}'. "
                "This type of imagery could be theologically inappropriate or harmful. "
                "Consider requesting symbolic, landscape, or abstract Christian imagery instead."
            ),
        )

    # Step 2: Get style prefix
    style_prefix = SAFE_STYLE_PREFIXES.get(style, SAFE_STYLE_PREFIXES["painterly"])

    # Step 3: Sanitize concept — remove potential prompt injections
    sanitized = concept.strip()
    # Remove any attempt to override instructions
    sanitized = re.sub(r'(?i)(ignore|override|forget|disregard)\s+(previous|above|all)', '', sanitized)
    sanitized = re.sub(r'(?i)system\s*prompt', '', sanitized)

    # Step 4: Build the full prompt
    prompt_parts = [
        style_prefix,
        sanitized,
        "Christian theme.",
        "No human faces.",
        "No text overlays.",
        "No historically inaccurate depictions.",
        "Respectful and reverent tone.",
    ]

    # Step 5: Add denomination-specific notes
    if denomination_sensitivity:
        denom_note = DENOMINATION_NOTES.get(
            denomination_sensitivity.lower(),
            DENOMINATION_NOTES["nondenominational"]
        )
        prompt_parts.append(denom_note)

    prompt = " ".join(prompt_parts)

    return ImagePromptResult(
        prompt=prompt,
        blocked=False,
        reason=None,
    )
