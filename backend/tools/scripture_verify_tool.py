"""
Scripture Verify Tool — Verifies Bible verse references before citation.
Prevents hallucination of scripture references by checking the knowledge base.
"""

from __future__ import annotations

import re

from agents import function_tool

from rag.pipeline import search


def _parse_reference(reference: str) -> dict:
    """
    Parse a scripture reference into components.

    Examples:
        "John 3:16" → {"book": "John", "chapter": 3, "verse_start": 16}
        "Genesis 1:1-3" → {"book": "Genesis", "chapter": 1, "verse_start": 1, "verse_end": 3}
        "1 Corinthians 13:4" → {"book": "1 Corinthians", "chapter": 13, "verse_start": 4}

    Args:
        reference: The scripture reference string.

    Returns:
        Dict with parsed components.
    """
    reference = reference.strip()

    # Pattern: [optional number] Book Chapter:Verse[-Verse]
    pattern = re.compile(
        r'^(\d?\s*\w+(?:\s+\w+)?)\s+'  # Book name (with optional number prefix)
        r'(\d+)'                         # Chapter
        r':(\d+)'                        # Verse start
        r'(?:-(\d+))?'                   # Optional verse end
        r'(?:\s*\(.+\))?$'              # Optional translation in parens
    )

    match = pattern.match(reference)
    if not match:
        return {"raw": reference}

    result = {
        "book": match.group(1).strip(),
        "chapter": int(match.group(2)),
        "verse_start": int(match.group(3)),
    }
    if match.group(4):
        result["verse_end"] = int(match.group(4))

    return result


@function_tool
async def scripture_verify_tool(
    reference: str,
    claimed_text: str = "",
) -> str:
    """Verify that a Bible verse reference is accurate before citing it.

    ALWAYS call this tool before stating a specific verse reference like 'John 3:16'
    or quoting any scripture. This prevents hallucination of scripture references.

    Args:
        reference: The scripture reference to verify (e.g., "John 3:16", "Genesis 1:1-3").
        claimed_text: Optional - the text you plan to quote. If provided, it will be
                      compared against the actual text found.

    Returns:
        Verification result with actual text if found, or warning if not found.
    """
    parsed = _parse_reference(reference)

    # Build search query from the reference
    search_query = reference
    if "book" in parsed:
        search_query = f"{parsed['book']} chapter {parsed.get('chapter', '')} verse {parsed.get('verse_start', '')}"

    # Build filter conditions
    filter_conditions = {}
    if "book" in parsed:
        filter_conditions["book"] = parsed["book"]
    if "chapter" in parsed:
        filter_conditions["chapter"] = parsed["chapter"]

    # Search the knowledge base
    results = await search(
        query=search_query,
        top_k=3,
        filter_conditions=filter_conditions if filter_conditions else None,
    )

    if not results:
        # Try a broader search without filters
        results = await search(query=reference, top_k=3)

    if not results:
        return (
            f"VERIFICATION FAILED: Reference '{reference}' was NOT found in the knowledge base. "
            "DO NOT cite this verse. Instead, say: 'I want to be careful here — I cannot verify "
            "that specific reference in my knowledge base. Let me share the theological concept "
            "without a specific citation.' "
            "If the user provided this reference, respond: 'I wasn't able to verify that "
            "reference — it may not be in the Biblical canon. Could you double-check the "
            "reference?'"
        )

    # Found results — extract the most relevant text
    best_result = results[0]
    actual_text = best_result.text

    # Check if claimed text matches (if provided)
    text_match_warning = ""
    if claimed_text:
        claimed_lower = claimed_text.lower().strip()
        actual_lower = actual_text.lower().strip()
        if claimed_lower not in actual_lower and actual_lower not in claimed_lower:
            text_match_warning = (
                f"\nWARNING: The claimed text does not closely match the retrieved text. "
                f"Use the retrieved text for accuracy."
            )

    source_info = ""
    meta = best_result.metadata
    if meta.get("book"):
        source_info = f" (Source: {meta['book']}"
        if meta.get("chapter"):
            source_info += f" {meta['chapter']}"
            if meta.get("verse_start"):
                source_info += f":{meta['verse_start']}"
                if meta.get("verse_end"):
                    source_info += f"-{meta['verse_end']}"
        source_info += ")"

    return (
        f"VERIFIED: Reference '{reference}' found in knowledge base{source_info}.\n"
        f"Retrieved text: {actual_text[:500]}\n"
        f"You may cite this reference.{text_match_warning}"
    )
