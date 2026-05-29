"""
Scripture Verify Tool — Verifies Bible verse references before citation.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool
from loguru import logger

from rag.pipeline import search
from tools.rag_tool import _current_vector_db


def _parse_reference(reference: str) -> dict:
    """Parse a scripture reference into components."""
    reference = reference.strip()

    pattern = re.compile(
        r'^(\d?\s*\w+(?:\s+\w+)?)\s+'
        r'(\d+)'
        r':(\d+)'
        r'(?:-(\d+))?'
        r'(?:\s*\(.+\))?$'
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


@tool
async def scripture_verify_tool(
    reference: str,
    claimed_text: str = "",
) -> str:
    """Verify a Bible verse reference before citing it.

    Always call before stating a specific reference like John 3:16 or quoting scripture.

    Args:
        reference: Scripture reference (e.g. John 3:16, Genesis 1:1-3).
        claimed_text: Optional text to compare against the knowledge base.
    """
    parsed = _parse_reference(reference)
    vector_db = _current_vector_db.get()
    logger.info(f"[scripture_verify] reference={reference!r} | db={vector_db}")

    search_query = reference
    if "book" in parsed:
        search_query = (
            f"{parsed['book']} chapter {parsed.get('chapter', '')} "
            f"verse {parsed.get('verse_start', '')}"
        )

    filter_conditions = {}
    if "book" in parsed:
        filter_conditions["book"] = parsed["book"]
    if "chapter" in parsed:
        filter_conditions["chapter"] = parsed["chapter"]

    results = await search(
        query=search_query,
        top_k=3,
        vector_db=vector_db,
        filter_conditions=filter_conditions if filter_conditions else None,
    )

    if not results:
        results = await search(query=reference, top_k=3, vector_db=vector_db)

    if not results:
        logger.warning(f"[scripture_verify] FAILED — '{reference}' not found in KB")
        return (
            f"VERIFICATION FAILED: Reference '{reference}' was NOT found in the knowledge base. "
            "DO NOT cite this verse. Instead, say: 'I want to be careful here — I cannot verify "
            "that specific reference in my knowledge base. Let me share the theological concept "
            "without a specific citation.' "
            "If the user provided this reference, respond: 'I wasn't able to verify that "
            "reference — it may not be in the Biblical canon. Could you double-check the "
            "reference?'"
        )

    # Find the result whose metadata actually matches the requested book/chapter.
    best_result = None
    for candidate in results:
        meta = candidate.metadata or {}
        cand_book = str(meta.get("book", "")).strip().lower()
        req_book = str(parsed.get("book", "")).strip().lower()
        if not req_book or not cand_book:
            continue
        book_matches = req_book in cand_book or cand_book in req_book
        chapter_matches = (
            "chapter" not in parsed
            or str(meta.get("chapter", "")) == str(parsed["chapter"])
        )
        if book_matches and chapter_matches:
            best_result = candidate
            break

    if best_result is None:
        # We retrieved semantically related passages but none confirm the exact reference.
        logger.warning(
            f"[scripture_verify] UNCONFIRMED — '{reference}' not matched by metadata"
        )
        related = results[0]
        related_meta = related.metadata or {}
        related_src = related_meta.get("book", "an unlabeled passage")
        return (
            f"UNCONFIRMED: I could not confirm the exact reference '{reference}' in the "
            f"knowledge base. The closest related passage is from '{related_src}', which "
            "is NOT the same reference. Do NOT present '" + reference + "' as a verified "
            "quote. You may discuss the theological concept, but say you could not verify "
            "the specific reference."
        )

    actual_text = best_result.text
    logger.info(f"[scripture_verify] VERIFIED — '{reference}' matched in KB")

    text_match_warning = ""
    if claimed_text:
        claimed_lower = claimed_text.lower().strip()
        actual_lower = actual_text.lower().strip()
        if claimed_lower not in actual_lower and actual_lower not in claimed_lower:
            text_match_warning = (
                "\nWARNING: The claimed text does not closely match the retrieved text. "
                "Use the retrieved text for accuracy."
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
