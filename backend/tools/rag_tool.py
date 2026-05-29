"""
RAG Tool — LangChain tool for searching the Bible and Christian theology knowledge base.
"""

from __future__ import annotations

from contextvars import ContextVar

from langchain_core.tools import tool
from loguru import logger

from rag.pipeline import search, format_search_results_for_context

_current_vector_db: ContextVar[str] = ContextVar("current_vector_db", default="qdrant")


def set_vector_db_context(vector_db: str) -> None:
    """Set the vector DB for the current async context (call before agent run)."""
    _current_vector_db.set(vector_db)


@tool
async def rag_tool(
    query: str,
    top_k: int = 5,
    testament: str = "",
    book: str = "",
) -> str:
    """Search the Bible and Christian theology knowledge base.

    Call when the user asks about scripture, biblical events, characters, teachings,
    or theological questions that need knowledge-base grounding.

    Args:
        query: The search query text.
        top_k: Number of results to return (default 5).
        testament: Optional filter — "Old" or "New".
        book: Optional filter — Bible book name (e.g. "John", "Genesis").
    """
    filter_conditions = {}
    if testament:
        filter_conditions["testament"] = testament
    if book:
        filter_conditions["book"] = book

    vector_db = _current_vector_db.get()

    logger.info(
        f"[rag_tool] query={query!r} | top_k={top_k} | db={vector_db} | "
        f"filters={filter_conditions or '{}'}"
    )

    results = await search(
        query=query,
        top_k=top_k,
        vector_db=vector_db,
        filter_conditions=filter_conditions if filter_conditions else None,
    )

    logger.info(f"[rag_tool] retrieved {len(results)} passage(s)")

    if not results:
        logger.warning(f"[rag_tool] no results for query={query!r}")
        return (
            "No relevant passages found in the knowledge base for this query. "
            "You may need to answer from general theological knowledge, but be "
            "transparent about the lack of specific knowledge base grounding."
        )

    formatted = format_search_results_for_context(results)

    return (
        f"Knowledge Base Results ({len(results)} passages found) [DB: {vector_db}]:\n\n"
        f"{formatted}\n\n"
        "IMPORTANT: Use these passages as your primary source. "
        "Cite the source references when using this content."
    )
