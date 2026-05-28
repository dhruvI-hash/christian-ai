"""
RAG Tool — Agent tool for searching the Bible and Christian theology knowledge base.
Registered with the main Christianity agent via OpenAI Agents SDK.
"""

from __future__ import annotations

from contextvars import ContextVar
from agents import function_tool

from rag.pipeline import search, format_search_results_for_context

# Context variable to carry the selected vector DB into the tool call
_current_vector_db: ContextVar[str] = ContextVar("current_vector_db", default="qdrant")


def set_vector_db_context(vector_db: str) -> None:
    """Set the vector DB for the current async context (call before Runner.run)."""
    _current_vector_db.set(vector_db)


@function_tool
async def rag_tool(
    query: str,
    top_k: int = 5,
    testament: str = "",
    book: str = "",
) -> str:
    """Search the Bible and Christian theology knowledge base.

    CALL THIS TOOL when:
    - User asks about a specific Bible verse or passage
    - User asks about biblical events, characters, or teachings
    - User asks theological questions that require scriptural grounding
    - You need to verify a scripture reference before citing it

    DO NOT call for general greetings or purely creative tasks with no factual claims.

    Args:
        query: The search query text.
        top_k: Number of results to return (default: 5).
        testament: Optional filter: "Old" or "New" testament.
        book: Optional filter: specific Bible book name (e.g., "John", "Genesis").

    Returns:
        Formatted context with citations from the knowledge base.
    """
    # Build filter conditions from metadata
    filter_conditions = {}
    if testament:
        filter_conditions["testament"] = testament
    if book:
        filter_conditions["book"] = book

    # Read the vector DB set by the current request context
    vector_db = _current_vector_db.get()

    # Perform hybrid search
    results = await search(
        query=query,
        top_k=top_k,
        vector_db=vector_db,
        filter_conditions=filter_conditions if filter_conditions else None,
    )

    if not results:
        return (
            "No relevant passages found in the knowledge base for this query. "
            "You may need to answer from general theological knowledge, but be "
            "transparent about the lack of specific knowledge base grounding."
        )

    # Format results with citations
    formatted = format_search_results_for_context(results)

    return (
        f"Knowledge Base Results ({len(results)} passages found) [DB: {vector_db}]:\n\n"
        f"{formatted}\n\n"
        "IMPORTANT: Use these passages as your primary source. "
        "Cite the source references when using this content."
    )
