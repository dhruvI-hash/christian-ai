"""
RAG Pipeline Orchestrator — Coordinates ingestion and search operations.
Entry point for all RAG functionality.
"""

from __future__ import annotations

from loguru import logger

from config.settings import get_settings
from rag.embedder import embed_text
from rag.hybrid_search import SearchResult
from rag.ingestor import ingest_file, ingest_text
from rag.sparse_encoder import compute_sparse_vector
from rag.stats import CollectionStats, IngestStats


def _get_vector_client():
    """Get the configured vector DB client."""
    settings = get_settings()
    if settings.default_vector_db == "qdrant":
        from vectordb.qdrant_client import QdrantVectorClient
        return QdrantVectorClient()
    else:
        from vectordb.chroma_client import ChromaVectorClient
        return ChromaVectorClient()


async def search(
    query: str,
    collection: str | None = None,
    vector_db: str | None = None,
    top_k: int = 5,
    filter_conditions: dict | None = None,
) -> list[SearchResult]:
    """
    Search the knowledge base using hybrid search.

    Args:
        query: The search query text.
        collection: Collection to search in.
        vector_db: Vector DB to search ("qdrant" or "chroma").
        top_k: Number of results to return.
        filter_conditions: Optional metadata filters (e.g., {"book": "John"}).

    Returns:
        List of SearchResult, ranked by hybrid score.
    """
    settings = get_settings()
    collection = collection or settings.default_collection
    vector_db = vector_db or settings.default_vector_db

    # Generate query embeddings
    dense_vector = embed_text(query)
    sparse_vector = compute_sparse_vector(query)

    # Get the appropriate client and perform hybrid search with Qdrant fallback
    if vector_db == "qdrant":
        try:
            from vectordb.qdrant_client import QdrantVectorClient
            client = QdrantVectorClient()
            results = await client.hybrid_search(
                collection=collection,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                filter_conditions=filter_conditions,
            )
        except Exception as e:
            logger.warning(f"Qdrant search failed (likely network/DNS issue): {e}. Falling back to local ChromaDB.")
            from vectordb.chroma_client import ChromaVectorClient
            client = ChromaVectorClient()
            results = await client.hybrid_search(
                collection=collection,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                filter_conditions=filter_conditions,
            )
    else:
        from vectordb.chroma_client import ChromaVectorClient
        client = ChromaVectorClient()
        results = await client.hybrid_search(
            collection=collection,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k,
            filter_conditions=filter_conditions,
        )

    return results


async def get_collection_stats(
    collection: str | None = None,
    vector_db: str | None = None,
) -> CollectionStats:
    """Get statistics for a collection."""
    settings = get_settings()
    collection = collection or settings.default_collection

    if (vector_db or settings.default_vector_db) == "qdrant":
        from vectordb.qdrant_client import QdrantVectorClient
        client = QdrantVectorClient()
    else:
        from vectordb.chroma_client import ChromaVectorClient
        client = ChromaVectorClient()

    return await client.get_stats(collection)


async def list_all_collections(
    vector_db: str | None = None,
) -> list[str]:
    """List all collections in the vector DB."""
    settings = get_settings()

    if (vector_db or settings.default_vector_db) == "qdrant":
        from vectordb.qdrant_client import QdrantVectorClient
        client = QdrantVectorClient()
    else:
        from vectordb.chroma_client import ChromaVectorClient
        client = ChromaVectorClient()

    return await client.list_collections()


async def delete_collection_by_name(
    name: str,
    vector_db: str | None = None,
) -> None:
    """Delete a collection from the vector DB."""
    settings = get_settings()

    if (vector_db or settings.default_vector_db) == "qdrant":
        from vectordb.qdrant_client import QdrantVectorClient
        client = QdrantVectorClient()
    else:
        from vectordb.chroma_client import ChromaVectorClient
        client = ChromaVectorClient()

    await client.delete_collection(name)


async def check_health(vector_db: str | None = None) -> dict:
    """Check health of vector DB connections."""
    results = {}

    try:
        from vectordb.qdrant_client import QdrantVectorClient
        qdrant = QdrantVectorClient()
        results["qdrant"] = await qdrant.health_check()
    except Exception:
        results["qdrant"] = False

    try:
        from vectordb.chroma_client import ChromaVectorClient
        chroma = ChromaVectorClient()
        results["chroma"] = await chroma.health_check()
    except Exception:
        results["chroma"] = False

    return results


def format_search_results_for_context(results: list[SearchResult]) -> str:
    """
    Format search results into a context string for the LLM.

    Args:
        results: List of search results.

    Returns:
        Formatted context string with citations.
    """
    if not results:
        return "No relevant passages found in the knowledge base."

    context_parts = []
    for i, result in enumerate(results, 1):
        meta = result.metadata
        source = meta.get("source_file", "Unknown")
        book = meta.get("book", "")
        chapter = meta.get("chapter", "")
        verse_start = meta.get("verse_start", "")
        verse_end = meta.get("verse_end", "")

        # Build citation string
        citation = source
        if book:
            citation = book
            if chapter:
                citation += f" {chapter}"
                if verse_start:
                    citation += f":{verse_start}"
                    if verse_end:
                        citation += f"-{verse_end}"

        heading = meta.get("section_heading", "")
        heading_str = f" ({heading})" if heading else ""

        context_parts.append(
            f"[Source {i}: {citation}{heading_str}]\n{result.text}\n"
        )

    return "\n---\n".join(context_parts)
