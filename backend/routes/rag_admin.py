"""
RAG Admin Routes — Collection management, search testing, and health checks.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import get_settings
from rag.pipeline import (
    check_health,
    delete_collection_by_name,
    get_collection_stats,
    list_all_collections,
    search,
    format_search_results_for_context,
)

router = APIRouter(prefix="/api/rag")


class SearchRequest(BaseModel):
    """Request body for search testing."""
    query: str
    top_k: int = 5
    filter: Optional[dict] = None
    vector_db: Optional[str] = None


class SearchResultItem(BaseModel):
    """A single search result."""
    id: str
    text: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    """Response from search testing."""
    results: list[SearchResultItem]
    formatted_context: str
    total: int


class CollectionStatsResponse(BaseModel):
    """Collection statistics response."""
    name: str
    total_vectors: int
    vector_dimension: int
    has_sparse: bool
    source_files: list[str]


class HealthResponse(BaseModel):
    """Health check response."""
    qdrant: bool
    chroma: bool


@router.get("/collections")
async def list_collections(vector_db: Optional[str] = None):
    """List all collections with their statistics."""
    try:
        collections = await list_all_collections(vector_db)
        result = []
        for name in collections:
            try:
                stats = await get_collection_stats(name, vector_db)
                result.append(stats.to_dict())
            except Exception:
                result.append({"name": name, "error": "Failed to get stats"})
        return {"collections": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{name}")
async def get_collection(name: str, vector_db: Optional[str] = None):
    """Get statistics for a specific collection."""
    try:
        stats = await get_collection_stats(name, vector_db)
        return stats.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{name}/search", response_model=SearchResponse)
async def search_collection(name: str, request: SearchRequest):
    """Test hybrid search against a collection."""
    try:
        results = await search(
            query=request.query,
            collection=name,
            vector_db=request.vector_db,
            top_k=request.top_k,
            filter_conditions=request.filter,
        )

        items = [
            SearchResultItem(
                id=r.id,
                text=r.text[:500],  # Truncate for API response
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]

        formatted = format_search_results_for_context(results)

        return SearchResponse(
            results=items,
            formatted_context=formatted,
            total=len(items),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{name}")
async def delete_collection(name: str, vector_db: Optional[str] = None):
    """Delete a collection."""
    try:
        await delete_collection_by_name(name, vector_db)
        return {"message": f"Collection '{name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def rag_health():
    """Check Qdrant and ChromaDB connectivity."""
    try:
        health = await check_health()
        return HealthResponse(
            qdrant=health.get("qdrant", False),
            chroma=health.get("chroma", False),
        )
    except Exception as e:
        return HealthResponse(qdrant=False, chroma=False)
