"""
Qdrant Vector DB Client — Implements the VectorDBClient interface for Qdrant.
Supports named dense vectors (384d cosine) and sparse vectors.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector as QdrantSparseVector,
    VectorParams,
    SparseVectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from config.settings import get_settings
from rag.hybrid_search import SearchResult
from rag.sparse_encoder import SparseVector
from rag.stats import CollectionStats


@dataclass
class VectorPoint:
    """A point to upsert into the vector DB."""
    id: str
    dense_vector: list[float]
    sparse_vector: SparseVector | None = None
    text: str = ""
    metadata: dict = field(default_factory=dict)


class VectorDBClientBase(ABC):
    """Abstract base class for vector DB clients."""

    @abstractmethod
    async def upsert(self, collection: str, points: list[VectorPoint]) -> int:
        """Upsert points into the collection. Returns count of upserted points."""
        ...

    @abstractmethod
    async def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: SparseVector | None = None,
        top_k: int = 10,
        filter_conditions: dict | None = None,
    ) -> list[SearchResult]:
        """Perform hybrid search. Returns ranked results."""
        ...

    @abstractmethod
    async def get_stats(self, collection: str) -> CollectionStats:
        """Get collection statistics."""
        ...

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """List all collection names."""
        ...

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        ...


class QdrantVectorClient(VectorDBClientBase):
    """Qdrant Cloud/Local vector DB client."""

    def __init__(self):
        settings = get_settings()
        if settings.qdrant_api_key:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            self._client = QdrantClient(url=settings.qdrant_url)

    def _ensure_collection(self, collection: str) -> None:
        """Create the collection if it doesn't exist."""
        collections = [c.name for c in self._client.get_collections().collections]
        if collection not in collections:
            self._client.create_collection(
                collection_name=collection,
                vectors_config={
                    "dense": VectorParams(size=384, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )

    def _build_filter(self, filter_conditions: dict | None) -> Filter | None:
        """Build a Qdrant filter from a dict of conditions."""
        if not filter_conditions:
            return None

        must_conditions = []
        for key, value in filter_conditions.items():
            must_conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

        return Filter(must=must_conditions)

    async def upsert(self, collection: str, points: list[VectorPoint]) -> int:
        """Upsert points with dense and sparse vectors."""
        self._ensure_collection(collection)

        qdrant_points = []
        for point in points:
            vectors: dict[str, Any] = {
                "dense": point.dense_vector,
            }

            # Add sparse vector if available
            if point.sparse_vector and point.sparse_vector.indices:
                vectors["sparse"] = QdrantSparseVector(
                    indices=point.sparse_vector.indices,
                    values=point.sparse_vector.values,
                )

            payload = {
                "text": point.text,
                **point.metadata,
            }

            qdrant_points.append(
                PointStruct(
                    id=point.id,
                    vector=vectors,
                    payload=payload,
                )
            )

        # Batch upsert (max 100 points per batch)
        batch_size = 100
        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i:i + batch_size]
            self._client.upsert(
                collection_name=collection,
                points=batch,
            )

        return len(points)

    async def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: SparseVector | None = None,
        top_k: int = 10,
        filter_conditions: dict | None = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search using named dense and sparse vectors.
        Returns separate dense and sparse results for RRF fusion.
        """
        self._ensure_collection(collection)
        qdrant_filter = self._build_filter(filter_conditions)

        # Dense search using query_points (modern Qdrant API)
        dense_response = self._client.query_points(
            collection_name=collection,
            query=dense_vector,
            using="dense",
            limit=top_k * 2,  # Over-retrieve for better fusion
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results = []
        for hit in dense_response.points:
            payload = hit.payload or {}
            results.append(SearchResult(
                id=str(hit.id),
                text=payload.get("text", ""),
                score=hit.score,
                metadata={k: v for k, v in payload.items() if k != "text"},
            ))

        # If sparse vector available, do sparse search and merge
        if sparse_vector and sparse_vector.indices:
            try:
                sparse_response = self._client.query_points(
                    collection_name=collection,
                    query=QdrantSparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values,
                    ),
                    using="sparse",
                    limit=top_k * 2,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )

                sparse_search_results = []
                for hit in sparse_response.points:
                    payload = hit.payload or {}
                    sparse_search_results.append(SearchResult(
                        id=str(hit.id),
                        text=payload.get("text", ""),
                        score=hit.score,
                        metadata={k: v for k, v in payload.items() if k != "text"},
                    ))

                # Fuse results using RRF
                from rag.hybrid_search import hybrid_merge
                from config.settings import get_settings
                settings = get_settings()

                results = hybrid_merge(
                    dense_results=results,
                    sparse_results=sparse_search_results,
                    strategy=settings.hybrid_search_strategy,
                    top_k=top_k,
                    dense_weight=settings.dense_weight,
                    sparse_weight=settings.sparse_weight,
                )
            except Exception:
                # Fall back to dense-only results if sparse search fails
                results = results[:top_k]
        else:
            results = results[:top_k]

        return results

    async def get_stats(self, collection: str) -> CollectionStats:
        """Get collection statistics from Qdrant."""
        try:
            info = self._client.get_collection(collection)
            return CollectionStats(
                name=collection,
                total_vectors=info.points_count or 0,
                vector_dimension=384,
                has_sparse=True,
            )
        except Exception:
            return CollectionStats(name=collection)

    async def list_collections(self) -> list[str]:
        """List all Qdrant collections."""
        try:
            collections = self._client.get_collections()
            return [c.name for c in collections.collections]
        except Exception:
            return []

    async def delete_collection(self, name: str) -> None:
        """Delete a Qdrant collection."""
        self._client.delete_collection(name)

    async def health_check(self) -> bool:
        """Check Qdrant connectivity."""
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
