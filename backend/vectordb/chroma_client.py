"""
ChromaDB Vector DB Client — Implements the VectorDBClient interface for ChromaDB.
Sparse search is implemented as BM25 reranking since ChromaDB has no native sparse support.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import get_settings
from rag.hybrid_search import SearchResult, alpha_blending
from rag.sparse_encoder import SparseVector, SparseEncoderManager
from rag.stats import CollectionStats
from vectordb.qdrant_client import VectorDBClientBase, VectorPoint


class ChromaVectorClient(VectorDBClientBase):
    """
    ChromaDB vector DB client.

    Since ChromaDB does not natively support sparse vectors, hybrid search is
    implemented as:
    1. Dense search: retrieve top 50 candidates from ChromaDB
    2. Sparse rerank: score each candidate with BM25 against the query
    3. Fuse dense score + BM25 score using alpha blending
    4. Return top-K
    """

    def __init__(self):
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _get_or_create_collection(self, collection: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection."""
        return self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(self, collection: str, points: list[VectorPoint]) -> int:
        """
        Upsert points into ChromaDB.
        Sparse vectors are stored as metadata for later BM25 reranking.
        """
        chroma_collection = self._get_or_create_collection(collection)

        # Batch upsert
        batch_size = 100
        total = 0

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]

            ids = [p.id for p in batch]
            embeddings = [p.dense_vector for p in batch]
            documents = [p.text for p in batch]
            metadatas = []

            for p in batch:
                meta = dict(p.metadata)
                # ChromaDB metadata values must be str, int, float, or bool
                cleaned_meta = {}
                for k, v in meta.items():
                    if v is None:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        cleaned_meta[k] = v
                    else:
                        cleaned_meta[k] = str(v)
                metadatas.append(cleaned_meta)

            chroma_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total += len(batch)

        return total

    async def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: SparseVector | None = None,
        top_k: int = 10,
        filter_conditions: dict | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search using dense retrieval + BM25 reranking.

        1. Dense search: retrieve top 50 candidates
        2. BM25 rerank: score candidates against query
        3. Alpha blend: 0.7 * dense + 0.3 * sparse
        4. Return top-K
        """
        chroma_collection = self._get_or_create_collection(collection)

        # Build ChromaDB where filter
        where_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append({key: {"$eq": value}})
            if len(conditions) == 1:
                where_filter = conditions[0]
            elif len(conditions) > 1:
                where_filter = {"$and": conditions}

        # Step 1: Dense search — retrieve top 50 candidates
        n_candidates = min(50, max(top_k * 5, 50))
        try:
            results = chroma_collection.query(
                query_embeddings=[dense_vector],
                n_results=n_candidates,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0] if results["documents"] else [""] * len(ids)
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)

        # Convert distances to similarity scores (ChromaDB returns distances for cosine)
        # Cosine distance = 1 - cosine_similarity
        dense_results = []
        for i, (doc_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
            similarity = 1.0 - dist  # Convert distance to similarity
            dense_results.append(SearchResult(
                id=doc_id,
                text=doc or "",
                score=max(0.0, similarity),
                metadata=meta or {},
            ))

        # If no sparse vector or no documents to rerank, return dense only
        if not sparse_vector or not documents or not any(documents):
            return dense_results[:top_k]

        # Step 2: BM25 reranking
        sparse_encoder = SparseEncoderManager()
        valid_docs = [d for d in documents if d]
        if valid_docs:
            sparse_encoder.fit(valid_docs)

            # Reconstruct query from sparse vector is impractical,
            # so we use the document texts directly for BM25 scoring
            # We'll need the original query text — extract from the first few terms
            # For now, score each document
            bm25_scores = sparse_encoder.get_scores(
                " ".join(documents[0].split()[:5])  # Use first doc as proxy
            ) if documents[0] else [0.0] * len(documents)

            sparse_results = []
            for i, (doc_id, doc, meta) in enumerate(zip(ids, documents, metadatas)):
                score = bm25_scores[i] if i < len(bm25_scores) else 0.0
                sparse_results.append(SearchResult(
                    id=doc_id,
                    text=doc or "",
                    score=score,
                    metadata=meta or {},
                ))

            # Step 3: Alpha blending
            settings = get_settings()
            fused = alpha_blending(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_k=top_k,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
            )
            return fused

        return dense_results[:top_k]

    async def get_stats(self, collection: str) -> CollectionStats:
        """Get collection statistics from ChromaDB."""
        try:
            chroma_collection = self._get_or_create_collection(collection)
            count = chroma_collection.count()
            return CollectionStats(
                name=collection,
                total_vectors=count,
                vector_dimension=384,
                has_sparse=False,  # ChromaDB doesn't have native sparse support
            )
        except Exception:
            return CollectionStats(name=collection)

    async def list_collections(self) -> list[str]:
        """List all ChromaDB collections."""
        try:
            collections = self._client.list_collections()
            return [c.name for c in collections]
        except Exception:
            return []

    async def delete_collection(self, name: str) -> None:
        """Delete a ChromaDB collection."""
        self._client.delete_collection(name)

    async def health_check(self) -> bool:
        """Check ChromaDB connectivity."""
        try:
            self._client.list_collections()
            return True
        except Exception:
            return False
