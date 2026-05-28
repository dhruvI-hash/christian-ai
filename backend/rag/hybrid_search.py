"""
Hybrid Search — Reciprocal Rank Fusion (RRF) and Alpha Blending
for merging dense and sparse search results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SearchResult:
    """A single search result with score and metadata."""
    id: str
    text: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


def reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k: int = 60,
    top_k: int = 10,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[SearchResult]:
    """
    Merge dense and sparse search results using Reciprocal Rank Fusion.

    RRF score = sum( weight / (k + rank + 1) ) for each result set.

    Args:
        dense_results: Results from dense vector search.
        sparse_results: Results from sparse BM25 search.
        k: RRF constant (default 60).
        top_k: Number of results to return.
        dense_weight: Weight for dense results.
        sparse_weight: Weight for sparse results.

    Returns:
        Merged and re-ranked list of SearchResult.
    """
    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    # Score dense results
    for i, r in enumerate(dense_results):
        rrf_score = dense_weight / (k + i + 1)
        scores[r.id] = scores.get(r.id, 0.0) + rrf_score
        result_map[r.id] = r

    # Score sparse results
    for i, r in enumerate(sparse_results):
        rrf_score = sparse_weight / (k + i + 1)
        scores[r.id] = scores.get(r.id, 0.0) + rrf_score
        # Merge metadata — prefer dense result metadata if exists
        if r.id not in result_map:
            result_map[r.id] = r

    # Build fused results
    fused_results = []
    for result_id, fused_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        result = result_map[result_id]
        fused_result = SearchResult(
            id=result.id,
            text=result.text,
            score=fused_score,
            metadata=result.metadata,
        )
        fused_results.append(fused_result)

    return fused_results[:top_k]


def alpha_blending(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    top_k: int = 10,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[SearchResult]:
    """
    Merge dense and sparse search results using alpha blending.

    Final score = dense_weight * normalized_dense_score + sparse_weight * normalized_sparse_score

    Args:
        dense_results: Results from dense vector search.
        sparse_results: Results from sparse BM25 search.
        top_k: Number of results to return.
        dense_weight: Weight for dense scores.
        sparse_weight: Weight for sparse scores.

    Returns:
        Merged and re-ranked list of SearchResult.
    """
    # Normalize dense scores
    dense_scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    if dense_results:
        max_dense = max(r.score for r in dense_results) or 1.0
        for r in dense_results:
            dense_scores[r.id] = r.score / max_dense
            result_map[r.id] = r

    # Normalize sparse scores
    sparse_scores: dict[str, float] = {}
    if sparse_results:
        max_sparse = max(r.score for r in sparse_results) or 1.0
        for r in sparse_results:
            sparse_scores[r.id] = r.score / max_sparse
            if r.id not in result_map:
                result_map[r.id] = r

    # Compute blended scores
    all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
    blended: list[SearchResult] = []

    for result_id in all_ids:
        d_score = dense_scores.get(result_id, 0.0)
        s_score = sparse_scores.get(result_id, 0.0)
        final_score = dense_weight * d_score + sparse_weight * s_score

        result = result_map[result_id]
        blended.append(SearchResult(
            id=result.id,
            text=result.text,
            score=final_score,
            metadata=result.metadata,
        ))

    # Sort by blended score descending
    blended.sort(key=lambda r: r.score, reverse=True)
    return blended[:top_k]


def hybrid_merge(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    strategy: Literal["rrf", "alpha"] = "rrf",
    top_k: int = 10,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[SearchResult]:
    """
    Merge search results using the configured strategy.

    Args:
        dense_results: Dense vector search results.
        sparse_results: Sparse BM25 search results.
        strategy: "rrf" for Reciprocal Rank Fusion or "alpha" for alpha blending.
        top_k: Number of results to return.
        dense_weight: Weight for dense results.
        sparse_weight: Weight for sparse results.

    Returns:
        Merged and re-ranked search results.
    """
    if strategy == "rrf":
        return reciprocal_rank_fusion(
            dense_results, sparse_results,
            top_k=top_k, dense_weight=dense_weight, sparse_weight=sparse_weight
        )
    elif strategy == "alpha":
        return alpha_blending(
            dense_results, sparse_results,
            top_k=top_k, dense_weight=dense_weight, sparse_weight=sparse_weight
        )
    else:
        raise ValueError(f"Unknown hybrid search strategy: {strategy}")
