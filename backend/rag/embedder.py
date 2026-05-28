"""
Embedder — Singleton sentence-transformers model for dense embeddings.
Uses ONLY all-MiniLM-L6-v2 (384 dimensions). Non-negotiable.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# Singleton model instance
_model: SentenceTransformer | None = None

# The ONE embedding model used throughout the entire system
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def get_model() -> SentenceTransformer:
    """
    Get the singleton SentenceTransformer model.
    Loads the model on first call and reuses it thereafter.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string.

    Args:
        text: The text to embed.

    Returns:
        Normalized embedding vector of dimension 384.
    """
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Embed a batch of text strings efficiently.

    Args:
        texts: List of texts to embed.
        batch_size: Number of texts to process at once.

    Returns:
        List of normalized embedding vectors.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
    )
    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
    return EMBEDDING_DIMENSION
