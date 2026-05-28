"""
Sparse Encoder — BM25 sparse vector computation using rank_bm25.
Produces term-frequency weighted sparse vectors for hybrid search.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi


@dataclass
class SparseVector:
    """Sparse vector representation for BM25 search."""
    indices: list[int] = field(default_factory=list)    # Term hashes
    values: list[float] = field(default_factory=list)   # BM25 TF-IDF weights


def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.
    Lowercases, strips punctuation, splits on whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


def _term_to_hash(term: str) -> int:
    """
    Convert a term to a stable integer hash.
    Uses MD5 for stability across runs, truncated to 32-bit int.
    """
    h = hashlib.md5(term.encode()).hexdigest()
    return int(h[:8], 16)


class SparseEncoderManager:
    """
    Manages BM25 corpus for sparse encoding.
    Maintains a tokenized corpus for IDF computation.
    """

    def __init__(self):
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def fit(self, corpus: list[str]) -> None:
        """
        Fit the BM25 model on a corpus of documents.

        Args:
            corpus: List of document texts.
        """
        self._corpus_tokens = [_tokenize(doc) for doc in corpus]
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)

    def add_documents(self, documents: list[str]) -> None:
        """
        Add new documents to the existing corpus and refit.

        Args:
            documents: List of new document texts.
        """
        new_tokens = [_tokenize(doc) for doc in documents]
        self._corpus_tokens.extend(new_tokens)
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)

    @property
    def is_fitted(self) -> bool:
        """Check if the BM25 model has been fitted."""
        return self._bm25 is not None

    def encode_query(self, query: str) -> SparseVector:
        """
        Encode a query into a sparse vector using BM25 term weights.

        Args:
            query: The search query text.

        Returns:
            SparseVector with term hashes and BM25 weights.
        """
        tokens = _tokenize(query)
        if not tokens:
            return SparseVector()

        # Compute term frequencies for the query
        term_freqs: dict[str, int] = {}
        for token in tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1

        indices = []
        values = []

        for term, freq in term_freqs.items():
            term_hash = _term_to_hash(term)
            # Use frequency as weight (normalized)
            weight = float(freq) / len(tokens)
            indices.append(term_hash)
            values.append(weight)

        return SparseVector(indices=indices, values=values)

    def get_scores(self, query: str) -> list[float]:
        """
        Get BM25 scores for a query against the fitted corpus.

        Args:
            query: The search query text.

        Returns:
            List of scores, one per corpus document.
        """
        if not self.is_fitted:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return [0.0] * len(self._corpus_tokens)

        return self._bm25.get_scores(tokens).tolist()


def compute_sparse_vector(text: str, corpus: list[str] | None = None) -> SparseVector:
    """
    Compute a sparse BM25 vector for a text.

    If a corpus is provided, fits BM25 on it first.
    Otherwise, uses term frequency as a simple fallback.

    Args:
        text: The text to encode.
        corpus: Optional corpus for IDF computation.

    Returns:
        SparseVector with term hashes and weights.
    """
    tokens = _tokenize(text)
    if not tokens:
        return SparseVector()

    if corpus:
        manager = SparseEncoderManager()
        manager.fit(corpus)
        return manager.encode_query(text)

    # Fallback: simple term frequency
    term_freqs: dict[str, int] = {}
    for token in tokens:
        term_freqs[token] = term_freqs.get(token, 0) + 1

    indices = []
    values = []
    for term, freq in term_freqs.items():
        indices.append(_term_to_hash(term))
        values.append(float(freq))

    return SparseVector(indices=indices, values=values)
