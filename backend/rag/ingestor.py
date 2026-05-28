"""
Document Ingestor — Handles the full ingestion pipeline:
file → markdown → chunks → embeddings → sparse vectors → upsert → stats.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from config.settings import get_settings
from rag.chunker import Chunk, chunk_text
from rag.embedder import embed_batch
from rag.md_converter import convert_to_markdown, save_markdown
from rag.sparse_encoder import SparseVector, compute_sparse_vector
from rag.stats import IngestStats, print_stats
from vectordb.qdrant_client import VectorPoint

# Token estimation constant (same as chunker)
CHARS_PER_TOKEN = 4


def _get_vector_client():
    """Get the configured vector DB client."""
    settings = get_settings()
    if settings.default_vector_db == "qdrant":
        from vectordb.qdrant_client import QdrantVectorClient
        return QdrantVectorClient()
    else:
        from vectordb.chroma_client import ChromaVectorClient
        return ChromaVectorClient()


async def ingest_file(
    file_path: str,
    collection: str | None = None,
    vector_db: str | None = None,
    chunk_strategy: str = "semantic",
) -> IngestStats:
    """
    Ingest a document into the vector database.

    Full pipeline:
    1. Convert document to clean Markdown
    2. Save the Markdown file alongside the source
    3. Chunk the Markdown text
    4. Generate dense embeddings for all chunks
    5. Compute BM25 sparse vectors
    6. Upsert to the selected vector DB
    7. Return ingestion statistics

    Args:
        file_path: Path to the source document (PDF, DOCX, TXT, MD).
        collection: Vector DB collection name (defaults to settings).
        vector_db: Vector DB to use ("qdrant" or "chroma", defaults to settings).
        chunk_strategy: Chunking strategy ("semantic" or "sliding_window").

    Returns:
        IngestStats with ingestion metrics.
    """
    settings = get_settings()
    collection = collection or settings.default_collection
    vector_db = vector_db or settings.default_vector_db

    start_time = time.time()

    # Step 1: Convert to Markdown
    markdown_text = convert_to_markdown(file_path)

    # Step 2: Save Markdown alongside source
    md_output_path = save_markdown(markdown_text, file_path)

    # Step 3: Chunk
    chunks = chunk_text(
        text=markdown_text,
        source_file=os.path.basename(file_path),
        strategy=chunk_strategy,
        chunk_size_tokens=settings.chunk_size,
        overlap_tokens=settings.chunk_overlap,
    )

    if not chunks:
        return IngestStats(
            source_file=os.path.basename(file_path),
            markdown_output=md_output_path,
            total_chunks=0,
            vectors_upserted=0,
            vector_db=vector_db,
            collection=collection,
        )

    # Step 4: Generate dense embeddings
    chunk_texts = [c.text for c in chunks]
    embeddings = embed_batch(chunk_texts)

    # Step 5: Compute sparse vectors
    sparse_vectors = [compute_sparse_vector(text) for text in chunk_texts]

    # Step 6: Build vector points
    points = []
    for i, (chunk, embedding, sparse_vec) in enumerate(zip(chunks, embeddings, sparse_vectors)):
        point_id = str(uuid.uuid4())
        metadata = {
            "source_file": chunk.metadata.source_file,
            "chunk_index": chunk.metadata.chunk_index,
            "total_chunks": chunk.metadata.total_chunks,
            "section_heading": chunk.metadata.section_heading or "",
            "char_start": chunk.metadata.char_start,
            "char_end": chunk.metadata.char_end,
        }
        # Add Bible-specific metadata if available
        if chunk.metadata.book:
            metadata["book"] = chunk.metadata.book
        if chunk.metadata.chapter is not None:
            metadata["chapter"] = chunk.metadata.chapter
        if chunk.metadata.verse_start is not None:
            metadata["verse_start"] = chunk.metadata.verse_start
        if chunk.metadata.verse_end is not None:
            metadata["verse_end"] = chunk.metadata.verse_end
        if chunk.metadata.testament:
            metadata["testament"] = chunk.metadata.testament

        points.append(VectorPoint(
            id=point_id,
            dense_vector=embedding,
            sparse_vector=sparse_vec,
            text=chunk.text,
            metadata=metadata,
        ))

    # Step 7: Upsert to vector DB
    if vector_db == "qdrant":
        from vectordb.qdrant_client import QdrantVectorClient
        client = QdrantVectorClient()
    else:
        from vectordb.chroma_client import ChromaVectorClient
        client = ChromaVectorClient()

    upserted = await client.upsert(collection, points)

    # Calculate chunk size statistics
    chunk_sizes = [len(c.text) // CHARS_PER_TOKEN for c in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0
    min_size = min(chunk_sizes) if chunk_sizes else 0

    duration_ms = (time.time() - start_time) * 1000

    # Build stats
    stats = IngestStats(
        source_file=os.path.basename(file_path),
        markdown_output=md_output_path,
        total_chunks=len(chunks),
        vectors_upserted=upserted,
        vector_db=vector_db,
        collection=collection,
        avg_chunk_size_tokens=avg_size,
        largest_chunk_tokens=max_size,
        smallest_chunk_tokens=min_size,
        ingest_duration_ms=duration_ms,
    )

    # Print stats to stdout
    print_stats(stats)

    return stats


async def ingest_text(
    text: str,
    source_name: str,
    collection: str | None = None,
    vector_db: str | None = None,
    chunk_strategy: str = "semantic",
) -> IngestStats:
    """
    Ingest raw text directly (without a file).

    Args:
        text: The text content to ingest.
        source_name: Display name for the source.
        collection: Vector DB collection name.
        vector_db: Vector DB to use.
        chunk_strategy: Chunking strategy.

    Returns:
        IngestStats with ingestion metrics.
    """
    settings = get_settings()
    collection = collection or settings.default_collection
    vector_db = vector_db or settings.default_vector_db

    start_time = time.time()

    # Chunk directly
    chunks = chunk_text(
        text=text,
        source_file=source_name,
        strategy=chunk_strategy,
        chunk_size_tokens=settings.chunk_size,
        overlap_tokens=settings.chunk_overlap,
    )

    if not chunks:
        return IngestStats(
            source_file=source_name,
            total_chunks=0,
            vectors_upserted=0,
            vector_db=vector_db,
            collection=collection,
        )

    # Embed
    chunk_texts = [c.text for c in chunks]
    embeddings = embed_batch(chunk_texts)
    sparse_vectors = [compute_sparse_vector(t) for t in chunk_texts]

    # Build points
    points = []
    for chunk, embedding, sparse_vec in zip(chunks, embeddings, sparse_vectors):
        metadata = {
            "source_file": chunk.metadata.source_file,
            "chunk_index": chunk.metadata.chunk_index,
            "total_chunks": chunk.metadata.total_chunks,
            "section_heading": chunk.metadata.section_heading or "",
            "char_start": chunk.metadata.char_start,
            "char_end": chunk.metadata.char_end,
        }
        if chunk.metadata.book:
            metadata["book"] = chunk.metadata.book
        if chunk.metadata.chapter is not None:
            metadata["chapter"] = chunk.metadata.chapter
        if chunk.metadata.verse_start is not None:
            metadata["verse_start"] = chunk.metadata.verse_start
        if chunk.metadata.verse_end is not None:
            metadata["verse_end"] = chunk.metadata.verse_end
        if chunk.metadata.testament:
            metadata["testament"] = chunk.metadata.testament

        points.append(VectorPoint(
            id=str(uuid.uuid4()),
            dense_vector=embedding,
            sparse_vector=sparse_vec,
            text=chunk.text,
            metadata=metadata,
        ))

    # Upsert
    if vector_db == "qdrant":
        from vectordb.qdrant_client import QdrantVectorClient
        client = QdrantVectorClient()
    else:
        from vectordb.chroma_client import ChromaVectorClient
        client = ChromaVectorClient()

    upserted = await client.upsert(collection, points)

    chunk_sizes = [len(c.text) // CHARS_PER_TOKEN for c in chunks]
    duration_ms = (time.time() - start_time) * 1000

    stats = IngestStats(
        source_file=source_name,
        total_chunks=len(chunks),
        vectors_upserted=upserted,
        vector_db=vector_db,
        collection=collection,
        avg_chunk_size_tokens=sum(chunk_sizes) / len(chunk_sizes),
        largest_chunk_tokens=max(chunk_sizes),
        smallest_chunk_tokens=min(chunk_sizes),
        ingest_duration_ms=duration_ms,
    )

    print_stats(stats)
    return stats
