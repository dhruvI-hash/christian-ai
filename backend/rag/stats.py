"""
RAG Stats Reporter — Generates formatted ingestion reports.
Both ASCII display and JSON serializable output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json


@dataclass
class IngestStats:
    """Statistics from a document ingestion operation."""
    source_file: str = ""
    markdown_output: str = ""
    total_chunks: int = 0
    vectors_upserted: int = 0
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db: str = ""
    collection: str = ""
    avg_chunk_size_tokens: float = 0.0
    largest_chunk_tokens: int = 0
    smallest_chunk_tokens: int = 0
    ingest_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "source_file": self.source_file,
            "markdown_output": self.markdown_output,
            "total_chunks": self.total_chunks,
            "vectors_upserted": self.vectors_upserted,
            "embedding_model": self.embedding_model,
            "vector_db": self.vector_db,
            "collection": self.collection,
            "avg_chunk_size_tokens": round(self.avg_chunk_size_tokens, 1),
            "largest_chunk_tokens": self.largest_chunk_tokens,
            "smallest_chunk_tokens": self.smallest_chunk_tokens,
            "ingest_duration_ms": round(self.ingest_duration_ms, 1),
        }

    def to_formatted_report(self) -> str:
        """Generate the formatted ASCII stat block."""
        lines = [
            "╔══════════════════════════════════════════╗",
            "║         RAG Ingestion Report             ║",
            "╠══════════════════════════════════════════╣",
            f"║ Source file     : {self.source_file:<23}║",
            f"║ Markdown output : {self.markdown_output:<23}║",
            f"║ Total chunks    : {self.total_chunks:<23}║",
            f"║ Vectors upserted: {self.vectors_upserted:<23}║",
            f"║ Embedding model : {self.embedding_model:<23}║",
            f"║ Vector DB       : {self.vector_db:<23}║",
            f"║ Collection      : {self.collection:<23}║",
            f"║ Avg chunk size  : {self.avg_chunk_size_tokens:<19.1f} tok ║",
            f"║ Largest chunk   : {self.largest_chunk_tokens:<19} tok ║",
            f"║ Smallest chunk  : {self.smallest_chunk_tokens:<19} tok ║",
            f"║ Ingest duration : {self.ingest_duration_ms:<19.1f} ms  ║",
            "╚══════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


@dataclass
class CollectionStats:
    """Statistics for a vector DB collection."""
    name: str = ""
    total_vectors: int = 0
    vector_dimension: int = 384
    has_sparse: bool = False
    source_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "total_vectors": self.total_vectors,
            "vector_dimension": self.vector_dimension,
            "has_sparse": self.has_sparse,
            "source_files": self.source_files,
        }


def print_stats(stats: IngestStats) -> None:
    """Print the formatted ingestion report to stdout."""
    print(stats.to_formatted_report())
