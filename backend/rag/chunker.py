"""
Chunker — Semantic and sliding window text chunking strategies.
Includes Bible-specific metadata extraction for chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ChunkMetadata:
    """Metadata attached to each chunk for retrieval and citation."""
    source_file: str
    chunk_index: int
    total_chunks: int
    section_heading: str | None = None
    char_start: int = 0
    char_end: int = 0
    book: str | None = None            # For Bible: Genesis, Exodus, etc.
    chapter: int | None = None         # For Bible: chapter number
    verse_start: int | None = None
    verse_end: int | None = None
    testament: Literal["Old", "New"] | None = None


@dataclass
class Chunk:
    """A chunk of text with its metadata."""
    text: str
    metadata: ChunkMetadata


# Token estimation: ~4 chars per token (rough approximation)
CHARS_PER_TOKEN = 4

# Bible books for metadata extraction
OLD_TESTAMENT_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Psalm",
    "Proverbs", "Ecclesiastes", "Song of Solomon", "Song of Songs",
    "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel",
    "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
    "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
    # Deuterocanonical / Catholic / Orthodox
    "Tobit", "Judith", "Wisdom", "Sirach", "Baruch",
    "1 Maccabees", "2 Maccabees",
    # LXX / Orthodox numbering
    "1 Kingdoms", "2 Kingdoms", "3 Kingdoms", "4 Kingdoms",
]

NEW_TESTAMENT_BOOKS = [
    "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians",
    "Galatians", "Ephesians", "Philippians", "Colossians",
    "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Hebrews", "James", "1 Peter", "2 Peter",
    "1 John", "2 John", "3 John", "Jude", "Revelation",
]

ALL_BIBLE_BOOKS = OLD_TESTAMENT_BOOKS + NEW_TESTAMENT_BOOKS


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _detect_bible_metadata(text: str, heading: str | None = None) -> dict:
    """
    Try to detect Bible book, chapter, and verse from text or heading.

    Returns:
        Dict with optional keys: book, chapter, verse_start, verse_end, testament.
    """
    metadata: dict = {}

    # Try to find book reference in heading or first line
    search_text = (heading or "") + " " + text[:200]

    for book in ALL_BIBLE_BOOKS:
        if book.lower() in search_text.lower():
            metadata["book"] = book
            if book in OLD_TESTAMENT_BOOKS:
                metadata["testament"] = "Old"
            else:
                metadata["testament"] = "New"
            break

    # Try to find chapter:verse pattern
    verse_pattern = re.compile(r'(\d+):(\d+)(?:-(\d+))?')
    match = verse_pattern.search(search_text)
    if match:
        metadata["chapter"] = int(match.group(1))
        metadata["verse_start"] = int(match.group(2))
        if match.group(3):
            metadata["verse_end"] = int(match.group(3))

    return metadata


def sliding_window_chunks(
    text: str,
    source_file: str,
    chunk_size_tokens: int = 512,
    overlap_tokens: int = 128,
    section_heading: str | None = None,
    char_offset: int = 0,
) -> list[Chunk]:
    """
    Split text into overlapping chunks using a sliding window.

    Args:
        text: The text to chunk.
        source_file: Source file path for metadata.
        chunk_size_tokens: Maximum chunk size in tokens.
        overlap_tokens: Overlap between consecutive chunks.
        section_heading: Optional section heading for this chunk group.
        char_offset: Character offset from the start of the full document.

    Returns:
        List of Chunk objects.
    """
    if not text.strip():
        return []

    chunk_size_chars = chunk_size_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    step = max(1, chunk_size_chars - overlap_chars)

    chunks = []
    pos = 0

    while pos < len(text):
        end = min(pos + chunk_size_chars, len(text))
        chunk_text = text[pos:end].strip()

        if chunk_text:
            bible_meta = _detect_bible_metadata(chunk_text, section_heading)

            metadata = ChunkMetadata(
                source_file=source_file,
                chunk_index=len(chunks),
                total_chunks=0,  # Will be set after all chunks are created
                section_heading=section_heading,
                char_start=char_offset + pos,
                char_end=char_offset + end,
                book=bible_meta.get("book"),
                chapter=bible_meta.get("chapter"),
                verse_start=bible_meta.get("verse_start"),
                verse_end=bible_meta.get("verse_end"),
                testament=bible_meta.get("testament"),
            )
            chunks.append(Chunk(text=chunk_text, metadata=metadata))

        pos += step

    # Set total_chunks
    for chunk in chunks:
        chunk.metadata.total_chunks = len(chunks)

    return chunks


def semantic_chunks(
    text: str,
    source_file: str,
    chunk_size_tokens: int = 512,
    overlap_tokens: int = 128,
) -> list[Chunk]:
    """
    Split text semantically by headings first, then apply sliding window
    within each section.

    Splits at markdown headings (## and ###), then applies sliding window
    chunking within each section.

    Args:
        text: The full document text.
        source_file: Source file path for metadata.
        chunk_size_tokens: Maximum chunk size in tokens.
        overlap_tokens: Overlap between chunks.

    Returns:
        List of Chunk objects with section heading metadata.
    """
    if not text.strip():
        return []

    # Split at heading boundaries
    heading_pattern = re.compile(r'^(#{2,3})\s+(.+)$', re.MULTILINE)
    sections: list[tuple[str | None, str, int]] = []  # (heading, content, char_start)

    matches = list(heading_pattern.finditer(text))

    if not matches:
        # No headings found — treat entire text as one section
        return sliding_window_chunks(
            text, source_file, chunk_size_tokens, overlap_tokens
        )

    # Text before first heading
    if matches[0].start() > 0:
        pre_text = text[:matches[0].start()]
        if pre_text.strip():
            sections.append((None, pre_text, 0))

    # Each section
    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end]
        if content.strip():
            sections.append((heading, content, start))

    # Apply sliding window within each section
    all_chunks: list[Chunk] = []
    for heading, content, char_start in sections:
        section_chunks = sliding_window_chunks(
            content, source_file, chunk_size_tokens, overlap_tokens,
            section_heading=heading, char_offset=char_start
        )
        all_chunks.extend(section_chunks)

    # Re-index all chunks
    for i, chunk in enumerate(all_chunks):
        chunk.metadata.chunk_index = i
        chunk.metadata.total_chunks = len(all_chunks)

    return all_chunks


def chunk_text(
    text: str,
    source_file: str,
    strategy: Literal["sliding_window", "semantic"] = "semantic",
    chunk_size_tokens: int = 512,
    overlap_tokens: int = 128,
) -> list[Chunk]:
    """
    Chunk text using the specified strategy.

    Args:
        text: The document text.
        source_file: Source file path.
        strategy: Chunking strategy to use.
        chunk_size_tokens: Maximum chunk size in tokens.
        overlap_tokens: Overlap between chunks.

    Returns:
        List of Chunk objects.
    """
    if strategy == "semantic":
        return semantic_chunks(text, source_file, chunk_size_tokens, overlap_tokens)
    else:
        return sliding_window_chunks(text, source_file, chunk_size_tokens, overlap_tokens)
