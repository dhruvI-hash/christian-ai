"""
Markdown Converter — Converts PDF, DOCX, TXT to clean Markdown.
Uses pypdf/pdfplumber for PDFs, python-docx for DOCX, markdownify for HTML cleanup.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _clean_markdown(text: str) -> str:
    """
    Post-process extracted text into clean Markdown.
    Removes excessive whitespace, fixes formatting issues.
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Remove trailing whitespace on each line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def convert_pdf_to_markdown(file_path: str) -> str:
    """
    Convert a PDF file to Markdown text.

    Tries pdfplumber first for better formatting, falls back to pypdf.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text as clean Markdown.
    """
    text = ""

    # Try pdfplumber first (better at handling tables and formatting)
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"<!-- Page {i + 1} -->\n\n{page_text}")
            text = "\n\n---\n\n".join(pages)
    except Exception:
        # Fallback to pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"<!-- Page {i + 1} -->\n\n{page_text}")
            text = "\n\n---\n\n".join(pages)
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}")

    return _clean_markdown(text)


def convert_docx_to_markdown(file_path: str) -> str:
    """
    Convert a DOCX file to Markdown text.

    Extracts paragraphs from python-docx and maps heading styles to Markdown.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Extracted text as clean Markdown.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX parsing.")

    doc = Document(file_path)
    lines: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            lines.append("")
            continue

        # Map heading styles to Markdown
        style_name = (para.style.name or "").lower()
        if "heading 1" in style_name:
            lines.append(f"# {text}")
        elif "heading 2" in style_name:
            lines.append(f"## {text}")
        elif "heading 3" in style_name:
            lines.append(f"### {text}")
        elif "heading 4" in style_name:
            lines.append(f"#### {text}")
        elif "list" in style_name:
            lines.append(f"- {text}")
        else:
            lines.append(text)

        lines.append("")  # Blank line after each paragraph

    return _clean_markdown("\n".join(lines))


def convert_txt_to_markdown(file_path: str) -> str:
    """
    Read a plain text file and return as Markdown.

    Args:
        file_path: Path to the text file.

    Returns:
        Text content as clean Markdown.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return _clean_markdown(text)


def convert_md_to_markdown(file_path: str) -> str:
    """
    Read a Markdown file and clean it up.

    Args:
        file_path: Path to the Markdown file.

    Returns:
        Cleaned Markdown content.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return _clean_markdown(text)


def convert_to_markdown(file_path: str) -> str:
    """
    Convert any supported document to clean Markdown.

    Supports: .pdf, .docx, .txt, .md

    Args:
        file_path: Path to the source file.

    Returns:
        Clean Markdown text.

    Raises:
        ValueError: If the file format is not supported.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    converters = {
        ".pdf": convert_pdf_to_markdown,
        ".docx": convert_docx_to_markdown,
        ".txt": convert_txt_to_markdown,
        ".md": convert_md_to_markdown,
    }

    converter = converters.get(suffix)
    if converter is None:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported formats: {list(converters.keys())}"
        )

    return converter(file_path)


def save_markdown(markdown_text: str, source_path: str) -> str:
    """
    Save converted Markdown alongside the source file.

    Args:
        markdown_text: The Markdown content to save.
        source_path: Path to the original source file.

    Returns:
        Path to the saved Markdown file.
    """
    path = Path(source_path)
    md_path = path.with_suffix(".converted.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    return str(md_path)
