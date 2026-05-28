"""
Ingest Route — POST /api/ingest
Handles document ingestion into the RAG knowledge base.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from config.settings import get_settings
from rag.ingestor import ingest_file, ingest_text

router = APIRouter(prefix="/api")


class IngestURLRequest(BaseModel):
    """Request body for URL-based ingestion."""
    url: str
    collection_name: str = "bible"
    vector_db: Optional[str] = None


class IngestResponse(BaseModel):
    """Response body from ingestion."""
    source_file: str
    markdown_output: str
    total_chunks: int
    vectors_upserted: int
    embedding_model: str
    vector_db: str
    collection: str
    avg_chunk_size_tokens: float
    largest_chunk_tokens: int
    smallest_chunk_tokens: int
    ingest_duration_ms: float
    formatted_report: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    collection: str = Form("bible"),
    vector_db: Optional[str] = Form(None),
    chunk_strategy: str = Form("semantic"),
):
    """
    Ingest a document file (PDF, DOCX, TXT, MD) into the RAG knowledge base.

    Accepts multipart file upload with optional collection and vector DB configuration.
    """
    # Validate file extension
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    file_ext = os.path.splitext(file.filename or "")[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_ext}'. Allowed: {allowed_extensions}"
        )

    settings = get_settings()
    vector_db = vector_db or settings.default_vector_db

    # Save uploaded file to temp location
    try:
        # Create temp directory within the project
        upload_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        temp_path = os.path.join(upload_dir, file.filename or "upload" + file_ext)

        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Run ingestion pipeline
        stats = await ingest_file(
            file_path=temp_path,
            collection=collection,
            vector_db=vector_db,
            chunk_strategy=chunk_strategy,
        )

        return IngestResponse(
            source_file=stats.source_file,
            markdown_output=stats.markdown_output,
            total_chunks=stats.total_chunks,
            vectors_upserted=stats.vectors_upserted,
            embedding_model=stats.embedding_model,
            vector_db=stats.vector_db,
            collection=stats.collection,
            avg_chunk_size_tokens=stats.avg_chunk_size_tokens,
            largest_chunk_tokens=stats.largest_chunk_tokens,
            smallest_chunk_tokens=stats.smallest_chunk_tokens,
            ingest_duration_ms=stats.ingest_duration_ms,
            formatted_report=stats.to_formatted_report(),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)[:200]}"
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_raw_text(
    text: str = Form(...),
    source_name: str = Form("manual_input"),
    collection: str = Form("bible"),
    vector_db: Optional[str] = Form(None),
    chunk_strategy: str = Form("semantic"),
):
    """
    Ingest raw text directly into the RAG knowledge base.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")

    settings = get_settings()
    vector_db = vector_db or settings.default_vector_db

    try:
        stats = await ingest_text(
            text=text,
            source_name=source_name,
            collection=collection,
            vector_db=vector_db,
            chunk_strategy=chunk_strategy,
        )

        return IngestResponse(
            source_file=stats.source_file,
            markdown_output=stats.markdown_output,
            total_chunks=stats.total_chunks,
            vectors_upserted=stats.vectors_upserted,
            embedding_model=stats.embedding_model,
            vector_db=stats.vector_db,
            collection=stats.collection,
            avg_chunk_size_tokens=stats.avg_chunk_size_tokens,
            largest_chunk_tokens=stats.largest_chunk_tokens,
            smallest_chunk_tokens=stats.smallest_chunk_tokens,
            ingest_duration_ms=stats.ingest_duration_ms,
            formatted_report=stats.to_formatted_report(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)[:200]}"
        )
