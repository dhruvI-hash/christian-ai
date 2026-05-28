"""
FastAPI Main Entry Point — ChristianAI Backend Server.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # Load environment variables first!

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import get_settings
from routes.chat import router as chat_router
from routes.ingest import router as ingest_router
from routes.rag_admin import router as rag_admin_router
from routes.image import router as image_router

# Disable agents SDK tracing to prevent getaddrinfo errors
from agents import set_tracing_disabled
set_tracing_disabled(True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events — startup and shutdown."""
    # Startup
    settings = get_settings()
    logger.info("ChristianAI Backend Starting...")
    logger.info(f"LLM Provider : {settings.default_llm_provider}")
    logger.info(f"LLM Model    : {settings.default_llm_model}")
    logger.info(f"Vector DB    : {settings.default_vector_db}")
    logger.info(f"Guardrail    : {'Enabled' if settings.enable_guardrail else 'Disabled'}")
    logger.info(f"Post-Safety  : {'Enabled' if settings.enable_post_safety else 'Disabled'}")

    yield

    # Shutdown
    logger.info("ChristianAI Backend shutting down...")


# Create the FastAPI application
app = FastAPI(
    title="ChristianAI",
    description=(
        "Christianity-focused AI Assistant API. "
        "Provides scripturally grounded, theologically accurate responses "
        "with RAG-powered knowledge retrieval and hallucination prevention."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allow frontend origin
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3003",
        "http://localhost:3004",
        "http://127.0.0.1:3004",
        "http://localhost:3005",
        "http://127.0.0.1:3005",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(rag_admin_router)
app.include_router(image_router)


@app.get("/")
async def root():
    """Root endpoint — API info."""
    return {
        "name": "ChristianAI",
        "version": "1.0.0",
        "description": "Christianity-focused AI Assistant",
        "endpoints": {
            "chat": "POST /api/chat",
            "ingest": "POST /api/ingest",
            "ingest_text": "POST /api/ingest/text",
            "image": "POST /api/image",
            "rag_collections": "GET /api/rag/collections",
            "rag_search": "POST /api/rag/collections/{name}/search",
            "rag_health": "GET /api/rag/health",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from rag.pipeline import check_health
    from llm.ollama_provider import check_ollama_health

    db_health = await check_health()
    ollama_ok = await check_ollama_health()

    return {
        "status": "ok",
        "services": {
            "qdrant": db_health.get("qdrant", False),
            "chromadb": db_health.get("chroma", False),
            "ollama": ollama_ok,
        },
    }
