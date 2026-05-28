"""
Chat Route — POST /api/chat
Main chat endpoint that orchestrates the full AI conversation pipeline.
"""

from __future__ import annotations

from typing import Optional
from loguru import logger
import openai
from agents import set_default_openai_client, set_default_openai_api

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from custom_agents.christianity_agent import run_christianity_agent
from config.settings import get_settings

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    message: str
    conversation_id: str
    denomination: Optional[str] = "Non-denominational"
    llm_config: Optional[dict] = None
    use_rag: bool = True
    kb_collection: str = "bible"
    vector_db: Optional[str] = None  # "qdrant" or "chroma"; defaults to settings


class ChatResponse(BaseModel):
    """Response body from the chat endpoint."""
    response: str
    citations: list[dict]
    guardrail_triggered: bool
    rag_used: bool
    image_url: Optional[str] = None
    model_used: str
    conversation_id: str
    safety_warnings: list[str] = []


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    Processes a user message through the full pipeline:
    1. Guardrail check
    2. Agent processing with RAG and tools
    3. Post-processing safety check
    4. Response formatting

    Returns structured response with citations, safety metadata, and model info.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    settings = get_settings()

    # Determine model and provider
    provider = settings.default_llm_provider
    model = settings.default_llm_model
    if request.llm_config:
        model = request.llm_config.get("model", model)

    # Automatically detect provider from model name
    if model in ["gpt-4o", "gpt-4o-mini"]:
        provider = "openai"
    elif model in ["gemini-3-flash-preview", "gemini-3.1-flash-lite"]:
        provider = "gemini"
    elif model in {
        "granite4:latest", "granite3.1-moe:latest",
        "qwen3.5:4b", "qwen3:8b",
        "gemma4:e2b", "gemma3:4b",
        "llama3.2:latest", "lfm2.5-thinking:latest",
    }:
        provider = "ollama"
    elif model.startswith("google/") or model.startswith("openai/") or model.startswith("mistralai/") or model.startswith("deepseek/"):
        provider = "openrouter"

    logger.info(f"Processing chat request. Model: {model}, Provider: {provider}, Use RAG: {request.use_rag}")

    # Set up client dynamically
    try:
        if provider == "openai":
            client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            set_default_openai_client(client, use_for_tracing=True)
            set_default_openai_api("chat_completions")
            guardrail_model = "gpt-4o-mini"
        elif provider == "gemini":
            client = openai.AsyncOpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            set_default_openai_client(client, use_for_tracing=False)
            set_default_openai_api("chat_completions")
            guardrail_model = "gemini-3-flash-preview"
        elif provider == "ollama":
            client = openai.AsyncOpenAI(
                api_key="ollama",
                base_url=f"{settings.ollama_base_url}/v1"
            )
            set_default_openai_client(client, use_for_tracing=False)
            set_default_openai_api("chat_completions")
            guardrail_model = model  # Use the same running model for guardrail
        elif provider == "openrouter":
            client = openai.AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            set_default_openai_client(client, use_for_tracing=False)
            set_default_openai_api("chat_completions")
            guardrail_model = model  # Use the same model for guardrail via OpenRouter
        else:
            client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            set_default_openai_client(client, use_for_tracing=True)
            set_default_openai_api("chat_completions")
            guardrail_model = "gpt-4o-mini"
    except Exception as e:
        logger.exception("Failed to initialize dynamic LLM client:")
        raise HTTPException(
            status_code=500,
            detail=f"Configuration error: Failed to initialize LLM provider client: {str(e)}"
        )

    try:
        result = await run_christianity_agent(
            user_message=request.message,
            conversation_id=request.conversation_id,
            denomination=request.denomination or "Non-denominational",
            model=model,
            use_rag=request.use_rag,
            guardrail_model=guardrail_model,
            vector_db=request.vector_db,
        )

        logger.info(f"Chat request processed successfully. Guardrail triggered: {result.guardrail_triggered}, RAG used: {result.rag_used}")

        return ChatResponse(
            response=result.response,
            citations=result.citations,
            guardrail_triggered=result.guardrail_triggered,
            rag_used=result.rag_used,
            image_url=result.image_url,
            model_used=result.model_used,
            conversation_id=result.conversation_id,
            safety_warnings=result.safety_warnings,
        )

    except Exception as e:
        logger.exception("Unhandled error in chat endpoint:")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)[:200]}"
        )
