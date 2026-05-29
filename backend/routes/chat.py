"""
Chat Route — POST /api/chat
Main chat endpoint that orchestrates the full AI conversation pipeline.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from config.settings import get_settings
from custom_agents.christianity_agent import run_christianity_agent

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    message: str
    conversation_id: str
    denomination: Optional[str] = "Non-denominational"
    llm_config: Optional[dict] = None
    use_rag: bool = True
    kb_collection: str = "bible"
    vector_db: Optional[str] = None


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

    Processes a user message through the LangChain agent pipeline.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    settings = get_settings()
    model = settings.default_llm_model
    if request.llm_config:
        model = request.llm_config.get("model", model)

    logger.info(
        f"Processing chat request. Model: {model}, Use RAG: {request.use_rag}, "
        f"Vector DB: {request.vector_db or settings.default_vector_db}"
    )

    try:
        result = await run_christianity_agent(
            user_message=request.message,
            conversation_id=request.conversation_id,
            denomination=request.denomination or "Non-denominational",
            model=model,
            use_rag=request.use_rag,
            vector_db=request.vector_db,
        )

        logger.info(
            f"Chat processed. Guardrail: {result.guardrail_triggered}, RAG: {result.rag_used}"
        )

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
            detail=f"An error occurred while processing your request: {str(e)[:200]}",
        )
