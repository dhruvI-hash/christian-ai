"""
LangChain LLM client factory — builds a tool-calling chat model per provider.

- OpenAI / OpenRouter / Ollama  -> ChatOpenAI (OpenAI-compatible APIs)
- Gemini                        -> ChatGoogleGenerativeAI (native, reliable tool calling)
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from loguru import logger

from config.settings import Settings

Provider = Literal["openai", "gemini", "ollama", "openrouter"]

DEFAULT_TEMPERATURE = 0.3


def detect_provider(model: str, default: Provider = "openai") -> Provider:
    """Infer LLM provider from model name."""
    if model in ("gpt-4o", "gpt-4o-mini"):
        return "openai"
    if model.startswith("gemini") or model in (
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
    ):
        return "gemini"
    if model in {
        "granite4:latest",
        "granite3.1-moe:latest",
        "qwen3.5:4b",
        "qwen3:8b",
        "gemma4:e2b",
        "gemma3:4b",
        "llama3.2:latest",
        "lfm2.5-thinking:latest",
    }:
        return "ollama"
    if (
        model.startswith("google/")
        or model.startswith("openai/")
        or model.startswith("mistralai/")
        or model.startswith("deepseek/")
    ):
        return "openrouter"
    return default


def build_chat_model(
    model: str,
    settings: Settings,
    provider: Provider | None = None,
) -> BaseChatModel:
    """Create a LangChain chat model with tool-calling support for the given provider."""
    provider = provider or detect_provider(model, settings.default_llm_provider)
    logger.debug(f"Building chat model | provider={provider} | model={model}")

    if provider == "gemini":
        # Native Google client gives the most reliable tool/function calling for Gemini.
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.gemini_api_key,
            temperature=DEFAULT_TEMPERATURE,
        )

    if provider == "ollama":
        return ChatOpenAI(
            model=model,
            api_key="ollama",
            base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
            temperature=DEFAULT_TEMPERATURE,
        )

    if provider == "openrouter":
        return ChatOpenAI(
            model=model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=DEFAULT_TEMPERATURE,
        )

    # openai (default)
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=DEFAULT_TEMPERATURE,
    )
