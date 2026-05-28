"""
LLM Router — selects and dispatches to the appropriate LLM provider
based on configuration. Supports OpenAI, Google Gemini, and Ollama.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


@dataclass
class LLMMessage:
    """A single message in the LLM conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    text: str
    provider: str
    model: str
    usage: Optional[dict] = None  # {"input_tokens": int, "output_tokens": int}


@dataclass
class RouterConfig:
    """Configuration for routing an LLM request."""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 1000


# Map of provider → available models
PROVIDER_MODELS: dict[LLMProvider, list[str]] = {
    LLMProvider.OPENAI: ["gpt-4o", "gpt-4o-mini"],
    LLMProvider.GEMINI: ["gemini-3-flash-preview", "gemini-3.1-flash-lite"],
    LLMProvider.OLLAMA: [
        "granite4:latest",
        "granite3.1-moe:latest",
        "qwen3.5:4b",
        "qwen3:8b",
        "gemma4:e2b",
        "gemma3:4b",
        "llama3.2:latest",
        "lfm2.5-thinking:latest",
    ],
    LLMProvider.OPENROUTER: [
        "google/gemini-3-flash-preview-001",
        "openai/gpt-4o-mini",
        "mistralai/mistral-small-3.2-24b-instruct",
        "deepseek/deepseek-chat-v3-0324",
    ],
}


def validate_config(config: RouterConfig) -> None:
    """Validate that the model is available for the selected provider."""
    available = PROVIDER_MODELS.get(config.provider, [])
    if config.model not in available:
        raise ValueError(
            f"Model '{config.model}' is not available for provider '{config.provider.value}'. "
            f"Available models: {available}"
        )


async def route(messages: list[LLMMessage], config: RouterConfig) -> LLMResponse:
    """
    Route a chat request to the appropriate LLM provider.

    Args:
        messages: List of conversation messages.
        config: Router configuration specifying provider, model, and parameters.

    Returns:
        LLMResponse with the generated text and metadata.

    Raises:
        ValueError: If the provider or model configuration is invalid.
    """
    validate_config(config)

    if config.provider == LLMProvider.OPENAI:
        from .openai_provider import call_openai
        return await call_openai(messages, config)
    elif config.provider == LLMProvider.GEMINI:
        from .gemini_provider import call_gemini
        return await call_gemini(messages, config)
    elif config.provider == LLMProvider.OLLAMA:
        from .ollama_provider import call_ollama
        return await call_ollama(messages, config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
