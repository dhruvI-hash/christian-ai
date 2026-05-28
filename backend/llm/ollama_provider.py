"""
Ollama LLM Provider — Local LLM via Ollama REST API.
Supports qwen3.5:4b and granite3.1:3b models.
Handles connection errors gracefully.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from config.settings import get_settings

if TYPE_CHECKING:
    from .router import LLMMessage, LLMResponse, RouterConfig


async def check_ollama_health() -> bool:
    """
    Check if the Ollama server is reachable.

    Returns:
        True if Ollama is running and responding, False otherwise.
    """
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return False


async def list_ollama_models() -> list[str]:
    """
    List all models available in the local Ollama instance.

    Returns:
        List of model names, or empty list if Ollama is unreachable.
    """
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return []


async def call_ollama(
    messages: list["LLMMessage"],
    config: "RouterConfig",
) -> "LLMResponse":
    """
    Call the Ollama local LLM API.

    Uses POST /api/chat with stream=False for synchronous response.

    Args:
        messages: List of conversation messages.
        config: Router configuration with model and parameters.

    Returns:
        LLMResponse with generated text.

    Raises:
        ValueError: If Ollama is unreachable or returns an error.
    """
    from .router import LLMResponse

    settings = get_settings()

    # Build message list for Ollama API
    api_messages = [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]

    payload = {
        "model": config.model,
        "messages": api_messages,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            )

            if response.status_code != 200:
                error_text = response.text
                raise ValueError(
                    f"Ollama returned status {response.status_code}: {error_text}"
                )

            data = response.json()

            # Extract response text
            text = data.get("message", {}).get("content", "")

            # Extract token usage if available
            usage = None
            if "eval_count" in data or "prompt_eval_count" in data:
                usage = {
                    "input_tokens": data.get("prompt_eval_count", 0),
                    "output_tokens": data.get("eval_count", 0),
                }

            return LLMResponse(
                text=text,
                provider="ollama",
                model=config.model,
                usage=usage,
            )

    except httpx.ConnectError:
        raise ValueError(
            f"Cannot connect to Ollama at {settings.ollama_base_url}. "
            "Please ensure Ollama is running (run 'ollama serve' in a terminal)."
        )
    except httpx.TimeoutException:
        raise ValueError(
            "Ollama request timed out. The model may be loading or the request "
            "is too complex. Please try again."
        )
    except httpx.HTTPError as e:
        raise ValueError(f"Ollama HTTP error: {e}")
