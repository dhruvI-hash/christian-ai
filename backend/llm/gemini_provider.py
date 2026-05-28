"""
Google Gemini LLM Provider — Uses google-generativeai SDK.
Supports Gemini 2.0 Flash and Gemini 2.0 Pro.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import google.generativeai as genai

from config.settings import get_settings

if TYPE_CHECKING:
    from .router import LLMMessage, LLMResponse, RouterConfig


def _map_role(role: str) -> str:
    """Map LLMMessage roles to Gemini Content roles."""
    # Gemini uses "model" instead of "assistant"
    if role == "assistant":
        return "model"
    return role


def _build_gemini_contents(messages: list["LLMMessage"]) -> tuple[str | None, list[dict]]:
    """
    Convert LLMMessage list to Gemini's Content format.

    Returns:
        Tuple of (system_instruction, contents_list).
        System messages are extracted as system_instruction.
    """
    system_instruction = None
    contents = []

    for msg in messages:
        if msg.role == "system":
            # Gemini handles system messages as system_instruction
            system_instruction = msg.content
        else:
            contents.append({
                "role": _map_role(msg.role),
                "parts": [{"text": msg.content}],
            })

    return system_instruction, contents


async def call_gemini(
    messages: list["LLMMessage"],
    config: "RouterConfig",
) -> "LLMResponse":
    """
    Call the Google Gemini API.

    Args:
        messages: List of conversation messages.
        config: Router configuration with model and parameters.

    Returns:
        LLMResponse with generated text and token usage.
    """
    from .router import LLMResponse

    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)

    system_instruction, contents = _build_gemini_contents(messages)

    # Build generation config
    generation_config = genai.GenerationConfig(
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
    )

    try:
        # Create model with system instruction
        model = genai.GenerativeModel(
            model_name=config.model,
            system_instruction=system_instruction,
            generation_config=generation_config,
        )

        # Generate response
        response = await model.generate_content_async(contents)

        # Extract token usage from usage_metadata
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
            }

        return LLMResponse(
            text=response.text or "",
            provider="gemini",
            model=config.model,
            usage=usage,
        )

    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg.upper() or "authentication" in error_msg.lower():
            raise ValueError("Invalid Gemini API key. Please check your GEMINI_API_KEY.")
        raise ValueError(f"Gemini API error: {error_msg}")
