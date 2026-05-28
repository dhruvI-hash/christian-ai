"""
OpenAI LLM Provider — Direct OpenAI SDK calls (not Agents SDK).
Supports GPT-4o and GPT-4o-mini with prompt caching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openai

from config.settings import get_settings

if TYPE_CHECKING:
    from .router import LLMMessage, LLMResponse, RouterConfig


async def call_openai(
    messages: list["LLMMessage"],
    config: "RouterConfig",
) -> "LLMResponse":
    """
    Call the OpenAI Chat Completions API.

    Supports prompt caching via cache_control on the system message.

    Args:
        messages: List of conversation messages.
        config: Router configuration with model and parameters.

    Returns:
        LLMResponse with generated text and token usage.
    """
    from .router import LLMResponse

    settings = get_settings()
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    # Build message list with cache_control on system messages
    api_messages = []
    for msg in messages:
        message_dict: dict = {
            "role": msg.role,
            "content": msg.content,
        }
        # Enable prompt caching for system messages
        if msg.role == "system":
            message_dict["cache_control"] = {"type": "ephemeral"}
        api_messages.append(message_dict)

    try:
        response = await client.chat.completions.create(
            model=config.model,
            messages=api_messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Extract usage information
        usage = None
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            text=response.choices[0].message.content or "",
            provider="openai",
            model=config.model,
            usage=usage,
        )

    except openai.AuthenticationError:
        raise ValueError("Invalid OpenAI API key. Please check your OPENAI_API_KEY.")
    except openai.RateLimitError:
        raise ValueError("OpenAI rate limit exceeded. Please wait and try again.")
    except openai.APIError as e:
        raise ValueError(f"OpenAI API error: {e}")
