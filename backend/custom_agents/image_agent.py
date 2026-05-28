"""
Image Agent — Sub-agent for handling image generation requests.
Wraps DALL-E 3 with theological safety guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import openai

from config.settings import get_settings
from prompts.image_prompt_builder import build_safe_image_prompt, ImagePromptResult


@dataclass
class ImageResult:
    """Result from image generation."""
    success: bool
    image_url: Optional[str] = None
    revised_prompt: Optional[str] = None
    blocked: bool = False
    error: Optional[str] = None


async def generate_christian_image(
    concept: str,
    style: str = "painterly",
    denomination_sensitivity: Optional[str] = None,
) -> ImageResult:
    """
    Generate a Christian-themed image with full safety pipeline.

    Steps:
    1. Build safe prompt via image_prompt_builder
    2. If blocked, return blocked result
    3. Call DALL-E 3
    4. Return image URL

    Args:
        concept: Description of the image to generate.
        style: Art style ("realistic", "painterly", "symbolic", "minimalist").
        denomination_sensitivity: Optional denomination context.

    Returns:
        ImageResult with URL or error information.
    """
    # Step 1: Safety check and prompt building
    prompt_result = build_safe_image_prompt(
        concept=concept,
        style=style,
        denomination_sensitivity=denomination_sensitivity,
    )

    if prompt_result.blocked:
        return ImageResult(
            success=False,
            blocked=True,
            error=prompt_result.reason,
        )

    # Step 2: Call DALL-E 3
    settings = get_settings()
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    try:
        response = await client.images.generate(
            model="dall-e-3",
            prompt=prompt_result.prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        return ImageResult(
            success=True,
            image_url=response.data[0].url,
            revised_prompt=response.data[0].revised_prompt,
        )

    except openai.BadRequestError as e:
        return ImageResult(
            success=False,
            error=f"Content policy violation: {str(e)}",
        )
    except openai.AuthenticationError:
        return ImageResult(
            success=False,
            error="Invalid OpenAI API key for image generation.",
        )
    except Exception as e:
        return ImageResult(
            success=False,
            error=f"Image generation failed: {str(e)}",
        )
