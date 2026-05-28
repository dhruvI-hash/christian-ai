"""
Image Generation Tool — DALL-E 3 image generation with safety filtering.
Always runs through build_safe_image_prompt() before generation.
"""

from __future__ import annotations

from typing import Literal

import openai
from agents import function_tool

from config.settings import get_settings
from prompts.image_prompt_builder import build_safe_image_prompt


@function_tool
async def image_gen_tool(
    concept: str,
    style: str = "painterly",
    denomination_sensitivity: str = "",
) -> str:
    """Generate a Christian-themed image using DALL-E 3.

    Use for: landscapes, symbolic imagery, abstract faith concepts, biblical scenes.
    NEVER use for: depicting Jesus or God's face, explicit religious violence,
    political religious content, or denominationally divisive imagery.

    Args:
        concept: Description of the image to generate.
        style: Art style - "realistic", "painterly", "symbolic", or "minimalist".
        denomination_sensitivity: Optional denomination context for style adaptation.

    Returns:
        Image URL if successful, or explanation of why the request was blocked.
    """
    # Step 1: Build safe prompt (includes content filtering)
    prompt_result = build_safe_image_prompt(
        concept=concept,
        style=style,
        denomination_sensitivity=denomination_sensitivity or None,
    )

    if prompt_result.blocked:
        return (
            f"IMAGE BLOCKED: {prompt_result.reason}\n"
            "I'm not able to generate that particular image. I can create respectful "
            "Christian-themed imagery — landscapes, symbols, abstract faith concepts. "
            "What theme would you like to explore?"
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

        image_url = response.data[0].url
        revised_prompt = response.data[0].revised_prompt

        return (
            f"IMAGE GENERATED SUCCESSFULLY\n"
            f"URL: {image_url}\n"
            f"Style: {style}\n"
            f"Revised prompt: {revised_prompt}"
        )

    except openai.BadRequestError as e:
        return (
            f"Image generation was rejected by the safety system: {str(e)}. "
            "This may be due to content policy restrictions. Try a different concept "
            "or use symbolic/abstract imagery."
        )
    except openai.AuthenticationError:
        return "Image generation failed: Invalid OpenAI API key."
    except Exception as e:
        return f"Image generation failed: {str(e)}"
