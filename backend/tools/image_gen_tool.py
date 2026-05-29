"""
Image Generation Tool — DALL-E 3 with safety filtering (LangChain tool).
"""

from __future__ import annotations

import openai
from langchain_core.tools import tool
from loguru import logger

from config.settings import get_settings
from prompts.image_prompt_builder import build_safe_image_prompt


@tool
async def image_gen_tool(
    concept: str,
    style: str = "painterly",
    denomination_sensitivity: str = "",
) -> str:
    """Generate a Christian-themed image using DALL-E 3.

    Use for landscapes, symbolic imagery, abstract faith concepts, biblical scenes.
    Never use for depicting Jesus or God's face, religious violence, or divisive imagery.

    Args:
        concept: Description of the image to generate.
        style: Art style — realistic, painterly, symbolic, or minimalist.
        denomination_sensitivity: Optional denomination context for style.
    """
    logger.info(f"[image_gen] concept={concept!r} | style={style}")
    prompt_result = build_safe_image_prompt(
        concept=concept,
        style=style,
        denomination_sensitivity=denomination_sensitivity or None,
    )

    if prompt_result.blocked:
        logger.warning(f"[image_gen] blocked: {prompt_result.reason}")
        return (
            f"IMAGE BLOCKED: {prompt_result.reason}\n"
            "I'm not able to generate that particular image. I can create respectful "
            "Christian-themed imagery — landscapes, symbols, abstract faith concepts. "
            "What theme would you like to explore?"
        )

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
        logger.info("[image_gen] image generated successfully")

        return (
            f"IMAGE GENERATED SUCCESSFULLY\n"
            f"URL: {image_url}\n"
            f"Style: {style}\n"
            f"Revised prompt: {revised_prompt}"
        )

    except openai.BadRequestError as e:
        return (
            f"Image generation was rejected by the safety system: {str(e)}. "
            "Try a different concept or use symbolic/abstract imagery."
        )
    except openai.AuthenticationError:
        return "Image generation failed: Invalid OpenAI API key."
    except Exception as e:
        return f"Image generation failed: {str(e)}"
