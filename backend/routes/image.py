"""
Image Route — POST /api/image
Image generation endpoint with safety filtering.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from custom_agents.image_agent import generate_christian_image

router = APIRouter(prefix="/api")


class ImageRequest(BaseModel):
    """Request body for image generation."""
    concept: str
    style: str = "painterly"
    denomination_sensitivity: Optional[str] = None


class ImageResponse(BaseModel):
    """Response from image generation."""
    success: bool
    image_url: Optional[str] = None
    revised_prompt: Optional[str] = None
    blocked: bool = False
    error: Optional[str] = None


@router.post("/image", response_model=ImageResponse)
async def generate_image(request: ImageRequest) -> ImageResponse:
    """
    Generate a Christian-themed image.

    Runs the concept through the safety filter before calling DALL-E 3.
    Returns the image URL or an explanation of why the request was blocked.
    """
    if not request.concept.strip():
        raise HTTPException(status_code=400, detail="Image concept cannot be empty.")

    try:
        result = await generate_christian_image(
            concept=request.concept,
            style=request.style,
            denomination_sensitivity=request.denomination_sensitivity,
        )

        return ImageResponse(
            success=result.success,
            image_url=result.image_url,
            revised_prompt=result.revised_prompt,
            blocked=result.blocked,
            error=result.error,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)[:200]}"
        )
