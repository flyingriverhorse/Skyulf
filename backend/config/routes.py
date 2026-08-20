"""
 2025 Murat Unsal  Skyulf Project

Configuration endpoints.

Exposes non-sensitive runtime settings to the frontend so client-side UI
can mirror server-side limits (upload size, allowed file types) instead
of hardcoding its own copy.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.config import Settings
from backend.dependencies import get_config

router = APIRouter(tags=["config"])


class UploadConfig(BaseModel):
    """Client-facing upload limits, sourced from server settings."""

    max_upload_size_bytes: int
    allowed_extensions: list[str]


@router.get("/config", response_model=UploadConfig)
def get_upload_config(settings: Settings = Depends(get_config)) -> UploadConfig:
    """Return the server-side upload limits for the frontend to mirror."""
    return UploadConfig(
        max_upload_size_bytes=settings.MAX_UPLOAD_SIZE,
        allowed_extensions=list(settings.ALLOWED_EXTENSIONS),
    )
