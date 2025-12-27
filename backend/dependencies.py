"""
 2025 Murat Unsal  Skyulf Project

Dependency Injection for FastAPI

This module provides common dependencies used across the application,
including database sessions and configuration.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings, get_settings
from backend.database.engine import get_async_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Database dependency that provides an async database session.
    Automatically handles session lifecycle.
    """
    async for session in get_async_session():
        yield session


def get_config() -> Settings:
    """Configuration dependency."""
    return get_settings()

