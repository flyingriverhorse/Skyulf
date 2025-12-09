"""
 2025 Murat Unsal  Skyulf Project

Dependency Injection for FastAPI

This module provides common dependencies used across the application,
including database sessions and configuration.
"""

from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings, Settings
from core.database.engine import get_async_session


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


class RateLimitDependency:
    """
    Rate limiting dependency.
    Can be used to limit requests per user or IP address.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute

    async def __call__(self, request):
        # Implementation would depend on your rate limiting strategy
        # Could use Redis, in-memory cache, or database-based approach
        # For now, this is a placeholder
        pass


# Common rate limit instances
rate_limit_standard = RateLimitDependency(requests_per_minute=60)
rate_limit_strict = RateLimitDependency(requests_per_minute=10)


class PaginationParams:
    """
    Common pagination parameters.
    """

    def __init__(
        self,
        skip: int = 0,
        limit: int = 20,
        max_limit: int = 100
    ):
        if limit > max_limit:
            limit = max_limit
        if skip < 0:
            skip = 0
        if limit <= 0:
            limit = 20

        self.skip = skip
        self.limit = limit


def get_pagination_params(
    skip: int = 0,
    limit: int = 20
) -> PaginationParams:
    """
    Pagination dependency for list endpoints.

    Args:
        skip: Number of items to skip
        limit: Maximum number of items to return

    Returns:
        PaginationParams: Validated pagination parameters
    """
    return PaginationParams(skip=skip, limit=limit)

