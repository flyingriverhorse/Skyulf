"""
© 2025 Murat Unsal — Skyulf Project

Dependency Injection for FastAPI

This module provides common dependencies used across the application,
including database sessions, authentication, and configuration.
"""

from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings, Settings
from core.database.engine import get_async_session
from core.auth.security import verify_access_token
from core.auth.models import User


# Security scheme for JWT tokens
security = HTTPBearer(auto_error=False)


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


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_config),
) -> User:
    """
    Extract and validate the current user from JWT token.
    
    Args:
        credentials: HTTP Bearer token credentials
        db: Database session
        settings: Application settings
    
    Returns:
        User: The authenticated user
    
    Raises:
        HTTPException: If token is invalid or user not found
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify and decode token
    payload = verify_access_token(credentials.credentials, settings.SECRET_KEY)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    from core.auth.service import get_user_by_id
    user = await get_user_by_id(db, payload.get("sub"))
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user (alias for backward compatibility).
    """
    return current_user


async def get_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency that requires admin privileges.
    
    Args:
        current_user: The authenticated user
    
    Returns:
        User: The admin user
    
    Raises:
        HTTPException: If user doesn't have admin privileges
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_config),
) -> Optional[User]:
    """
    Optional authentication dependency.
    Returns user if authenticated, None otherwise.
    Useful for endpoints that work both authenticated and unauthenticated.
    
    Args:
        credentials: HTTP Bearer token credentials (optional)
        db: Database session
        settings: Application settings
    
    Returns:
        Optional[User]: The authenticated user or None
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db, settings)
    except HTTPException:
        return None


class RateLimitDependency:
    """
    Rate limiting dependency.
    Can be used to limit requests per user or IP address.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
    
    async def __call__(
        self,
        request,
        user: Optional[User] = Depends(get_optional_user)
    ):
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