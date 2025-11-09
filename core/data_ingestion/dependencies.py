"""FastAPI dependencies for data ingestion."""

from typing import Annotated
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth.dependencies import get_current_active_user
from core.database.engine import get_async_session
from core.database.models import User
from .service import DataIngestionService, get_data_ingestion_service
from .exceptions import DataIngestionException


async def get_data_service(
    session: Annotated[AsyncSession, Depends(get_async_session)]
) -> DataIngestionService:
    """Get data ingestion service dependency."""
    return get_data_ingestion_service(session)


async def require_data_access(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    """Require user permission for data access."""
    if not current_user.has_permission("user"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for data access"
        )
    return current_user


async def require_data_admin(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    """Require admin permission for data administration."""
    if not current_user.has_permission("admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required for data administration"
        )
    return current_user


def handle_data_ingestion_exception(exc: DataIngestionException) -> HTTPException:
    """Convert data ingestion exceptions to HTTP exceptions."""
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "message": exc.message,
            "detail": exc.detail,
            "type": exc.__class__.__name__
        }
    )
