"""FastAPI dependencies for data ingestion."""

from typing import Annotated
from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.engine import get_async_session
from .service import DataIngestionService
from .exceptions import DataIngestionException


async def get_data_service(
    session: Annotated[AsyncSession, Depends(get_async_session)]
) -> DataIngestionService:
    """Get data ingestion service dependency."""
    return DataIngestionService(session)


async def require_data_access() -> None:
    """Require user permission for data access (Disabled)."""
    return None


async def require_data_admin() -> None:
    """Require admin permission for data administration (Disabled)."""
    return None


def handle_data_ingestion_exception(exc: DataIngestionException) -> HTTPException:
    """Convert data ingestion exceptions to HTTP exceptions."""
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "message": exc.message,
            "detail": exc.details,
            "type": exc.__class__.__name__
        }
    )
