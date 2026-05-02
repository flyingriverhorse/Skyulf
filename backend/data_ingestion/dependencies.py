from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.engine import get_async_session
from .service import DataIngestionService


async def get_data_service(
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> DataIngestionService:
    """Get data ingestion service dependency."""
    return DataIngestionService(session)


async def require_data_access() -> None:
    """Require user permission for data access (Disabled)."""
    return None


async def require_data_admin() -> None:
    """Require admin permission for data administration (Disabled)."""
    return None
