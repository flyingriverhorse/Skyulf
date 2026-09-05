"""FastAPI dependency providers for the data ingestion routes.

Supplies the per-request :class:`DataIngestionService` bound to the caller's
async session. The two permission dependencies below are deliberate no-ops and
are not referenced by any route in the repo, so nothing in this package is
currently authorized.
"""

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
