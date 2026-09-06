"""Async PostgreSQL-specific queries for data_sources table.

This is the async equivalent of the Flask db/data_sources/postgres_queries.py
"""

import logging
from typing import Any

from sqlalchemy import column, literal_column, select, table
from sqlalchemy import text as sa_text

from backend.config import Settings

from ..adapter import async_session_or_connection

# Engine-agnostic: both writers are implemented once in ``_common`` and re-exported
# here so ``_DB_PEERS`` dispatch and the ``postgres_*`` names in ``__init__`` resolve.
from ._common import delete_data_source, update_data_source

logger = logging.getLogger(__name__)

TABLE = "data_sources"

__all__ = [
    "delete_data_source",
    "insert_data_source",
    "select_data_source_by_file_hash",
    "select_data_sources",
    "update_data_source",
]


async def insert_data_source(settings: Settings, row: dict[str, Any]) -> dict[str, Any]:
    """Insert a row into the PostgreSQL data_sources table. Returns the inserted row dict."""
    async with async_session_or_connection(settings) as session:
        try:
            tbl = table(TABLE, *[column(c) for c in row])
            stmt: Any = tbl.insert().values(**row).returning(literal_column("*"))
            result = await session.execute(stmt)
            # Fetch BEFORE commit — asyncpg closes the server-side cursor on commit.
            fetched = result.fetchone()
            await session.commit()

            if fetched:
                return dict(fetched._mapping)

            return {"success": True}

        except Exception as e:
            logger.exception(f"Failed to insert data source: {e}")
            await session.rollback()
            raise


async def select_data_sources(
    settings: Settings,
    filter_dict: dict[str, Any] | None = None,
    one: bool = False,
) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Select data sources with optional filtering."""
    async with async_session_or_connection(settings) as session:
        try:
            tbl = table(TABLE)
            if filter_dict:
                # Build WHERE clause dynamically using SQLAlchemy Core
                conditions = []
                for k, v in filter_dict.items():
                    conditions.append(column(k) == v)

                stmt: Any = select(literal_column("*")).select_from(tbl).where(*conditions)
                result = await session.execute(stmt)
            else:
                stmt = select(literal_column("*")).select_from(tbl)
                result = await session.execute(stmt)

            rows = result.fetchall()
            data = [dict(row._mapping) for row in rows]

            if one:
                return data[0] if data else None
            return data

        except Exception as e:
            logger.exception(f"Failed to select data sources: {e}")
            raise


async def select_data_source_by_file_hash(
    settings: Settings, file_hash: str
) -> dict[str, Any] | None:
    """Select a single data source by file hash using PostgreSQL JSON extraction."""
    if not file_hash:
        return None

    async with async_session_or_connection(settings) as session:
        try:
            sql = sa_text(
                "SELECT * FROM data_sources WHERE config ->> 'file_hash' = :file_hash LIMIT 1"
            )
            result = await session.execute(sql, {"file_hash": file_hash})
            row = result.fetchone()
            return dict(row._mapping) if row else None
        except Exception as e:
            logger.exception(f"Failed to select data source by file hash (PostgreSQL): {e}")
            raise
