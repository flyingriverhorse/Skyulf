"""Async SQLite-specific queries for data_sources table.

This is the async equivalent of the Flask db/data_sources/sqlite_queries.py
"""

import logging
from typing import Any

from sqlalchemy import column, delete, func, literal_column, select, table, update
from sqlalchemy import text as sa_text

from backend.config import Settings

from ..adapter import async_session_or_connection

logger = logging.getLogger(__name__)

TABLE = "data_sources"


async def insert_data_source(settings: Settings, row: dict[str, Any]) -> dict[str, Any]:
    """Insert a row into the data_sources table. Returns the inserted row dict."""
    async with async_session_or_connection(settings) as session:
        try:
            # Use SQLAlchemy async session
            tbl = table(TABLE, *[column(c) for c in row])
            stmt = tbl.insert().values(**row)
            await session.execute(stmt)
            await session.commit()

            # Fetch the inserted row back
            if "id" in row:
                # Use SQLAlchemy Core for SELECT
                tbl_select = table(TABLE, column("id"))
                stmt_select: Any = (
                    select(literal_column("*"))
                    .select_from(tbl_select)
                    .where(column("id") == row["id"])
                )
                result = await session.execute(stmt_select)
                fetched = result.fetchone()
                if fetched:
                    return dict(fetched._mapping)

            # If no ID provided, get by last_insert_rowid for SQLite
            rid_result = await session.execute(sa_text("SELECT last_insert_rowid() AS rid"))
            rid = rid_result.scalar()
            if rid:
                # Use SQLAlchemy Core for SELECT by rowid
                tbl_select = table(TABLE, column("rowid"))
                stmt_select = (
                    select(literal_column("*"))
                    .select_from(tbl_select)
                    .where(column("rowid") == rid)
                )
                result = await session.execute(stmt_select)
                fetched = result.fetchone()
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
                # Build WHERE clause
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


async def update_data_source(
    settings: Settings, filter_dict: dict[str, Any], update_data: dict[str, Any]
):
    """Update the data source records matching every key in ``filter_dict``.

    An empty ``filter_dict`` is rejected rather than read as "match everything":
    the WHERE clause is built one key at a time below, so with no keys the
    statement compiles to an unscoped ``UPDATE data_sources SET ...`` that
    rewrites every row and still reports ``affected_rows`` as though that had
    been the requested operation. Checked before the session opens so no
    transaction is started for a call that cannot run.
    """
    if not filter_dict:
        raise ValueError("update_data_source requires a non-empty filter_dict")

    async with async_session_or_connection(settings) as session:
        try:
            # Use SQLAlchemy Core for UPDATE
            tbl = table(
                TABLE,
                *[column(c) for c in update_data] + [column(c) for c in filter_dict],
            )

            stmt = update(tbl).values(**update_data)

            for k, v in filter_dict.items():
                stmt = stmt.where(column(k) == v)

            result = await session.execute(stmt)
            await session.commit()

            return {"affected_rows": result.rowcount}

        except Exception as e:
            logger.exception(f"Failed to update data source: {e}")
            await session.rollback()
            raise


async def delete_data_source(settings: Settings, filter_dict: dict[str, Any]):
    """Delete the data source records matching every key in ``filter_dict``.

    An empty ``filter_dict`` is rejected rather than read as "match everything":
    the WHERE clause is built one key at a time below, so with no keys the
    statement compiles to a bare ``DELETE FROM data_sources`` that empties the
    table and still reports ``affected_rows`` as though that had been the
    requested operation. Checked before the session opens so no transaction is
    started for a call that cannot run.
    """
    if not filter_dict:
        raise ValueError("delete_data_source requires a non-empty filter_dict")

    async with async_session_or_connection(settings) as session:
        try:
            # Use SQLAlchemy Core for DELETE
            tbl = table(TABLE, *[column(c) for c in filter_dict])
            stmt = delete(tbl)

            for k, v in filter_dict.items():
                stmt = stmt.where(column(k) == v)

            result = await session.execute(stmt)
            await session.commit()

            return {"affected_rows": result.rowcount}

        except Exception as e:
            logger.exception(f"Failed to delete data source: {e}")
            await session.rollback()
            raise


async def count_data_sources(settings: Settings, filter_dict: dict[str, Any] | None = None) -> int:
    """Count data sources with optional filtering."""
    async with async_session_or_connection(settings) as session:
        try:
            tbl = table(TABLE)
            stmt = select(func.count()).select_from(tbl)

            if filter_dict:
                for k, v in filter_dict.items():
                    stmt = stmt.where(column(k) == v)

            result = await session.execute(stmt)
            return result.scalar() or 0

        except Exception as e:
            logger.exception(f"Failed to count data sources: {e}")
            raise


async def select_data_source_by_file_hash(
    settings: Settings, file_hash: str
) -> dict[str, Any] | None:
    """Select a single data source by the stored file hash (JSON field)."""
    if not file_hash:
        return None

    async with async_session_or_connection(settings) as session:
        try:
            # Use SQLAlchemy Core with json_extract
            tbl = table(TABLE)
            stmt: Any = (
                select(literal_column("*"))
                .select_from(tbl)
                .where(func.json_extract(column("config"), "$.file_hash") == file_hash)
                .limit(1)
            )

            result = await session.execute(stmt)
            row = result.fetchone()
            return dict(row._mapping) if row else None
        except Exception as e:
            logger.exception(f"Failed to select data source by file hash: {e}")
            raise
