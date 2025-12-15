"""
Async PostgreSQL-specific queries for data_sources table.
This is the async equivalent of the Flask db/data_sources/postgres_queries.py
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import column, literal_column, table
from sqlalchemy import text as sa_text

from backend.config import Settings

from ..adapter import async_session_or_connection

logger = logging.getLogger(__name__)

TABLE = "data_sources"


async def insert_data_source(settings: Settings, row: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a row into the PostgreSQL data_sources table. Returns the inserted row dict."""
    async with async_session_or_connection(settings) as session:
        try:
            tbl = table(TABLE, *[column(c) for c in row.keys()])
            stmt = tbl.insert().values(**row).returning(literal_column("*"))
            result = await session.execute(stmt)
            await session.commit()

            fetched = result.fetchone()
            if fetched:
                return dict(fetched._mapping)

            return {"success": True}

        except Exception as e:
            logger.error(f"Failed to insert data source: {e}")
            await session.rollback()
            raise


async def select_data_sources(
    settings: Settings,
    filter_dict: Optional[Dict[str, Any]] = None,
    one: bool = False,
) -> List[Dict[str, Any]] | Dict[str, Any] | None:
    """Select data sources with optional filtering."""
    async with async_session_or_connection(settings) as session:
        try:
            if filter_dict:
                conditions = []
                params = {}
                for k, v in filter_dict.items():
                    conditions.append(f"{k} = :{k}")
                    params[k] = v
                where_clause = " WHERE " + " AND ".join(conditions)
                sql = sa_text(f"SELECT * FROM {TABLE}{where_clause}")
                result = await session.execute(sql, params)
            else:
                sql = sa_text(f"SELECT * FROM {TABLE}")
                result = await session.execute(sql)

            rows = result.fetchall()
            data = [dict(row._mapping) for row in rows]

            if one:
                return data[0] if data else None
            return data

        except Exception as e:
            logger.error(f"Failed to select data sources: {e}")
            raise


async def update_data_source(
    settings: Settings, filter_dict: Dict[str, Any], update_data: Dict[str, Any]
):
    """Update data source records."""
    async with async_session_or_connection(settings) as session:
        try:
            set_parts = []
            params = {}
            for k, v in update_data.items():
                set_parts.append(f"{k} = :u_{k}")
                params[f"u_{k}"] = v

            where_parts = []
            for k, v in filter_dict.items():
                where_parts.append(f"{k} = :w_{k}")
                params[f"w_{k}"] = v

            set_clause = ", ".join(set_parts)
            where_clause = " WHERE " + " AND ".join(where_parts)

            sql = sa_text(f"UPDATE {TABLE} SET {set_clause}{where_clause}")
            result = await session.execute(sql, params)
            await session.commit()

            return {"affected_rows": result.rowcount}

        except Exception as e:
            logger.error(f"Failed to update data source: {e}")
            await session.rollback()
            raise


async def delete_data_source(settings: Settings, filter_dict: Dict[str, Any]):
    """Delete data source records."""
    async with async_session_or_connection(settings) as session:
        try:
            where_parts = []
            params = {}
            for k, v in filter_dict.items():
                where_parts.append(f"{k} = :{k}")
                params[k] = v

            where_clause = " WHERE " + " AND ".join(where_parts)
            sql = sa_text(f"DELETE FROM {TABLE}{where_clause}")

            result = await session.execute(sql, params)
            await session.commit()

            return {"affected_rows": result.rowcount}

        except Exception as e:
            logger.error(f"Failed to delete data source: {e}")
            await session.rollback()
            raise


async def select_data_source_by_file_hash(
    settings: Settings, file_hash: str
) -> Optional[Dict[str, Any]]:
    """Select a single data source by file hash using PostgreSQL JSON extraction."""
    if not file_hash:
        return None

    async with async_session_or_connection(settings) as session:
        try:
            sql = sa_text(
                f"SELECT * FROM {TABLE} WHERE config ->> 'file_hash' = :file_hash LIMIT 1"
            )
            result = await session.execute(sql, {"file_hash": file_hash})
            row = result.fetchone()
            return dict(row._mapping) if row else None
        except Exception as e:
            logger.error(f"Failed to select data source by file hash (PostgreSQL): {e}")
            raise
