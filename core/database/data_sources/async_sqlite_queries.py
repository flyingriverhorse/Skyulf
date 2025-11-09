"""
Async SQLite-specific queries for data_sources table.
This is the async equivalent of the Flask db/data_sources/sqlite_queries.py
"""

from typing import Any, Dict, List, Optional
import logging
from sqlalchemy import text as sa_text

from ..adapter import async_session_or_connection
from config import Settings

logger = logging.getLogger(__name__)

TABLE = "data_sources"


async def insert_data_source(settings: Settings, row: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a row into the data_sources table. Returns the inserted row dict."""
    async with async_session_or_connection(settings) as session:
        try:
            # Use SQLAlchemy async session
            cols = ", ".join(row.keys())
            placeholders = ", ".join([f":{k}" for k in row.keys()])
            sql = sa_text(f"INSERT INTO {TABLE} ({cols}) VALUES ({placeholders})")
            await session.execute(sql, row)
            await session.commit()

            # Fetch the inserted row back
            if "id" in row:
                result = await session.execute(
                    sa_text(f"SELECT * FROM {TABLE} WHERE id = :id"),
                    {"id": row["id"]}
                )
                fetched = result.fetchone()
                if fetched:
                    return dict(fetched._mapping)

            # If no ID provided, get by last_insert_rowid for SQLite
            rid_result = await session.execute(sa_text("SELECT last_insert_rowid() AS rid"))
            rid = rid_result.scalar()
            if rid:
                result = await session.execute(
                    sa_text(f"SELECT * FROM {TABLE} WHERE rowid = :rid"),
                    {"rid": rid}
                )
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
                # Build WHERE clause
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
    settings: Settings,
    filter_dict: Dict[str, Any],
    update_data: Dict[str, Any]
):
    """Update data source records."""
    async with async_session_or_connection(settings) as session:
        try:
            # Build SET clause
            set_parts = []
            params = {}
            for k, v in update_data.items():
                set_parts.append(f"{k} = :u_{k}")
                params[f"u_{k}"] = v

            # Build WHERE clause
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
            # Build WHERE clause
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


async def count_data_sources(
    settings: Settings,
    filter_dict: Optional[Dict[str, Any]] = None
) -> int:
    """Count data sources with optional filtering."""
    async with async_session_or_connection(settings) as session:
        try:
            if filter_dict:
                conditions = []
                params = {}
                for k, v in filter_dict.items():
                    conditions.append(f"{k} = :{k}")
                    params[k] = v
                where_clause = " WHERE " + " AND ".join(conditions)
                sql = sa_text(f"SELECT COUNT(*) FROM {TABLE}{where_clause}")
                result = await session.execute(sql, params)
            else:
                sql = sa_text(f"SELECT COUNT(*) FROM {TABLE}")
                result = await session.execute(sql)

            return result.scalar() or 0

        except Exception as e:
            logger.error(f"Failed to count data sources: {e}")
            raise


async def select_data_source_by_file_hash(
    settings: Settings,
    file_hash: str
) -> Optional[Dict[str, Any]]:
    """Select a single data source by the stored file hash (JSON field)."""
    if not file_hash:
        return None

    async with async_session_or_connection(settings) as session:
        try:
            sql = sa_text(
                f"SELECT * FROM {TABLE} WHERE json_extract(config, '$.file_hash') = :file_hash LIMIT 1"
            )
            result = await session.execute(sql, {"file_hash": file_hash})
            row = result.fetchone()
            return dict(row._mapping) if row else None
        except Exception as e:
            logger.error(f"Failed to select data source by file hash: {e}")
            raise
