"""
Async migration utilities for registry-related data.

This module centralizes migration entrypoints for async FastAPI application.
"""

from typing import Any, Dict, List
import logging

from core.config import Settings
from .data_sources import async_sqlite_queries, async_postgres_queries

logger = logging.getLogger(__name__)


async def migrate_sqlite_to_postgres(settings: Settings) -> Dict[str, int]:
    """Copy all rows from async SQLite data_sources into async PostgreSQL.

    Returns a summary dict: {'copied': n, 'skipped': m}
    """
    copied = 0
    skipped = 0

    try:
        raw_rows = await async_sqlite_queries.select_data_sources(settings, None)
        if isinstance(raw_rows, list):
            rows: List[Dict[str, Any]] = raw_rows
        elif raw_rows is None:
            rows = []
        else:
            rows = [raw_rows]

        logger.info(f"Found {len(rows)} rows to migrate from SQLite to PostgreSQL")
    except Exception:
        logger.exception("Failed to read from async SQLite for migration")
        raise

    for r in rows:
        try:
            await async_postgres_queries.insert_data_source(settings, r)
            copied += 1
            logger.debug(f"Migrated data source: {r.get('id', 'unknown')}")
        except Exception:
            skipped += 1
            logger.exception(f"Skipping row during migration: {r.get('id', 'unknown')}")

    logger.info(f"Migration completed: copied={copied}, skipped={skipped}")
    return {"copied": copied, "skipped": skipped}


async def migrate_postgres_to_sqlite(settings: Settings) -> Dict[str, int]:
    """Copy all rows from async PostgreSQL data_sources into async SQLite.

    Returns a summary dict: {'copied': n, 'skipped': m}
    """
    copied = 0
    skipped = 0

    try:
        raw_rows = await async_postgres_queries.select_data_sources(settings, None)
        if isinstance(raw_rows, list):
            rows: List[Dict[str, Any]] = raw_rows
        elif raw_rows is None:
            rows = []
        else:
            rows = [raw_rows]

        logger.info(f"Found {len(rows)} rows to migrate from PostgreSQL to SQLite")
    except Exception:
        logger.exception("Failed to read from async PostgreSQL for migration")
        raise

    for r in rows:
        try:
            await async_sqlite_queries.insert_data_source(settings, r)
            copied += 1
            logger.debug(f"Migrated data source: {r.get('id', 'unknown')}")
        except Exception:
            skipped += 1
            logger.exception(f"Skipping row during migration: {r.get('id', 'unknown')}")

    logger.info(f"Migration completed: copied={copied}, skipped={skipped}")
    return {"copied": copied, "skipped": skipped}
