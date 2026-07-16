"""
Async registry DB helpers: centralized DDL for the `data_sources` registry.

Provides `ensure_registry_tables(settings)` which ensures the `data_sources` table
exists in the configured registry backend (async SQLAlchemy engine or local sqlite).
This centralizes creation of registry-related tables so other modules don't
duplicate DDL logic.
"""

import logging
from pathlib import Path
from typing import Any

import aiosqlite
from sqlalchemy import Column, MetaData, String, Table, Text

from backend.config import Settings

from .engine import get_engine

logger = logging.getLogger(__name__)


def _build_data_sources_metadata() -> MetaData:
    """Build the SQLAlchemy MetaData/Table definition for the data_sources registry table."""
    metadata = MetaData()
    Table(
        "data_sources",
        metadata,
        Column("id", String, primary_key=True),
        Column("source_type", String),
        Column("name", String),
        Column("connection_info", Text),
        Column("metadata", Text),
        Column("category", String),
        Column("created_at", String),
    )
    return metadata


async def _create_via_engine_connection(async_engine: Any, metadata: MetaData) -> bool:
    """Run metadata.create_all against the async engine; return True on success."""
    if async_engine is None:
        return False
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        logger.info("Successfully created registry tables via async engine")
        return True
    except Exception:
        logger.exception("Failed to create registry tables via async engine")
        return False


async def _try_create_via_engine(async_engine: Any) -> bool:
    """Build the registry table metadata and attempt to create it via the async engine.

    Returns True only if the table was actually created.
    """
    try:
        metadata = _build_data_sources_metadata()
        return await _create_via_engine_connection(async_engine, metadata)
    except Exception:
        logger.exception("Failed to setup registry table metadata")
        return False


def _resolve_registry_db_path(settings: Settings) -> Path:
    """Resolve the SQLite file path for the registry DB, creating its parent directory if needed."""
    # Use DB_PATH from settings, fallback to default location
    db_filename = getattr(settings, "DB_PATH", "mlops_database.db")
    # For FastAPI, use current working directory as base when the path isn't absolute
    dbpath = Path(db_filename) if Path(db_filename).is_absolute() else Path.cwd() / db_filename

    d = dbpath.parent
    if d and not d.exists():
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.warning(f"Could not create directory {d}")

    return dbpath


async def _create_via_sqlite_fallback(settings: Settings) -> None:
    """Fallback: create the data_sources table directly via aiosqlite when no async engine is available."""
    try:
        dbpath = _resolve_registry_db_path(settings)

        async with aiosqlite.connect(dbpath) as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_sources (
                    id TEXT PRIMARY KEY,
                    source_type TEXT,
                    name TEXT,
                    connection_info TEXT,
                    metadata TEXT,
                    category TEXT,
                    created_at TEXT
                )
                """
            )
            await conn.commit()
            logger.info(f"Successfully created registry tables in SQLite at {dbpath}")

    except Exception:
        logger.exception("Failed to create registry tables in async SQLite fallback")


async def ensure_registry_tables(settings: Settings) -> None:
    """Ensure registry tables exist (data_sources) asynchronously.

    Uses the async engine from the database configuration.
    Fallback to async SQLite file if engine is not available.
    """
    try:
        async_engine = get_engine()

        executed = await _try_create_via_engine(async_engine)

        # Fallback to async SQLite file DDL
        if not executed:
            await _create_via_sqlite_fallback(settings)

    except Exception:
        logger.exception("Unexpected error in ensure_registry_tables")


async def ensure_user_sync_table(settings: Settings) -> None:
    """Compatibility wrapper; user_sync_limits feature removed and this is a no-op."""
    return None
