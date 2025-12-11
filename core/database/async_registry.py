"""
Async registry DB helpers: centralized DDL for the `data_sources` registry.

Provides `ensure_registry_tables(settings)` which ensures the `data_sources` table
exists in the configured registry backend (async SQLAlchemy engine or local sqlite).
This centralizes creation of registry-related tables so other modules don't
duplicate DDL logic.
"""
import logging
import os
import aiosqlite
from sqlalchemy import MetaData, Table, Column, String, Text

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from core.config import Settings
from .engine import get_async_session

logger = logging.getLogger(__name__)


async def ensure_registry_tables(settings: Settings) -> None:
    """Ensure registry tables exist (data_sources) asynchronously.

    Uses the async engine from the database configuration.
    Fallback to async SQLite file if engine is not available.
    """
    try:
        async_engine = get_engine()

        executed = False
        try:
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

            if async_engine is not None:
                try:
                    async with async_engine.begin() as conn:
                        await conn.run_sync(metadata.create_all)
                    executed = True
                    logger.info("Successfully created registry tables via async engine")
                except Exception:
                    logger.exception("Failed to create registry tables via async engine")
                    executed = False
        except Exception:
            logger.exception("Failed to setup registry table metadata")
            executed = False

        # Fallback to async SQLite file DDL
        if not executed:
            try:
                # Use DB_PATH from settings, fallback to default location
                db_filename = getattr(settings, "DB_PATH", "mlops_database.db")
                if os.path.isabs(db_filename):
                    dbpath = db_filename
                else:
                    # For FastAPI, use current working directory as base
                    dbpath = os.path.join(os.getcwd(), db_filename)

                d = os.path.dirname(dbpath)
                if d and not os.path.exists(d):
                    try:
                        os.makedirs(d, exist_ok=True)
                    except Exception:
                        logger.warning(f"Could not create directory {d}")

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

    except Exception:
        logger.exception("Unexpected error in ensure_registry_tables")


async def ensure_user_sync_table(settings: Settings) -> None:
    """Compatibility wrapper; user_sync_limits feature removed and this is a no-op."""
    return None
