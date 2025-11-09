"""
Async database initializer utility for FastAPI.

This module provides a simple `initialize()` function that will attempt to initialize
one of the supported database backends based on the configuration.

Supported database types: 'postgres', 'mysql', 'mongodb', 'sqlite'. Default: 'postgres'.

Behavior:
- For PostgreSQL: calls async `init_db()` from the postgres connection package.
- For SQLite: uses async SQLite with aiosqlite.
- For MySQL/MongoDB: placeholder for future async implementations.

This file is intentionally conservative and will not modify data. It logs helpful
errors when drivers or configuration are missing.
"""

import logging
from typing import Optional

from config import Settings
from .async_registry import ensure_registry_tables

logger = logging.getLogger(__name__)


async def _try_init_postgres(settings: Settings):
    """Initialize PostgreSQL with async connection."""
    try:
        from .connections.postgres import init_db as pg_init

        logger.info("Initializing async PostgreSQL (calling init_db())")
        await pg_init()
        logger.info("Async PostgreSQL init_db() completed")
        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.exception("Async Postgres init module not available: %s", e)
        return False
    except Exception as e:
        logger.exception("Async Postgres init failed: %s", e)
        return False


async def _try_init_sqlite(settings: Settings):
    """Initialize SQLite with async connection."""
    try:
        import aiosqlite
        import os

        # Get SQLite database path from settings
        db_path = getattr(settings, "DB_PATH", "mlops_database.db")
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.getcwd(), db_path)

        # Create directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # Test connection
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("SELECT 1")
            await conn.commit()

        logger.info(f"Async SQLite initialized at: {db_path}")
        return True

    except ImportError as ie:
        logger.warning("aiosqlite missing: %s", ie)
        return False
    except Exception as e:
        logger.exception("Async SQLite initialization failed: %s", e)
        return False


async def _try_init_mysql(settings: Settings):
    """Placeholder for async MySQL initialization."""
    logger.warning("Async MySQL initialization not yet implemented")
    return False


async def _try_init_mongodb(settings: Settings):
    """Placeholder for async MongoDB initialization."""
    logger.warning("Async MongoDB initialization not yet implemented")
    return False


async def initialize(settings: Settings, preferred: Optional[str] = None) -> bool:
    """Initialize the selected database backend asynchronously.

    preferred: optional override ('postgres'|'mysql'|'mongodb'|'sqlite').
    If not provided, uses settings.DB_TYPE. Returns True on success, False otherwise.
    """
    db_type_value = preferred or getattr(settings, "DB_TYPE", "postgres")
    db_type = db_type_value.lower() if isinstance(db_type_value, str) else "postgres"
    logger.info("Async DB initializer selected type: %s", db_type)

    success = False

    if db_type == "postgres":
        success = await _try_init_postgres(settings)
    elif db_type == "sqlite":
        success = await _try_init_sqlite(settings)
    elif db_type == "mysql":
        success = await _try_init_mysql(settings)
    elif db_type in ("mongodb", "mongo"):
        success = await _try_init_mongodb(settings)
    else:
        logger.error("Unsupported database type: %s", db_type)
        return False

    # Ensure registry tables exist after successful initialization
    if success:
        try:
            await ensure_registry_tables(settings)
            logger.info("Registry tables ensured successfully")
        except Exception as e:
            logger.exception("Failed to ensure registry tables: %s", e)
            # Don't fail the entire initialization for registry table issues

    return success


# For standalone testing
async def main():
    """Standalone test function."""
    from config import Settings

    settings = Settings()
    success = await initialize(settings)
    if success:
        logger.info("Database initialization successful")
    else:
        logger.error("Database initialization failed")


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
