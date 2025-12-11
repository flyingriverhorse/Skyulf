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

from core.config import Settings
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


async def initialize(settings: Optional[Settings] = None):
    """
    Initialize the database backend based on configuration.
    """
    if settings is None:
        from core.config import Settings
        settings = Settings()

    db_type = settings.DB_TYPE.lower() if settings.DB_TYPE else "postgres"


# For standalone testing
async def main():
    """Standalone test function."""
    from core.config import Settings

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
