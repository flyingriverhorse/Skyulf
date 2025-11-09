"""
Async Database Engine for FastAPI

This module provides async database connectivity using SQLAlchemy 2.0+
with support for the same databases as the Flask version (SQLite, PostgreSQL).
"""

import logging
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

try:
    # Avoid mutating sys.path here which is
    # brittle and hides import problems.
    from config import get_settings
except ImportError as exc:
    raise ImportError(
        "Could not import 'config'. Ensure you're running the project as a package (python -m <package>) "
        "or that the project root is on PYTHONPATH. Original error: "
        + str(exc)
    ) from exc

logger = logging.getLogger(__name__)

# Global database engine and session factory
async_engine: Optional[AsyncEngine] = None
async_session_factory: Optional[async_sessionmaker] = None

# Sync database engine and session factory for compatibility
sync_engine = None
sync_session_factory = None

# Base class for SQLAlchemy models
Base = declarative_base()


async def init_db() -> None:
    """
    Initialize async database connections.
    Sets up the global engine and session factory.
    """
    global async_engine, async_session_factory, sync_engine, sync_session_factory

    settings = get_settings()

    # Configure async engine based on database URL
    if settings.DATABASE_URL.startswith("sqlite"):
        # SQLite async configuration
        async_engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DB_ECHO,
            future=True,
            # SQLite specific settings for async
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                # Enable WAL mode for better concurrency
                "timeout": 30,
                "isolation_level": None,
            },
        )
    else:
        # PostgreSQL async configuration
        async_engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DB_ECHO,
            future=True,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections after 1 hour
        )

    # Create session factory
    async_session_factory = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )

    # Setup sync database for compatibility (convert async URL to sync)
    sync_url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite:///")
    if sync_url.startswith("postgresql+asyncpg://"):
        sync_url = sync_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")

    sync_engine = create_engine(sync_url, echo=settings.DB_ECHO)
    sync_session_factory = sessionmaker(bind=sync_engine)

    logger.info(f"✅ Database engine initialized: {settings.DATABASE_URL.split('://')[0]}")


async def close_db() -> None:
    """Close database connections and cleanup."""
    global async_engine, async_session_factory, sync_engine, sync_session_factory

    if async_engine:
        await async_engine.dispose()
        logger.info("✅ Async database connections closed")

    if sync_engine:
        sync_engine.dispose()
        logger.info("✅ Sync database connections closed")

    async_engine = None
    async_session_factory = None
    sync_engine = None
    sync_session_factory = None


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session for dependency injection.

    Yields:
        AsyncSession: Database session for async operations
    """
    if not async_session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_db():
    """
    Get sync database session for dependency injection.
    Compatible with non-async operations.

    Yields:
        Session: Database session for sync operations
    """
    if not sync_session_factory:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    session = sync_session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_engine() -> AsyncEngine:
    """Get the global async engine instance."""
    if not async_engine:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return async_engine


async def create_tables() -> None:
    """Create database tables from SQLAlchemy models."""
    if not async_engine:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("✅ Database tables created/updated")


async def health_check() -> bool:
    """
    Check database connectivity for health checks.

    Returns:
        bool: True if database is accessible, False otherwise
    """
    if not async_engine:
        return False

    try:
        from sqlalchemy import text
        async with async_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
