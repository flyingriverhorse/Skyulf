"""Async database engine for FastAPI.

This module provides async database connectivity using SQLAlchemy 2.0+
with support for the same databases as the Flask version (SQLite, PostgreSQL).
"""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

try:
    # Avoid mutating sys.path here which is
    # brittle and hides import problems.
    from backend.config import get_settings
except ImportError as exc:
    raise ImportError(
        "Could not import 'core.config'. Ensure you're running the project as a package (python -m <package>) "
        "or that the project root is on PYTHONPATH. Original error: " + str(exc)
    ) from exc

logger = logging.getLogger(__name__)

# Global database engine and session factory
async_engine: AsyncEngine | None = None
async_session_factory: async_sessionmaker | None = None

# Sync database engine and session factory for compatibility
sync_engine = None
sync_session_factory = None

# Base class for SQLAlchemy models
Base = declarative_base()


async def init_db() -> None:
    """Initialize async database connections.

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
            # File databases need distinct connections for concurrent
            # transactions. Only memory databases must share one connection.
            **(
                {"poolclass": StaticPool}
                if make_url(settings.DATABASE_URL).database in (None, "", ":memory:")
                else {}
            ),
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
    if settings.DATABASE_URL.startswith("sqlite+aiosqlite://"):
        sync_url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
    else:
        sync_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")

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
    """Get async database session for dependency injection.

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
    """Get sync database session for dependency injection.

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

    # Run lightweight migrations for columns added after initial schema
    await _run_migrations()

    logger.info("✅ Database tables created/updated")


def _is_duplicate_column(original, dialect: str, column: str) -> bool:
    """Recognize only driver-confirmed duplicate-column migration failures."""
    return (dialect == "postgresql" and getattr(original, "sqlstate", None) == "42701") or (
        dialect == "sqlite"
        and getattr(original, "sqlite_errorcode", None) == 1
        and str(original) == f"duplicate column name: {column}"
    )


async def _run_migrations() -> None:
    """Apply incremental schema migrations for columns added after initial table creation.

    ``Base.metadata.create_all`` only creates *new* tables — it never ALTERs
    existing ones.  When a column is added to a model, append an entry to the
    ``_MIGRATIONS`` list below so that existing databases are patched on the
    next startup.

    Inspect columns before executing each ALTER so repeated startups are
    idempotent. Missing legacy tables are optional; inspection and DDL failures
    for the current schema propagate to prevent a falsely successful startup.

    HOW TO ADD A FUTURE MIGRATION:
        1. Add the column to the SQLAlchemy model as usual.
        2. Append one tuple per table to ``_MIGRATIONS``:
           ("0.X.Y", "ALTER TABLE <table> ADD COLUMN <col> <TYPE>")
        3. Restart — done.
    """
    if not async_engine:
        return

    from sqlalchemy import inspect, text
    from sqlalchemy.exc import DBAPIError

    _MIGRATIONS: list[tuple[str, str]] = [
        # v0.5.0 — Promote Winner
        ("0.5.0", "ALTER TABLE basic_training_jobs ADD COLUMN promoted_at TIMESTAMP"),
        ("0.5.0", "ALTER TABLE advanced_tuning_jobs ADD COLUMN promoted_at TIMESTAMP"),
        # v0.6.0 — Threshold Tuning Phase 2
        ("0.6.0", "ALTER TABLE training_jobs ADD COLUMN tuned_thresholds JSON"),
        (
            "0.6.0",
            "ALTER TABLE training_jobs ADD COLUMN tuned_thresholds_enabled BOOLEAN NOT NULL DEFAULT FALSE",
        ),
        # v0.7.6 — OPS-002 model-to-deployment lineage
        ("0.7.6", "ALTER TABLE deployments ADD COLUMN previous_deployment_id INTEGER"),
        # v0.7.6 — OPS-003 durable drift alert lifecycle
        (
            "0.7.6",
            "ALTER TABLE drift_check_results ADD COLUMN severity VARCHAR(20) NOT NULL DEFAULT 'none'",
        ),
        (
            "0.7.6",
            "ALTER TABLE drift_check_results ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'new'",
        ),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN owner VARCHAR(255)"),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN acknowledged_at TIMESTAMP"),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN resolved_at TIMESTAMP"),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN disposition_history JSON"),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN threshold_version INTEGER"),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN threshold_psi FLOAT"),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN threshold_ks FLOAT"),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN threshold_wasserstein FLOAT"),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN threshold_kl FLOAT"),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN deployment_id INTEGER"),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN model_version VARCHAR(50)"),
        (
            "0.7.6",
            "ALTER TABLE drift_check_results ADD COLUMN evaluation_status VARCHAR(20) "
            "NOT NULL DEFAULT 'completed'",
        ),
        ("0.7.6", "ALTER TABLE drift_check_results ADD COLUMN error_message TEXT"),
    ]

    def needs_column(sync_conn, table, column):
        """Skip existing columns and absent, retired job tables only."""
        inspector = inspect(sync_conn)
        if table in {"basic_training_jobs", "advanced_tuning_jobs"} and not inspector.has_table(
            table
        ):
            return False
        return column not in {item["name"] for item in inspector.get_columns(table)}

    applied = 0
    for version, ddl in _MIGRATIONS:
        # Entries use the documented ALTER TABLE <table> ADD COLUMN <column>
        # format; identifiers come exclusively from the static list above.
        tokens = ddl.split()
        table, column = tokens[2], tokens[5]

        try:
            async with async_engine.begin() as conn:
                if not await conn.run_sync(needs_column, table, column):
                    continue
                await conn.execute(text(ddl))
        except DBAPIError as exc:
            # A second startup worker may add the column after inspection.
            # Accept only a driver-confirmed duplicate, then verify the schema
            # in a new transaction after the failed ALTER has rolled back.
            duplicate = _is_duplicate_column(exc.orig, async_engine.dialect.name, column)
            if not duplicate:
                raise
            async with async_engine.begin() as conn:
                columns = await conn.run_sync(
                    lambda sync_conn, table_name: inspect(sync_conn).get_columns(table_name), table
                )
            if column not in {item["name"] for item in columns}:
                raise
            continue
        applied += 1
        logger.info("Migration [%s] applied: %s", version, ddl)

    if applied:
        logger.info("✅ %d migration(s) applied", applied)


async def health_check() -> bool:
    """Check database connectivity for health checks.

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
    except Exception as e:  # noqa: BLE001 - health probe returns False
        logger.error(f"Database health check failed: {e}")
        return False
