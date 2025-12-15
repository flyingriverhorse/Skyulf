"""
Async PostgreSQL connection module for FastAPI.
This is the async equivalent of the Flask db/connections/postgres_db_connection/connection.py
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from urllib.parse import quote_plus

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

# Load .env into environment when present (development convenience)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv is optional; environment variables may already be set in production
    pass

# Provider helpers can be added later for AWS/Azure/GCP
# try:
#     from .providers import aws, azure, gcp
# except ImportError:
#     aws = azure = gcp = None


def build_async_database_url_from_env() -> str:
    """
    Build an async SQLAlchemy database URL from environment variables.

    Priority:
      1. `DATABASE_URL` (raw full URL, modified for async)
      2. Constructed from DB_USER/DB_PASSWORD/DB_HOST/DB_PORT/DB_NAME

    Supports an optional `DB_PROVIDER` hint (e.g. 'aws', 'azure', 'gcp', or 'generic').
    SSL and extra options may be provided via `DB_SSLMODE` and `DB_EXTRA_PARAMS`.

    Returns a full async SQLAlchemy URL string. If no DB settings are found, falls back to
    an async SQLite file for development: `sqlite+aiosqlite:///mlops_postgres_fallback.db`.
    """

    # TODO: Add provider secret loading for AWS/Azure/GCP when needed

    raw = os.environ.get("DATABASE_URL")
    if raw:
        # Convert sync URL to async equivalent
        if raw.startswith("postgresql://") or raw.startswith("postgresql+psycopg2://"):
            return raw.replace("postgresql", "postgresql+asyncpg", 1)
        elif raw.startswith("sqlite://"):
            return raw.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return raw

    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    host = os.environ.get("DB_HOST")
    port = os.environ.get("DB_PORT")
    dbname = os.environ.get("DB_NAME")

    # If required components are missing, fallback to async SQLite for local dev
    if not any([user, password, host, dbname]):
        logger.warning(
            "No DB connection info found in environment; using async SQLite fallback."
        )
        return "sqlite+aiosqlite:///mlops_postgres_fallback.db"

    # Use asyncpg dialect for PostgreSQL async
    scheme = "postgresql+asyncpg"

    # Percent-encode user and password when present
    user_enc = quote_plus(user) if user else ""
    pwd_enc = quote_plus(password) if password else ""

    host_part = host or "localhost"
    port_part = f":{port}" if port else ""
    db_part = f"/{dbname}" if dbname else ""

    # Optional SSL mode and extra params
    sslmode = os.environ.get("DB_SSLMODE")  # e.g. require, verify-full
    extra = os.environ.get(
        "DB_EXTRA_PARAMS"
    )  # e.g. application_name=mlops&connect_timeout=10

    query_parts = []
    if sslmode:
        query_parts.append(f"ssl={sslmode}")  # asyncpg uses ssl instead of sslmode
    if extra:
        query_parts.append(extra)

    query = f"?{'&'.join(query_parts)}" if query_parts else ""

    # Construct URL
    auth = f"{user_enc}:{pwd_enc}@" if user_enc or pwd_enc else ""
    url = f"{scheme}://{auth}{host_part}{port_part}{db_part}{query}"

    return url


# Compose the final async DATABASE_URL and create async SQLAlchemy engine/session
DATABASE_URL = build_async_database_url_from_env()

# Create async engine
if DATABASE_URL.startswith("sqlite"):
    async_engine = create_async_engine(DATABASE_URL, future=True)
else:
    # For PostgreSQL with asyncpg
    async_engine = create_async_engine(DATABASE_URL, future=True, echo=False)

# Create async session maker
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

# Base for declarative models
Base = declarative_base()


async def init_db() -> None:
    """Create database tables asynchronously. Call this at application startup or during tests."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def startup_check_log():
    """Log non-sensitive DB settings on startup and warn about SSL certificate config."""
    provider = os.environ.get("DB_PROVIDER", "generic")
    host = os.environ.get("DB_HOST") or "sqlite_fallback"
    port = os.environ.get("DB_PORT") or ""
    user = os.environ.get("DB_USER") or ""
    dbname = os.environ.get("DB_NAME") or ""
    sslmode = os.environ.get("DB_SSLMODE")

    logger.info(
        "Async DB startup config: provider=%s host=%s port=%s user=%s db=%s",
        provider,
        host,
        port,
        user,
        dbname,
    )
    if sslmode == "verify-full":
        logger.info("SSL mode configured: %s", sslmode)


# Run startup checks
startup_check_log()


@asynccontextmanager
async def get_async_db():
    """Async context manager yielding a DB session.

    Use in application code as:
        async with get_async_db() as session:
            await session.execute(...)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to get async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Cleanup function for application shutdown
async def close_db():
    """Close the async database engine. Call this during application shutdown."""
    await async_engine.dispose()
