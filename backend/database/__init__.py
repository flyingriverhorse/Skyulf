"""
Async database module for FastAPI application.
This provides all the async database functionality migrated from Flask.
"""

from .adapter import async_session_or_connection
from .async_connection_manager import (
    AsyncDatabaseManager,
    close_async_database_manager,
    get_async_database_manager,
    initialize_async_database_manager,
)
from .async_init_db import initialize
from .async_migrate import migrate_postgres_to_sqlite, migrate_sqlite_to_postgres
from .async_registry import ensure_registry_tables
from .engine import close_db, get_async_session, get_engine, init_db
from .models import Base, DataSource, User
from .repository import (
    BaseRepository,
    DataSourceRepository,
    UserRepository,
    get_data_source_repository,
    get_user_repository,
)

__all__ = [
    # Core database components
    "get_engine",
    "init_db",
    "close_db",
    "get_async_session",
    "Base",
    "User",
    "DataSource",
    "async_session_or_connection",
    # Repository utilities
    "BaseRepository",
    "UserRepository",
    "DataSourceRepository",
    "get_user_repository",
    "get_data_source_repository",
    # Connection management
    "AsyncDatabaseManager",
    "initialize_async_database_manager",
    "get_async_database_manager",
    "close_async_database_manager",
    # Registry and migration utilities
    "ensure_registry_tables",
    "migrate_sqlite_to_postgres",
    "migrate_postgres_to_sqlite",
    "initialize",
]
