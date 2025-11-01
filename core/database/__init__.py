"""
Async database module for FastAPI application.
This provides all the async database functionality migrated from Flask.
"""

from .engine import get_engine, init_db, close_db, get_async_session
from .models import Base, User, DataSource
from .repository import (
    BaseRepository, 
    UserRepository, 
    DataSourceRepository, 
    get_user_repository,
    get_data_source_repository
)
from .adapter import async_session_or_connection
from .async_connection_manager import (
    AsyncDatabaseManager,
    initialize_async_database_manager,
    get_async_database_manager,
    close_async_database_manager
)
from .async_registry import ensure_registry_tables
from .async_migrate import migrate_sqlite_to_postgres, migrate_postgres_to_sqlite
from .async_init_db import initialize

__all__ = [
    # Core database components
    "get_engine",
    "init_db",
    "close_db",
    "get_async_session",
    "Base",
    "AsyncRepository",
    "get_async_adapter",
    "async_session_or_connection",
    "AsyncCRUD",
    
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