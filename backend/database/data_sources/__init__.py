"""
Async data sources module for FastAPI application.
Contains async CRUD operations and queries for the data_sources table.
"""

from .async_sqlite_queries import (
    insert_data_source as sqlite_insert_data_source,
    select_data_sources as sqlite_select_data_sources,
    select_data_source_by_file_hash as sqlite_select_data_source_by_file_hash,
    update_data_source as sqlite_update_data_source,
    delete_data_source as sqlite_delete_data_source,
    count_data_sources as sqlite_count_data_sources
)

from .async_postgres_queries import (
    insert_data_source as postgres_insert_data_source,
    select_data_sources as postgres_select_data_sources,
    select_data_source_by_file_hash as postgres_select_data_source_by_file_hash,
    update_data_source as postgres_update_data_source,
    delete_data_source as postgres_delete_data_source
)

from .async_data_sources_crud import (
    create,
    read,
    update,
    delete,
    get_by_file_hash,
    migrate_to_postgres,
    get_database_status
)

__all__ = [
    # SQLite operations
    "sqlite_insert_data_source",
    "sqlite_select_data_sources",
    "sqlite_select_data_source_by_file_hash",
    "sqlite_update_data_source",
    "sqlite_delete_data_source",
    "sqlite_count_data_sources",

    # PostgreSQL operations
    "postgres_insert_data_source",
    "postgres_select_data_sources",
    "postgres_select_data_source_by_file_hash",
    "postgres_update_data_source",
    "postgres_delete_data_source",

    # High-level CRUD operations
    "create",
    "read",
    "update",
    "delete",
    "get_by_file_hash",
    "migrate_to_postgres",
    "get_database_status",
]
