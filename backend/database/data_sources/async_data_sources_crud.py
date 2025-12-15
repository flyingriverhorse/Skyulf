"""
Async orchestration layer for data_sources operations.

This module centralizes the logic to write/read/update/delete the
`data_sources` registry across PostgreSQL and SQLite. It intentionally
keeps behavior separate from the generic database CRUD dispatcher.
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast

from backend.config import Settings

from . import async_postgres_queries as pg_q
from . import async_sqlite_queries as sqlite_q

logger = logging.getLogger(__name__)


def get_primary_database(settings: Settings) -> str:
    """Get the primary database type from configuration.

    Returns 'sqlite' or 'postgres' based on DB_PRIMARY setting.
    Defaults to 'sqlite' if not specified.
    """
    return getattr(settings, "DB_PRIMARY", "sqlite").lower()


async def create(settings: Settings, row: Dict[str, Any]) -> Any:
    """Create a new data source record with configurable primary database.

    Strategy: DB_PRIMARY-first approach
    1) Insert into primary database (SQLite or PostgreSQL)
    2) Optionally sync to secondary database if available (best-effort, non-blocking)
    """

    primary_db = get_primary_database(settings)

    if primary_db == "sqlite":
        # SQLite as primary
        try:
            sqlite_res = await sqlite_q.insert_data_source(settings, row)
            logger.info(
                f"Successfully inserted data source into SQLite (primary): {row.get('id', 'unknown')}"
            )
        except Exception as e:
            # Handle duplicate key errors
            msg = str(e).lower()
            if "unique" in msg or "duplicate" in msg or "constraint" in msg:
                try:
                    existing = await sqlite_q.select_data_sources(
                        settings, {"id": row.get("id")}
                    )
                    if isinstance(existing, list) and existing:
                        logger.info(
                            f"Data source already exists in SQLite: {row.get('id')}"
                        )
                        return existing[0]
                    if isinstance(existing, dict):
                        logger.info(
                            f"Data source already exists in SQLite: {row.get('id')}"
                        )
                        return existing
                except Exception:
                    pass

            logger.exception("SQLite insert failed for data_sources")
            raise RuntimeError(
                "Failed to persist data_sources row to primary database (SQLite)"
            )

        # Secondary sync to PostgreSQL (best-effort)
        try:
            await pg_q.insert_data_source(settings, row)
            logger.info(
                f"Successfully synced data source to PostgreSQL (secondary): {row.get('id', 'unknown')}"
            )
        except Exception:
            logger.warning(
                f"Failed to sync data source to PostgreSQL (non-critical): {row.get('id', 'unknown')}"
            )

        return sqlite_res

    elif primary_db == "postgres":
        # PostgreSQL as primary
        try:
            pg_res = await pg_q.insert_data_source(settings, row)
            logger.info(
                f"Successfully inserted data source into PostgreSQL (primary): {row.get('id', 'unknown')}"
            )
        except Exception as e:
            # Handle duplicate key errors
            msg = str(e).lower()
            if "unique" in msg or "duplicate" in msg or "constraint" in msg:
                try:
                    existing = await pg_q.select_data_sources(
                        settings, {"id": row.get("id")}
                    )
                    if isinstance(existing, list) and existing:
                        logger.info(
                            f"Data source already exists in PostgreSQL: {row.get('id')}"
                        )
                        return existing[0]
                    if isinstance(existing, dict):
                        logger.info(
                            f"Data source already exists in PostgreSQL: {row.get('id')}"
                        )
                        return existing
                except Exception:
                    pass

            logger.exception("PostgreSQL insert failed for data_sources")
            raise RuntimeError(
                "Failed to persist data_sources row to primary database (PostgreSQL)"
            )

        # Secondary sync to SQLite (best-effort)
        try:
            await sqlite_q.insert_data_source(settings, row)
            logger.info(
                f"Successfully synced data source to SQLite (secondary): {row.get('id', 'unknown')}"
            )
        except Exception:
            logger.warning(
                f"Failed to sync data source to SQLite (non-critical): {row.get('id', 'unknown')}"
            )

        return pg_res

    else:
        raise RuntimeError(f"Unsupported primary database: {primary_db}")


async def read(
    settings: Settings, filter: Optional[Dict[str, Any]] = None, one: bool = False
) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """Read data sources from primary database."""
    primary_db = get_primary_database(settings)

    if primary_db == "sqlite":
        try:
            rows = await sqlite_q.select_data_sources(settings, filter, one=one)
            row_count = len(rows) if isinstance(rows, list) else 1
            logger.debug("Successfully read %s rows from SQLite (primary)", row_count)
            return cast(Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], rows)
        except Exception:
            logger.exception(
                "Failed reading data_sources from SQLite (primary database)"
            )
            raise RuntimeError("Failed to read from primary database (SQLite)")

    elif primary_db == "postgres":
        try:
            rows = await pg_q.select_data_sources(settings, filter, one=one)
            row_count = len(rows) if isinstance(rows, list) else 1
            logger.debug(
                "Successfully read %s rows from PostgreSQL (primary)", row_count
            )
            return cast(Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], rows)
        except Exception:
            logger.exception(
                "Failed reading data_sources from PostgreSQL (primary database)"
            )
            raise RuntimeError("Failed to read from primary database (PostgreSQL)")

    else:
        raise RuntimeError(f"Unsupported primary database: {primary_db}")


async def update(
    settings: Settings, filter: Dict[str, Any], update_data: Dict[str, Any]
):
    """Update data sources with configurable primary database.

    Strategy: Primary database-first approach
    1) Update primary database (SQLite or PostgreSQL)
    2) Optionally sync to secondary database if available (best-effort, non-blocking)
    """

    # Normalize filter (map 'source_id' -> 'id')
    if filter is None:
        filter = {}
    normalized_filter = {}
    for k, v in (filter or {}).items():
        if k == "source_id":
            normalized_filter["id"] = v
        else:
            normalized_filter[k] = v
    filter = normalized_filter

    primary_db = get_primary_database(settings)

    if primary_db == "sqlite":
        # Primary update in SQLite
        try:
            sqlite_rows = await sqlite_q.update_data_source(
                settings, filter, update_data
            )
            logger.info(
                f"Successfully updated data source in SQLite (primary): {filter}"
            )
        except Exception:
            logger.exception("SQLite update failed for data_sources")
            raise RuntimeError("Failed to update in primary database (SQLite)")

        # Optional PostgreSQL sync (best-effort, non-blocking)
        try:
            await pg_q.update_data_source(settings, filter, update_data)
            logger.info(
                f"Successfully synced update to PostgreSQL (secondary): {filter}"
            )
        except Exception:
            logger.warning(
                f"Failed to sync update to PostgreSQL (non-critical): {filter}"
            )

        return sqlite_rows

    elif primary_db == "postgres":
        # Primary update in PostgreSQL
        try:
            pg_rows = await pg_q.update_data_source(settings, filter, update_data)
            logger.info(
                f"Successfully updated data source in PostgreSQL (primary): {filter}"
            )
        except Exception:
            logger.exception("PostgreSQL update failed for data_sources")
            raise RuntimeError("Failed to update in primary database (PostgreSQL)")

        # Optional SQLite sync (best-effort, non-blocking)
        try:
            await sqlite_q.update_data_source(settings, filter, update_data)
            logger.info(f"Successfully synced update to SQLite (secondary): {filter}")
        except Exception:
            logger.warning(f"Failed to sync update to SQLite (non-critical): {filter}")

        return pg_rows

    else:
        raise RuntimeError(f"Unsupported primary database: {primary_db}")


async def delete(settings: Settings, filter: Dict[str, Any]):
    """Delete data sources with configurable primary database.

    Strategy: Primary database-first approach
    1) Delete from primary database (SQLite or PostgreSQL)
    2) Optionally sync to secondary database if available (best-effort, non-blocking)
    """

    # Normalize filter (map 'source_id' -> 'id')
    if filter is None:
        filter = {}
    normalized_filter = {}
    for k, v in (filter or {}).items():
        if k == "source_id":
            normalized_filter["id"] = v
        else:
            normalized_filter[k] = v
    filter = normalized_filter

    primary_db = get_primary_database(settings)

    if primary_db == "sqlite":
        # Primary delete from SQLite
        try:
            sqlite_rows = await sqlite_q.delete_data_source(settings, filter)
            logger.info(
                f"Successfully deleted data source from SQLite (primary): {filter}"
            )
        except Exception:
            logger.exception("SQLite delete failed for data_sources")
            raise RuntimeError("Failed to delete from primary database (SQLite)")

        # Optional PostgreSQL sync (best-effort, non-blocking)
        try:
            await pg_q.delete_data_source(settings, filter)
            logger.info(
                f"Successfully synced delete to PostgreSQL (secondary): {filter}"
            )
        except Exception:
            logger.warning(
                f"Failed to sync delete to PostgreSQL (non-critical): {filter}"
            )

        return sqlite_rows

    elif primary_db == "postgres":
        # Primary delete from PostgreSQL
        try:
            pg_rows = await pg_q.delete_data_source(settings, filter)
            logger.info(
                f"Successfully deleted data source from PostgreSQL (primary): {filter}"
            )
        except Exception:
            logger.exception("PostgreSQL delete failed for data_sources")
            raise RuntimeError("Failed to delete from primary database (PostgreSQL)")

        # Optional SQLite sync (best-effort, non-blocking)
        try:
            await sqlite_q.delete_data_source(settings, filter)
            logger.info(f"Successfully synced delete to SQLite (secondary): {filter}")
        except Exception:
            logger.warning(f"Failed to sync delete to SQLite (non-critical): {filter}")

        return pg_rows

    else:
        raise RuntimeError(f"Unsupported primary database: {primary_db}")


async def get_by_file_hash(
    settings: Settings, file_hash: str
) -> Optional[Dict[str, Any]]:
    """Retrieve a data source row by the stored file hash in the JSON config."""
    if not file_hash:
        return None

    primary_db = get_primary_database(settings)

    if primary_db == "sqlite":
        try:
            row = await sqlite_q.select_data_source_by_file_hash(settings, file_hash)
            if row:
                return row
        except Exception:
            logger.warning("Primary SQLite file-hash lookup failed", exc_info=True)

        try:
            return await pg_q.select_data_source_by_file_hash(settings, file_hash)
        except Exception:
            logger.warning(
                "Secondary PostgreSQL file-hash lookup failed", exc_info=True
            )
            return None

    elif primary_db == "postgres":
        try:
            row = await pg_q.select_data_source_by_file_hash(settings, file_hash)
            if row:
                return row
        except Exception:
            logger.warning("Primary PostgreSQL file-hash lookup failed", exc_info=True)

        try:
            return await sqlite_q.select_data_source_by_file_hash(settings, file_hash)
        except Exception:
            logger.warning("Secondary SQLite file-hash lookup failed", exc_info=True)
            return None

    else:
        raise RuntimeError(f"Unsupported primary database: {primary_db}")


async def migrate_to_postgres(settings: Settings) -> Dict[str, int]:
    """Migration function to copy all data from SQLite to PostgreSQL.

    This function can be called manually or triggered by a migration button.
    Returns statistics about the migration process.
    """
    from backend.database.async_migrate import migrate_sqlite_to_postgres

    logger.info("Starting migration from SQLite to PostgreSQL...")

    try:
        stats = await migrate_sqlite_to_postgres(settings)
        logger.info(f"Migration completed successfully: {stats}")
        return stats
    except Exception:
        logger.exception("Migration from SQLite to PostgreSQL failed")
        raise RuntimeError("Migration to PostgreSQL failed")


async def get_database_status(settings: Settings) -> Dict[str, Any]:
    """Get status of both SQLite and PostgreSQL databases.

    Returns information about connectivity and record counts.
    """
    primary_db = get_primary_database(settings)

    status: Dict[str, Any] = {
        "sqlite": {"connected": False, "count": 0, "error": None, "path": None},
        "postgres": {"connected": False, "count": 0, "error": None},
        "primary": primary_db,
    }

    # Check SQLite status
    try:
        sqlite_rows = await sqlite_q.select_data_sources(settings, None)
        status["sqlite"]["connected"] = True
        status["sqlite"]["count"] = len(sqlite_rows) if sqlite_rows else 0
        status["sqlite"]["path"] = getattr(settings, "DB_PATH", "registry.db")

    except Exception as e:
        logger.exception("SQLite status check failed")
        status["sqlite"]["error"] = str(e)

    # Check PostgreSQL status
    try:
        pg_rows = await pg_q.select_data_sources(settings, None)
        status["postgres"]["connected"] = True
        status["postgres"]["count"] = len(pg_rows) if pg_rows else 0
    except Exception as e:
        status["postgres"]["error"] = str(e)

    return status
