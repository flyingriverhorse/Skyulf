"""
Async orchestration layer for data_sources operations.

This module centralizes the logic to write/read/update/delete the
`data_sources` registry across PostgreSQL and SQLite. It intentionally
keeps behavior separate from the generic database CRUD dispatcher.
"""

# pylint: disable=broad-exception-caught

import logging
from typing import Any

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


async def _lookup_existing_on_duplicate(
    primary_mod: Any, primary_name: str, settings: Settings, row: dict[str, Any], error: Exception
) -> tuple[Any, bool]:
    """Check whether `error` looks like a duplicate-key error and, if so, look up the existing row.

    Returns `(result, found)`. `found` is True only when the row already exists
    in `primary_mod` and was located via lookup.
    """
    msg = str(error).lower()
    if not ("unique" in msg or "duplicate" in msg or "constraint" in msg):
        return None, False

    try:
        existing = await primary_mod.select_data_sources(settings, {"id": row.get("id")})
        if isinstance(existing, list) and existing:
            logger.info(f"Data source already exists in {primary_name}: {row.get('id')}")
            return existing[0], True
        if isinstance(existing, dict):
            logger.info(f"Data source already exists in {primary_name}: {row.get('id')}")
            return existing, True
    except Exception:
        # Ignore if check fails, try insert
        logger.debug(f"Failed to check for existing data source in {primary_name}", exc_info=True)

    return None, False


async def _insert_primary_with_dup_check(
    primary_mod: Any, primary_name: str, settings: Settings, row: dict[str, Any]
) -> tuple[Any, bool]:
    """Insert `row` into `primary_mod`, falling back to a lookup on duplicate-key errors.

    Returns `(result, is_duplicate)`. `is_duplicate` is True when the insert failed
    because the row already exists and `result` is the pre-existing row instead.
    """
    try:
        res = await primary_mod.insert_data_source(settings, row)
        logger.info(
            f"Successfully inserted data source into {primary_name} (primary): {row.get('id', 'unknown')}"
        )
        return res, False
    except Exception as e:
        existing, found = await _lookup_existing_on_duplicate(
            primary_mod, primary_name, settings, row, e
        )
        if found:
            return existing, True

        logger.exception(f"{primary_name} insert failed for data_sources")
        raise RuntimeError(
            f"Failed to persist data_sources row to primary database ({primary_name})"
        ) from e


async def _sync_secondary(
    secondary_mod: Any, secondary_name: str, settings: Settings, row: dict[str, Any]
) -> None:
    """Best-effort sync of `row` into the secondary database; failures are logged, not raised."""
    try:
        await secondary_mod.insert_data_source(settings, row)
        logger.info(
            f"Successfully synced data source to {secondary_name} (secondary): {row.get('id', 'unknown')}"
        )
    except Exception:
        logger.warning(
            f"Failed to sync data source to {secondary_name} (non-critical): {row.get('id', 'unknown')}"
        )


# Maps primary DB name -> (primary_module, secondary_module, primary_label, secondary_label)
_DB_PEERS: dict[str, tuple[Any, Any, str, str]] = {
    "sqlite": (sqlite_q, pg_q, "SQLite", "PostgreSQL"),
    "postgres": (pg_q, sqlite_q, "PostgreSQL", "SQLite"),
}


async def create(settings: Settings, row: dict[str, Any]) -> Any:
    """Create a new data source record with configurable primary database.

    Strategy: DB_PRIMARY-first approach
    1) Insert into primary database (SQLite or PostgreSQL)
    2) Optionally sync to secondary database if available (best-effort, non-blocking)
    """

    primary_db = get_primary_database(settings)
    peers = _DB_PEERS.get(primary_db)
    if peers is None:
        raise RuntimeError(f"Unsupported primary database: {primary_db}")
    primary_mod, secondary_mod, primary_name, secondary_name = peers

    result, is_duplicate = await _insert_primary_with_dup_check(
        primary_mod, primary_name, settings, row
    )
    if is_duplicate:
        return result

    # Secondary sync (best-effort)
    await _sync_secondary(secondary_mod, secondary_name, settings, row)

    return result


async def read(
    settings: Settings, filter_dict: dict[str, Any] | None = None, one: bool = False
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Read data sources from primary database."""
    primary_db = get_primary_database(settings)

    if primary_db == "sqlite":
        try:
            rows = await sqlite_q.select_data_sources(settings, filter_dict, one=one)
            row_count = len(rows) if isinstance(rows, list) else 1
            logger.debug("Successfully read %s rows from SQLite (primary)", row_count)
            return rows
        except Exception:
            logger.exception("Failed reading data_sources from SQLite (primary database)")
            raise RuntimeError("Failed to read from primary database (SQLite)") from None

    elif primary_db == "postgres":
        try:
            rows = await pg_q.select_data_sources(settings, filter_dict, one=one)
            row_count = len(rows) if isinstance(rows, list) else 1
            logger.debug("Successfully read %s rows from PostgreSQL (primary)", row_count)
            return rows
        except Exception:
            logger.exception("Failed reading data_sources from PostgreSQL (primary database)")
            raise RuntimeError("Failed to read from primary database (PostgreSQL)") from None

    else:
        raise RuntimeError(f"Unsupported primary database: {primary_db}")


def _normalize_filter(filter_dict: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a data_sources filter dict, mapping the legacy 'source_id' key to 'id'."""
    normalized: dict[str, Any] = {}
    for k, v in (filter_dict or {}).items():
        normalized["id" if k == "source_id" else k] = v
    return normalized


def _get_db_peers(settings: Settings) -> tuple[Any, Any, str, str]:
    """Resolve the (primary_module, secondary_module, primary_name, secondary_name) peers.

    Raises RuntimeError if the configured primary database is unsupported.
    """
    primary_db = get_primary_database(settings)
    peers = _DB_PEERS.get(primary_db)
    if peers is None:
        raise RuntimeError(f"Unsupported primary database: {primary_db}")
    return peers


async def _update_primary(
    primary_mod: Any,
    primary_name: str,
    settings: Settings,
    filter_dict: dict[str, Any],
    update_data: dict[str, Any],
):
    """Update the primary database, raising RuntimeError if the update fails."""
    try:
        rows = await primary_mod.update_data_source(settings, filter_dict, update_data)
        logger.info(f"Successfully updated data source in {primary_name} (primary): {filter_dict}")
        return rows
    except Exception:
        logger.exception(f"{primary_name} update failed for data_sources")
        raise RuntimeError(f"Failed to update in primary database ({primary_name})") from None


async def _sync_secondary_update(
    secondary_mod: Any,
    secondary_name: str,
    settings: Settings,
    filter_dict: dict[str, Any],
    update_data: dict[str, Any],
) -> None:
    """Best-effort sync of an update to the secondary database; failures are logged, not raised."""
    try:
        await secondary_mod.update_data_source(settings, filter_dict, update_data)
        logger.info(f"Successfully synced update to {secondary_name} (secondary): {filter_dict}")
    except Exception:
        logger.warning(f"Failed to sync update to {secondary_name} (non-critical): {filter_dict}")


async def update(settings: Settings, filter_dict: dict[str, Any], update_data: dict[str, Any]):
    """Update data sources with configurable primary database.

    Strategy: Primary database-first approach
    1) Update primary database (SQLite or PostgreSQL)
    2) Optionally sync to secondary database if available (best-effort, non-blocking)
    """
    filter_dict = _normalize_filter(filter_dict)
    primary_mod, secondary_mod, primary_name, secondary_name = _get_db_peers(settings)

    rows = await _update_primary(primary_mod, primary_name, settings, filter_dict, update_data)
    await _sync_secondary_update(secondary_mod, secondary_name, settings, filter_dict, update_data)
    return rows


async def _delete_primary(
    primary_mod: Any, primary_name: str, settings: Settings, filter_dict: dict[str, Any]
):
    """Delete from the primary database, raising RuntimeError if the delete fails."""
    try:
        rows = await primary_mod.delete_data_source(settings, filter_dict)
        logger.info(
            f"Successfully deleted data source from {primary_name} (primary): {filter_dict}"
        )
        return rows
    except Exception:
        logger.exception(f"{primary_name} delete failed for data_sources")
        raise RuntimeError(f"Failed to delete from primary database ({primary_name})") from None


async def _sync_secondary_delete(
    secondary_mod: Any, secondary_name: str, settings: Settings, filter_dict: dict[str, Any]
) -> None:
    """Best-effort sync of a delete to the secondary database; failures are logged, not raised."""
    try:
        await secondary_mod.delete_data_source(settings, filter_dict)
        logger.info(f"Successfully synced delete to {secondary_name} (secondary): {filter_dict}")
    except Exception:
        logger.warning(f"Failed to sync delete to {secondary_name} (non-critical): {filter_dict}")


async def delete(settings: Settings, filter_dict: dict[str, Any]):
    """Delete data sources with configurable primary database.

    Strategy: Primary database-first approach
    1) Delete from primary database (SQLite or PostgreSQL)
    2) Optionally sync to secondary database if available (best-effort, non-blocking)
    """
    filter_dict = _normalize_filter(filter_dict)
    primary_mod, secondary_mod, primary_name, secondary_name = _get_db_peers(settings)

    rows = await _delete_primary(primary_mod, primary_name, settings, filter_dict)
    await _sync_secondary_delete(secondary_mod, secondary_name, settings, filter_dict)
    return rows


async def _lookup_by_hash(
    mod: Any, mod_label: str, settings: Settings, file_hash: str
) -> dict[str, Any] | None:
    """Try a file-hash lookup against `mod`; return None (and log a warning) on failure."""
    try:
        return await mod.select_data_source_by_file_hash(settings, file_hash)
    except Exception:
        logger.warning(f"{mod_label} file-hash lookup failed", exc_info=True)
        return None


async def get_by_file_hash(settings: Settings, file_hash: str) -> dict[str, Any] | None:
    """Retrieve a data source row by the stored file hash in the JSON config."""
    if not file_hash:
        return None

    primary_mod, secondary_mod, primary_name, secondary_name = _get_db_peers(settings)

    row = await _lookup_by_hash(primary_mod, f"Primary {primary_name}", settings, file_hash)
    if row:
        return row

    return await _lookup_by_hash(secondary_mod, f"Secondary {secondary_name}", settings, file_hash)


async def migrate_to_postgres(settings: Settings) -> dict[str, int]:
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
        raise RuntimeError("Migration to PostgreSQL failed") from None


async def get_database_status(settings: Settings) -> dict[str, Any]:
    """Get status of both SQLite and PostgreSQL databases.

    Returns information about connectivity and record counts.
    """
    primary_db = get_primary_database(settings)

    sqlite_status: dict[str, Any] = {
        "connected": False,
        "count": 0,
        "error": None,
        "path": None,
    }
    postgres_status: dict[str, Any] = {"connected": False, "count": 0, "error": None}
    status: dict[str, Any] = {
        "sqlite": sqlite_status,
        "postgres": postgres_status,
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
