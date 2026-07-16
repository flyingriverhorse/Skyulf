"""Focused unit tests for backend.database.data_sources.async_data_sources_crud.

These tests exercise the primary/secondary DB orchestration helpers using
mocked query modules so we don't depend on a real Postgres/SQLite connection.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.config import get_settings
from backend.database.data_sources import async_data_sources_crud as crud


@pytest.fixture
def settings():
    """Return the shared test Settings instance."""
    return get_settings()


async def test_lookup_existing_on_duplicate_not_duplicate_error(settings):
    """A non-duplicate error should be reported as not found, without a lookup call."""
    primary_mod = AsyncMock()
    result, found = await crud._lookup_existing_on_duplicate(
        primary_mod, "SQLite", settings, {"id": "abc"}, Exception("some other failure")
    )
    assert result is None
    assert found is False
    primary_mod.select_data_sources.assert_not_called()


async def test_lookup_existing_on_duplicate_found_list(settings):
    """A duplicate-key error should trigger a lookup; a list result returns its first row."""
    primary_mod = AsyncMock()
    primary_mod.select_data_sources.return_value = [{"id": "abc", "name": "existing"}]
    result, found = await crud._lookup_existing_on_duplicate(
        primary_mod, "SQLite", settings, {"id": "abc"}, Exception("UNIQUE constraint failed")
    )
    assert found is True
    assert result == {"id": "abc", "name": "existing"}


async def test_lookup_existing_on_duplicate_found_dict(settings):
    """A duplicate-key error whose lookup returns a dict is also treated as found."""
    primary_mod = AsyncMock()
    primary_mod.select_data_sources.return_value = {"id": "abc", "name": "existing"}
    result, found = await crud._lookup_existing_on_duplicate(
        primary_mod, "SQLite", settings, {"id": "abc"}, Exception("duplicate key value")
    )
    assert found is True
    assert result == {"id": "abc", "name": "existing"}


async def test_lookup_existing_on_duplicate_lookup_fails(settings):
    """If the duplicate-check lookup itself raises, treat it as not found."""
    primary_mod = AsyncMock()
    primary_mod.select_data_sources.side_effect = RuntimeError("db down")
    result, found = await crud._lookup_existing_on_duplicate(
        primary_mod, "SQLite", settings, {"id": "abc"}, Exception("constraint violation")
    )
    assert result is None
    assert found is False


async def test_lookup_existing_on_duplicate_empty_list(settings):
    """A duplicate-key error whose lookup returns an empty list is treated as not found."""
    primary_mod = AsyncMock()
    primary_mod.select_data_sources.return_value = []
    result, found = await crud._lookup_existing_on_duplicate(
        primary_mod, "SQLite", settings, {"id": "abc"}, Exception("unique violation")
    )
    assert result is None
    assert found is False


async def test_insert_primary_with_dup_check_success(settings):
    """A successful insert returns the inserted row and is_duplicate=False."""
    primary_mod = AsyncMock()
    primary_mod.insert_data_source.return_value = {"id": "abc"}
    result, is_duplicate = await crud._insert_primary_with_dup_check(
        primary_mod, "SQLite", settings, {"id": "abc"}
    )
    assert result == {"id": "abc"}
    assert is_duplicate is False


async def test_insert_primary_with_dup_check_duplicate_found(settings):
    """An insert failing with a duplicate error, resolved via lookup, returns is_duplicate=True."""
    primary_mod = AsyncMock()
    primary_mod.insert_data_source.side_effect = Exception("UNIQUE constraint failed: id")
    primary_mod.select_data_sources.return_value = [{"id": "abc"}]
    result, is_duplicate = await crud._insert_primary_with_dup_check(
        primary_mod, "SQLite", settings, {"id": "abc"}
    )
    assert result == {"id": "abc"}
    assert is_duplicate is True


async def test_insert_primary_with_dup_check_raises(settings):
    """An insert failing with a non-duplicate error raises a wrapped RuntimeError."""
    primary_mod = AsyncMock()
    primary_mod.insert_data_source.side_effect = Exception("connection refused")
    with pytest.raises(RuntimeError, match="Failed to persist data_sources row"):
        await crud._insert_primary_with_dup_check(primary_mod, "SQLite", settings, {"id": "abc"})


async def test_sync_secondary_success(settings):
    """A successful secondary sync completes without error."""
    secondary_mod = AsyncMock()
    await crud._sync_secondary(secondary_mod, "PostgreSQL", settings, {"id": "abc"})
    secondary_mod.insert_data_source.assert_awaited_once_with(settings, {"id": "abc"})


async def test_sync_secondary_failure_is_swallowed(settings):
    """A failing secondary sync is logged but does not raise."""
    secondary_mod = AsyncMock()
    secondary_mod.insert_data_source.side_effect = Exception("secondary unreachable")
    # Should not raise.
    await crud._sync_secondary(secondary_mod, "PostgreSQL", settings, {"id": "abc"})


async def test_create_unsupported_primary_db(settings):
    """create() raises RuntimeError for an unsupported DB_PRIMARY value."""
    with (
        patch.object(crud, "get_primary_database", return_value="oracle"),
        pytest.raises(RuntimeError, match="Unsupported primary database"),
    ):
        await crud.create(settings, {"id": "abc"})


async def test_create_success_syncs_secondary(settings):
    """create() inserts into primary then syncs secondary when not a duplicate."""
    with (
        patch.object(crud, "get_primary_database", return_value="sqlite"),
        patch.object(crud.sqlite_q, "insert_data_source", new=AsyncMock(return_value={"id": "x"})),
        patch.object(crud.pg_q, "insert_data_source", new=AsyncMock()) as pg_insert,
    ):
        result = await crud.create(settings, {"id": "x"})
    assert result == {"id": "x"}
    pg_insert.assert_awaited_once()


async def test_read_sqlite_success(settings):
    """read() with sqlite primary returns rows from sqlite_q.select_data_sources."""
    with (
        patch.object(crud, "get_primary_database", return_value="sqlite"),
        patch.object(
            crud.sqlite_q, "select_data_sources", new=AsyncMock(return_value=[{"id": "a"}])
        ),
    ):
        rows = await crud.read(settings, {"id": "a"})
    assert rows == [{"id": "a"}]


async def test_read_sqlite_failure_raises(settings):
    """read() wraps sqlite failures in a RuntimeError."""
    with (
        patch.object(crud, "get_primary_database", return_value="sqlite"),
        patch.object(
            crud.sqlite_q,
            "select_data_sources",
            new=AsyncMock(side_effect=Exception("boom")),
        ),
        pytest.raises(RuntimeError, match="Failed to read from primary database"),
    ):
        await crud.read(settings, {"id": "a"})


async def test_read_postgres_success(settings):
    """read() with postgres primary returns rows from pg_q.select_data_sources."""
    with (
        patch.object(crud, "get_primary_database", return_value="postgres"),
        patch.object(crud.pg_q, "select_data_sources", new=AsyncMock(return_value={"id": "a"})),
    ):
        rows = await crud.read(settings, {"id": "a"}, one=True)
    assert rows == {"id": "a"}


async def test_read_postgres_failure_raises(settings):
    """read() wraps postgres failures in a RuntimeError."""
    with (
        patch.object(crud, "get_primary_database", return_value="postgres"),
        patch.object(
            crud.pg_q, "select_data_sources", new=AsyncMock(side_effect=Exception("boom"))
        ),
        pytest.raises(RuntimeError, match="Failed to read from primary database"),
    ):
        await crud.read(settings, {"id": "a"})


async def test_read_unsupported_primary_db(settings):
    """read() raises RuntimeError for an unsupported DB_PRIMARY value."""
    with (
        patch.object(crud, "get_primary_database", return_value="oracle"),
        pytest.raises(RuntimeError, match="Unsupported primary database"),
    ):
        await crud.read(settings, {"id": "a"})


def test_normalize_filter_maps_source_id():
    """_normalize_filter renames the legacy 'source_id' key to 'id'."""
    assert crud._normalize_filter({"source_id": "abc", "name": "n"}) == {"id": "abc", "name": "n"}


def test_normalize_filter_handles_none():
    """_normalize_filter tolerates a None filter dict."""
    assert crud._normalize_filter(None) == {}


def test_get_db_peers_sqlite(settings):
    """_get_db_peers resolves sqlite as primary correctly."""
    with patch.object(crud, "get_primary_database", return_value="sqlite"):
        primary_mod, secondary_mod, primary_name, secondary_name = crud._get_db_peers(settings)
    assert primary_mod is crud.sqlite_q
    assert secondary_mod is crud.pg_q
    assert primary_name == "SQLite"
    assert secondary_name == "PostgreSQL"


def test_get_db_peers_unsupported(settings):
    """_get_db_peers raises RuntimeError for an unsupported primary DB."""
    with (
        patch.object(crud, "get_primary_database", return_value="oracle"),
        pytest.raises(RuntimeError, match="Unsupported primary database"),
    ):
        crud._get_db_peers(settings)


async def test_update_primary_success(settings):
    """_update_primary returns the updated rows on success."""
    primary_mod = AsyncMock()
    primary_mod.update_data_source.return_value = [{"id": "a", "name": "new"}]
    rows = await crud._update_primary(primary_mod, "SQLite", settings, {"id": "a"}, {"name": "new"})
    assert rows == [{"id": "a", "name": "new"}]


async def test_update_primary_failure_raises(settings):
    """_update_primary wraps failures in a RuntimeError."""
    primary_mod = AsyncMock()
    primary_mod.update_data_source.side_effect = Exception("boom")
    with pytest.raises(RuntimeError, match="Failed to update in primary database"):
        await crud._update_primary(primary_mod, "SQLite", settings, {"id": "a"}, {"name": "new"})


async def test_sync_secondary_update_success(settings):
    """_sync_secondary_update completes without error on success."""
    secondary_mod = AsyncMock()
    await crud._sync_secondary_update(
        secondary_mod, "PostgreSQL", settings, {"id": "a"}, {"name": "n"}
    )
    secondary_mod.update_data_source.assert_awaited_once()


async def test_sync_secondary_update_failure_swallowed(settings):
    """_sync_secondary_update logs and does not raise when the secondary update fails."""
    secondary_mod = AsyncMock()
    secondary_mod.update_data_source.side_effect = Exception("unreachable")
    await crud._sync_secondary_update(
        secondary_mod, "PostgreSQL", settings, {"id": "a"}, {"name": "n"}
    )


async def test_update_end_to_end(settings):
    """update() normalizes the filter, updates primary and syncs secondary."""
    with (
        patch.object(crud, "get_primary_database", return_value="sqlite"),
        patch.object(
            crud.sqlite_q, "update_data_source", new=AsyncMock(return_value=[{"id": "a"}])
        ),
        patch.object(crud.pg_q, "update_data_source", new=AsyncMock()) as pg_update,
    ):
        rows = await crud.update(settings, {"source_id": "a"}, {"name": "new"})
    assert rows == [{"id": "a"}]
    pg_update.assert_awaited_once_with(settings, {"id": "a"}, {"name": "new"})


async def test_delete_primary_success(settings):
    """_delete_primary returns the deleted rows on success."""
    primary_mod = AsyncMock()
    primary_mod.delete_data_source.return_value = [{"id": "a"}]
    rows = await crud._delete_primary(primary_mod, "SQLite", settings, {"id": "a"})
    assert rows == [{"id": "a"}]


async def test_delete_primary_failure_raises(settings):
    """_delete_primary wraps failures in a RuntimeError."""
    primary_mod = AsyncMock()
    primary_mod.delete_data_source.side_effect = Exception("boom")
    with pytest.raises(RuntimeError, match="Failed to delete from primary database"):
        await crud._delete_primary(primary_mod, "SQLite", settings, {"id": "a"})


async def test_sync_secondary_delete_success(settings):
    """_sync_secondary_delete completes without error on success."""
    secondary_mod = AsyncMock()
    await crud._sync_secondary_delete(secondary_mod, "PostgreSQL", settings, {"id": "a"})
    secondary_mod.delete_data_source.assert_awaited_once_with(settings, {"id": "a"})


async def test_sync_secondary_delete_failure_swallowed(settings):
    """_sync_secondary_delete logs and does not raise when the secondary delete fails."""
    secondary_mod = AsyncMock()
    secondary_mod.delete_data_source.side_effect = Exception("unreachable")
    await crud._sync_secondary_delete(secondary_mod, "PostgreSQL", settings, {"id": "a"})


async def test_delete_end_to_end(settings):
    """delete() normalizes the filter, deletes from primary and syncs secondary."""
    with (
        patch.object(crud, "get_primary_database", return_value="sqlite"),
        patch.object(
            crud.sqlite_q, "delete_data_source", new=AsyncMock(return_value=[{"id": "a"}])
        ),
        patch.object(crud.pg_q, "delete_data_source", new=AsyncMock()) as pg_delete,
    ):
        rows = await crud.delete(settings, {"source_id": "a"})
    assert rows == [{"id": "a"}]
    pg_delete.assert_awaited_once_with(settings, {"id": "a"})


async def test_lookup_by_hash_success(settings):
    """_lookup_by_hash returns the row found by the module's hash lookup."""
    mod = AsyncMock()
    mod.select_data_source_by_file_hash.return_value = {"id": "a"}
    result = await crud._lookup_by_hash(mod, "Primary SQLite", settings, "abc123")
    assert result == {"id": "a"}


async def test_lookup_by_hash_failure_returns_none(settings):
    """_lookup_by_hash swallows lookup errors and returns None."""
    mod = AsyncMock()
    mod.select_data_source_by_file_hash.side_effect = Exception("boom")
    result = await crud._lookup_by_hash(mod, "Primary SQLite", settings, "abc123")
    assert result is None


async def test_get_by_file_hash_empty_hash_returns_none(settings):
    """get_by_file_hash short-circuits and returns None for a falsy file_hash."""
    assert await crud.get_by_file_hash(settings, "") is None


async def test_get_by_file_hash_found_in_primary(settings):
    """get_by_file_hash returns the primary DB's row without checking the secondary."""
    with (
        patch.object(crud, "get_primary_database", return_value="sqlite"),
        patch.object(
            crud.sqlite_q,
            "select_data_source_by_file_hash",
            new=AsyncMock(return_value={"id": "a"}),
        ),
        patch.object(crud.pg_q, "select_data_source_by_file_hash", new=AsyncMock()) as pg_lookup,
    ):
        result = await crud.get_by_file_hash(settings, "abc123")
    assert result == {"id": "a"}
    pg_lookup.assert_not_called()


async def test_get_by_file_hash_falls_back_to_secondary(settings):
    """get_by_file_hash falls back to the secondary DB when primary lookup misses."""
    with (
        patch.object(crud, "get_primary_database", return_value="sqlite"),
        patch.object(
            crud.sqlite_q,
            "select_data_source_by_file_hash",
            new=AsyncMock(return_value=None),
        ),
        patch.object(
            crud.pg_q,
            "select_data_source_by_file_hash",
            new=AsyncMock(return_value={"id": "b"}),
        ),
    ):
        result = await crud.get_by_file_hash(settings, "abc123")
    assert result == {"id": "b"}


async def test_migrate_to_postgres_success(settings):
    """migrate_to_postgres returns migration stats on success."""
    with patch(
        "backend.database.async_migrate.migrate_sqlite_to_postgres",
        new=AsyncMock(return_value={"migrated": 5}),
    ):
        stats = await crud.migrate_to_postgres(settings)
    assert stats == {"migrated": 5}


async def test_migrate_to_postgres_failure_raises(settings):
    """migrate_to_postgres wraps failures in a RuntimeError."""
    with (
        patch(
            "backend.database.async_migrate.migrate_sqlite_to_postgres",
            new=AsyncMock(side_effect=Exception("boom")),
        ),
        pytest.raises(RuntimeError, match="Migration to PostgreSQL failed"),
    ):
        await crud.migrate_to_postgres(settings)


async def test_get_database_status_success_both(settings):
    """get_database_status reports connected=True and counts for both DBs on success."""
    with (
        patch.object(
            crud.sqlite_q, "select_data_sources", new=AsyncMock(return_value=[{"id": "a"}])
        ),
        patch.object(
            crud.pg_q, "select_data_sources", new=AsyncMock(return_value=[{"id": "a"}, {"id": "b"}])
        ),
    ):
        status = await crud.get_database_status(settings)
    assert status["sqlite"]["connected"] is True
    assert status["sqlite"]["count"] == 1
    assert status["postgres"]["connected"] is True
    assert status["postgres"]["count"] == 2


async def test_get_database_status_failures(settings):
    """get_database_status records errors for both DBs when queries fail."""
    with (
        patch.object(
            crud.sqlite_q,
            "select_data_sources",
            new=AsyncMock(side_effect=Exception("sqlite down")),
        ),
        patch.object(
            crud.pg_q, "select_data_sources", new=AsyncMock(side_effect=Exception("pg down"))
        ),
    ):
        status = await crud.get_database_status(settings)
    assert status["sqlite"]["connected"] is False
    assert "sqlite down" in status["sqlite"]["error"]
    assert status["postgres"]["connected"] is False
    assert "pg down" in status["postgres"]["error"]
