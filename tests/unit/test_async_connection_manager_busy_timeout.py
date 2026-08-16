"""Regression test for the config-driven SQLite busy_timeout PRAGMA.

`AsyncSQLiteConnectionManager` used to hardcode `PRAGMA busy_timeout=30000`
in two places; it now reads `Settings.DB_SQLITE_BUSY_TIMEOUT_MS`. This test
verifies the PRAGMA actually reflects the configured value at both call
sites (initialization optimizations and per-connection setup).
"""

import asyncio

import pytest

from backend.database.async_connection_manager import AsyncSQLiteConnectionManager


class _FakeSettings:
    DB_SQLITE_BUSY_TIMEOUT_MS = 12345


@pytest.mark.asyncio
async def test_setup_database_optimizations_uses_configured_busy_timeout(tmp_path, monkeypatch):
    """The WAL-mode setup issues a busy_timeout PRAGMA using DB_SQLITE_BUSY_TIMEOUT_MS.

    busy_timeout is a per-connection PRAGMA (not persisted to the db file), so we
    capture the executed SQL directly rather than re-querying after disconnect.
    """
    import aiosqlite

    import backend.database.async_connection_manager as mod

    monkeypatch.setattr(mod, "get_settings", lambda: _FakeSettings())

    executed: list[str] = []
    original_execute = aiosqlite.Connection.execute

    async def _spy_execute(self, sql, *args, **kwargs):
        executed.append(sql)
        return await original_execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(aiosqlite.Connection, "execute", _spy_execute)

    db_path = tmp_path / "busy_timeout.db"
    manager = AsyncSQLiteConnectionManager(str(db_path))
    await manager._setup_database_optimizations()

    assert "PRAGMA busy_timeout=12345" in executed


@pytest.mark.asyncio
async def test_get_connection_uses_configured_busy_timeout(tmp_path, monkeypatch):
    """The per-connection PRAGMA in get_connection() reflects DB_SQLITE_BUSY_TIMEOUT_MS."""
    import backend.database.async_connection_manager as mod

    monkeypatch.setattr(mod, "get_settings", lambda: _FakeSettings())

    db_path = tmp_path / "busy_timeout2.db"
    manager = AsyncSQLiteConnectionManager(str(db_path))
    await manager.initialize()

    async with manager.get_connection() as conn:
        cursor = await conn.execute("PRAGMA busy_timeout")
        row = await cursor.fetchone()
        assert row[0] == 12345
