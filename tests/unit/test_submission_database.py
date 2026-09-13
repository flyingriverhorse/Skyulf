"""Database adapter and connection-pool guarantees used by job reservations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

from backend.database import engine as database_engine
from backend.ml_pipeline._execution.submission import _acquire_reservation, _advisory_key


@pytest.mark.asyncio
@pytest.mark.parametrize("memory", [False, True])
async def test_sqlite_pool_preserves_transaction_isolation_or_memory_database(
    tmp_path, monkeypatch, memory
):
    """File sessions need independent transactions; memory sessions must share their database."""
    url = (
        "sqlite+aiosqlite:///:memory:" if memory else f"sqlite+aiosqlite:///{tmp_path / 'pool.db'}"
    )
    settings = SimpleNamespace(DATABASE_URL=url, DB_ECHO=False)
    monkeypatch.setattr(database_engine, "get_settings", lambda: settings)
    for name in ("async_engine", "async_session_factory", "sync_engine", "sync_session_factory"):
        monkeypatch.setattr(database_engine, name, None)
    await database_engine.init_db()
    try:
        engine = database_engine.async_engine
        async with engine.begin() as connection:
            await connection.execute(text("CREATE TABLE isolation_probe (id INTEGER)"))
        async with engine.connect() as owner, engine.connect() as observer:
            await owner.exec_driver_sql("BEGIN IMMEDIATE")
            await owner.execute(text("INSERT INTO isolation_probe VALUES (1)"))
            count = (await observer.execute(text("SELECT COUNT(*) FROM isolation_probe"))).scalar()
            assert count == (1 if memory else 0)
            assert isinstance(engine.pool, StaticPool) is memory
            await owner.rollback()
    finally:
        await database_engine.close_db()


@pytest.mark.asyncio
async def test_postgresql_reservation_uses_bound_transaction_advisory_key():
    """PostgreSQL must wait on the stable key instead of skipping a locked or absent row."""
    connection = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"), execute=AsyncMock())
    await _acquire_reservation(connection, ("dataset", "node", 2))
    statement, parameters = connection.execute.await_args.args
    assert str(statement) == "SELECT pg_advisory_xact_lock(:key)"
    assert parameters == {"key": _advisory_key(("dataset", "node", 2))}


def test_advisory_keys_distinguish_tuple_boundaries_and_branch_indices():
    """Colon-containing identifiers and parallel branches must retain independent locks."""
    values = {
        _advisory_key(("a:b", "c", 0)),
        _advisory_key(("a", "b:c", 0)),
        _advisory_key(("a:b", "c", 1)),
    }
    assert len(values) == 3
    assert all(-(2**63) <= value < 2**63 for value in values)


@pytest.mark.asyncio
async def test_unsupported_database_is_rejected_before_querying():
    """An unimplemented dialect must not silently fall back to unprotected creation."""
    connection = SimpleNamespace(dialect=SimpleNamespace(name="unsupported"), execute=AsyncMock())
    with pytest.raises(ValueError, match="Unsupported submission database"):
        await _acquire_reservation(connection, ("dataset", "node", 0))
    connection.execute.assert_not_awaited()
