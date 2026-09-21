"""Migration failures must prevent startup instead of masquerading as skips."""

import asyncio
import os
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import event, inspect, text
from sqlalchemy.exc import DatabaseError, NoSuchTableError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

import backend.database.engine as db_engine
import backend.database.models  # noqa: F401 - register model metadata


@pytest_asyncio.fixture(params=["sqlite", "postgresql"])
async def migration_engine(monkeypatch, request, tmp_path):
    """Isolate migrations from the application's configured database."""
    admin = None
    schema = f"qw88_{uuid4().hex}"
    if request.param == "postgresql":
        url = os.environ.get("SKYULF_TEST_POSTGRES_URL")
        if not url:
            pytest.skip("Set SKYULF_TEST_POSTGRES_URL to an isolated PostgreSQL test database")
        admin = create_async_engine(url)
        async with admin.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'migrations.db'}")
    monkeypatch.setattr(db_engine, "async_engine", engine)
    try:
        yield engine
    finally:
        await engine.dispose()
        if admin is not None:
            async with admin.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            await admin.dispose()


async def create_old_schema(engine):
    """Create the required pre-migration tables with existing data."""
    async with engine.begin() as conn:
        for name in ("training_jobs", "deployments", "drift_check_results"):
            await conn.execute(text(f"CREATE TABLE {name} (id INTEGER PRIMARY KEY)"))
            await conn.execute(text(f"INSERT INTO {name} (id) VALUES (1)"))


@pytest.mark.asyncio
async def test_missing_required_table_fails(migration_engine):
    """An incomplete schema cannot be reported as successfully migrated."""
    with pytest.raises(NoSuchTableError, match="training_jobs"):
        await db_engine._run_migrations()


@pytest.mark.asyncio
async def test_sqlite_denied_ddl_propagates(migration_engine):
    """A real SQLite read-only failure must abort startup."""
    if migration_engine.dialect.name != "sqlite":
        pytest.skip("SQLite read-only connection regression")
    await create_old_schema(migration_engine)
    async with migration_engine.connect() as conn:
        await conn.execute(text("PRAGMA query_only = ON"))
    try:
        with pytest.raises(DatabaseError, match="readonly"):
            await db_engine._run_migrations()
    finally:
        async with migration_engine.connect() as conn:
            await conn.execute(text("PRAGMA query_only = OFF"))


@pytest.mark.asyncio
async def test_old_schema_upgrades_and_repeated_startup_skips_ddl(migration_engine):
    """Upgrade existing rows once and avoid duplicate-column exceptions entirely."""
    await create_old_schema(migration_engine)
    await db_engine._run_migrations()
    statements = []

    @event.listens_for(migration_engine.sync_engine, "before_cursor_execute")
    def record_statement(conn, cursor, statement, parameters, context, executemany):
        """Record actual SQL to detect duplicate ALTER attempts."""
        statements.append(statement)

    await db_engine._run_migrations()
    async with migration_engine.connect() as conn:
        enabled = await conn.scalar(text("SELECT tuned_thresholds_enabled FROM training_jobs"))
        status = await conn.scalar(text("SELECT status FROM drift_check_results"))
    assert enabled == 0
    assert status == "new"
    assert not any(sql.startswith("ALTER TABLE") for sql in statements)


@pytest.mark.asyncio
async def test_fresh_schema_does_not_attempt_ddl(migration_engine):
    """Current model metadata should need no incremental changes or legacy tables."""
    async with migration_engine.begin() as conn:
        await conn.run_sync(db_engine.Base.metadata.create_all)
    statements = []

    @event.listens_for(migration_engine.sync_engine, "before_cursor_execute")
    def record_statement(conn, cursor, statement, parameters, context, executemany):
        """Detect attempts to add columns already provided by metadata."""
        statements.append(statement)

    await db_engine._run_migrations()
    assert not any(sql.startswith("ALTER TABLE") for sql in statements)


@pytest.mark.asyncio
async def test_existing_legacy_tables_still_upgrade(migration_engine):
    """Preserve the historical upgrade path without requiring removed models."""
    await create_old_schema(migration_engine)
    async with migration_engine.begin() as conn:
        for name in ("basic_training_jobs", "advanced_tuning_jobs"):
            await conn.execute(text(f"CREATE TABLE {name} (id INTEGER PRIMARY KEY)"))
    await db_engine._run_migrations()
    async with migration_engine.connect() as conn:
        for name in ("basic_training_jobs", "advanced_tuning_jobs"):
            columns = await conn.run_sync(
                lambda sync, table: inspect(sync).get_columns(table), name
            )
            assert "promoted_at" in {column["name"] for column in columns}


@pytest.mark.asyncio
async def test_concurrent_migrations_accept_only_verified_duplicate_columns(
    migration_engine, monkeypatch
):
    """Two workers inspecting the same old schema must both finish startup."""
    await create_old_schema(migration_engine)
    original = AsyncConnection.run_sync
    barrier = asyncio.Barrier(2)

    async def pause_after_inspection(self, fn, *args, **kwargs):
        """Force real transactions to observe the same missing column."""
        result = await original(self, fn, *args, **kwargs)
        if (
            fn.__name__ == "needs_column"
            and args == ("training_jobs", "tuned_thresholds")
            and result
        ):
            await asyncio.wait_for(barrier.wait(), timeout=10)
        return result

    monkeypatch.setattr(AsyncConnection, "run_sync", pause_after_inspection)
    outcomes = await asyncio.wait_for(
        asyncio.gather(
            db_engine._run_migrations(), db_engine._run_migrations(), return_exceptions=True
        ),
        timeout=30,
    )
    assert outcomes == [None, None]


@pytest.mark.asyncio
async def test_postgres_permission_error_propagates(migration_engine):
    """A real role without table ownership must not silently skip ALTER failures."""
    if migration_engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL table-owner permission regression")
    await create_old_schema(migration_engine)
    role = f"qw88_reader_{uuid4().hex}"
    async with migration_engine.begin() as conn:
        schema = await conn.scalar(text("SELECT current_schema()"))
        await conn.execute(text(f'CREATE ROLE "{role}" NOLOGIN'))
        await conn.execute(text(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"'))
        await conn.execute(text(f'GRANT SELECT ON ALL TABLES IN SCHEMA "{schema}" TO "{role}"'))
        await conn.execute(text(f'SET ROLE "{role}"'))
    try:
        with pytest.raises(DatabaseError, match="must be owner of table training_jobs"):
            await db_engine._run_migrations()
    finally:
        async with migration_engine.begin() as conn:
            await conn.execute(text("RESET ROLE"))
            await conn.execute(text(f'DROP OWNED BY "{role}"'))
            await conn.execute(text(f'DROP ROLE "{role}"'))
