"""Test database migrations."""

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

import backend.database.engine as db_engine
from backend.database.engine import Base


@pytest.mark.asyncio
async def test_migrations_add_tuned_thresholds_columns():
    """Test that tuned_thresholds and tuned_thresholds_enabled columns exist after migrations."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    # Temporarily set the global async_engine for _run_migrations
    original_engine = db_engine.async_engine
    db_engine.async_engine = engine

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Run migrations using the global engine
        await db_engine._run_migrations()

        async with engine.begin() as conn:

            def _get_columns(sync_conn):
                return {col["name"] for col in inspect(sync_conn).get_columns("training_jobs")}

            columns = await conn.run_sync(_get_columns)

        assert "tuned_thresholds" in columns
        assert "tuned_thresholds_enabled" in columns
    finally:
        db_engine.async_engine = original_engine
        await engine.dispose()
