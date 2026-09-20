"""Exercise naive UTC monitoring timestamps through PostgreSQL's asyncpg driver."""

import asyncio
import importlib
import os
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema

from backend.database.models import ErrorEvent
from backend.monitoring import tasks

monitoring = importlib.import_module("backend.monitoring.router")
NOW = datetime(2026, 9, 20, 2, 30, tzinfo=UTC)


class FrozenClock(datetime):
    """Keep retention and timeline bounds deterministic across timezone changes."""

    @classmethod
    def now(cls, tz=None):
        """Honor the production clock's requested timezone at a fixed instant."""
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


@pytest.fixture(params=["UTC", "Asia/Kathmandu"])
def postgres_sessions(request):
    """Use an opt-in PostgreSQL URL and an isolated disposable schema per test."""
    url = os.environ.get("SKYULF_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("Set SKYULF_TEST_POSTGRES_URL to an isolated PostgreSQL+asyncpg test DB")
    if not url.startswith("postgresql+asyncpg://"):
        pytest.fail("SKYULF_TEST_POSTGRES_URL must use PostgreSQL+asyncpg")
    schema = f"qw72_{uuid4().hex}"
    admin = create_async_engine(url, poolclass=NullPool)
    engine = create_async_engine(
        url,
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": schema, "timezone": request.param}},
    )

    async def setup():
        """Create only the owned schema and production error-event table."""
        async with admin.begin() as connection:
            await connection.execute(CreateSchema(schema))
        async with engine.begin() as connection:
            await connection.run_sync(ErrorEvent.__table__.create)

    async def teardown():
        """Remove only the generated schema, including its test rows."""
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(DropSchema(schema, cascade=True))
        await admin.dispose()

    try:
        asyncio.run(setup())
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        asyncio.run(teardown())


async def insert_events(sessions, timestamps):
    """Persist naive UTC timestamps through the real mapped column and driver."""
    async with sessions() as session:
        session.add_all(
            ErrorEvent(created_at=value, error_type="TestError", message=str(index))
            for index, value in enumerate(timestamps)
        )
        await session.commit()


def test_timeline_filters_naive_utc_across_midnight(postgres_sessions, monkeypatch):
    """Session timezone must not shift naive UTC events or break the timeline query."""
    monkeypatch.setattr(monitoring, "datetime", FrozenClock)
    cutoff = NOW.replace(tzinfo=None) - timedelta(hours=3)
    asyncio.run(
        insert_events(
            postgres_sessions,
            [cutoff - timedelta(microseconds=1), cutoff, cutoff + timedelta(hours=1)],
        )
    )

    async def timeline():
        """Call the actual route with a PostgreSQL-backed session."""
        async with postgres_sessions() as session:
            return await monitoring.get_error_timeline(hours=3, db=session)

    result = asyncio.run(timeline())
    assert [(entry.hour, entry.count) for entry in result] == [
        ("2026-09-19T23:00", 1),
        ("2026-09-20T00:00", 1),
        ("2026-09-20T01:00", 0),
    ]


def test_retention_deletes_only_strictly_older_events(postgres_sessions, monkeypatch):
    """The Celery task must delete old records and preserve the exact cutoff and newer rows."""
    monkeypatch.setattr(tasks, "datetime", FrozenClock)
    monkeypatch.setattr(tasks, "get_settings", lambda: SimpleNamespace(ERROR_LOG_RETENTION_DAYS=30))
    monkeypatch.setattr(tasks, "async_session_factory", postgres_sessions)
    cutoff = NOW.replace(tzinfo=None) - timedelta(days=30)
    expected = [cutoff, cutoff + timedelta(microseconds=1)]
    asyncio.run(insert_events(postgres_sessions, [cutoff - timedelta(microseconds=1), *expected]))

    result = tasks.cleanup_error_events.run()

    async def remaining():
        """Read persisted rows through a fresh connection after the task commits."""
        async with postgres_sessions() as session:
            return list(
                await session.scalars(select(ErrorEvent.created_at).order_by(ErrorEvent.created_at))
            )

    assert result == {"deleted": 1, "retention_days": 30}
    assert asyncio.run(remaining()) == expected


def test_offset_filter_normalizes_before_postgres_comparison(postgres_sessions):
    """An offset crossing UTC midnight must select the correct naive-column boundary."""
    cutoff = datetime(2026, 9, 19, 23, 30)
    asyncio.run(insert_events(postgres_sessions, [cutoff - timedelta(microseconds=1), cutoff]))
    offset = datetime(2026, 9, 20, 5, 15, tzinfo=timezone(timedelta(hours=5, minutes=45)))

    async def matching():
        """Execute the shared normalization helper's bound using asyncpg."""
        async with postgres_sessions() as session:
            return list(
                await session.scalars(
                    select(ErrorEvent.created_at).where(
                        ErrorEvent.created_at
                        >= monitoring._normalize_since_for_naive_column(offset)
                    )
                )
            )

    assert asyncio.run(matching()) == [cutoff]
