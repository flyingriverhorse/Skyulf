"""Count every error in the rolling window, including the current partial hour."""

import importlib
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema

from backend.database.models import ErrorEvent
from backend.dependencies import get_db

monitoring = importlib.import_module("backend.monitoring.router")


@pytest.mark.asyncio
@pytest.mark.parametrize("minute", [0, 30])
@pytest.mark.parametrize("database", ["sqlite", "postgres"])
async def test_timeline_includes_partial_hours_and_excludes_outside_window(
    monkeypatch, minute, database
):
    """The real HTTP response must preserve cutoff/current events without future leakage."""
    now = datetime(2026, 9, 20, 2, minute, tzinfo=UTC)
    cutoff = now - timedelta(hours=3)

    class Clock(datetime):
        """Freeze the server clock at an hour boundary or halfway through it."""

        @classmethod
        def now(cls, tz=None):
            """Return the controlled instant in the requested timezone."""
            return now.astimezone(tz)

    monkeypatch.setattr(monitoring, "datetime", Clock)
    admin = None
    schema = f"timeline_{uuid4().hex}"
    if database == "postgres":
        url = os.environ.get("SKYULF_TEST_POSTGRES_URL")
        if not url:
            pytest.skip("Set SKYULF_TEST_POSTGRES_URL to an isolated PostgreSQL test DB")
        admin = create_async_engine(url)
        async with admin.begin() as conn:
            await conn.execute(CreateSchema(schema))
        engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    else:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(ErrorEvent.__table__.create)
        async with AsyncSession(engine) as session:
            for stamp in [
                cutoff - timedelta(microseconds=1),
                cutoff,
                now - timedelta(minutes=15),
                now,
                now + timedelta(microseconds=1),
            ]:
                session.add(
                    ErrorEvent(
                        created_at=stamp.replace(tzinfo=None),
                        error_type="TestError",
                        message="timeline",
                    )
                )
            await session.commit()

        async def get_session():
            """Read through a fresh session on the isolated database."""
            async with AsyncSession(engine) as session:
                yield session

        app = FastAPI()
        app.include_router(monitoring.router)
        app.dependency_overrides[get_db] = get_session
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/monitoring/errors/timeline", params={"hours": 3})
        assert response.status_code == 200
        rows = response.json()
        assert len(rows) == 4
        assert rows[0] == {"hour": "2026-09-19T23:00", "count": 1}
        assert rows[-1] == {"hour": "2026-09-20T02:00", "count": 2 if minute else 1}
        assert sum(row["count"] for row in rows) == 3
    finally:
        await engine.dispose()
        if admin is not None:
            async with admin.begin() as conn:
                await conn.execute(DropSchema(schema, cascade=True))
            await admin.dispose()
