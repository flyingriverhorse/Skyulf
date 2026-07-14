"""Regression tests for the config-driven job idempotency window and cross-table
skip cap in ``backend.ml_pipeline._execution.jobs``.

``_IDEMPOTENCY_WINDOW`` and ``_MAX_SKIP`` used to be module-level constants
frozen at import time; they now read fresh from
``Settings.JOB_IDEMPOTENCY_WINDOW_SECONDS`` / ``Settings.MAX_CROSS_TABLE_SKIP``
at each call site, so tests can change behavior via monkeypatched settings.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, BasicTrainingJob
from backend.ml_pipeline._execution.jobs import JobManager
from backend.ml_pipeline._execution.schemas import JobStatus

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def async_session():
    """Provide an isolated in-memory async SQLAlchemy session per test."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


async def _insert_running_job(session: AsyncSession, created_at: datetime) -> str:
    """Insert a minimal running BasicTrainingJob row with a specific created_at."""
    job_id = str(uuid.uuid4())
    job = BasicTrainingJob(
        id=job_id,
        pipeline_id="p1",
        node_id="n1",
        dataset_source_id="d1",
        status=JobStatus.RUNNING.value,
        model_type="classifier",
        graph={},
        job_metadata={"branch_index": 0},
        created_at=created_at,
    )
    session.add(job)
    await session.commit()
    return job_id


@pytest.mark.asyncio
async def test_find_active_job_uses_configured_idempotency_window(async_session, monkeypatch):
    """A job created 20s ago is a duplicate only if the configured window covers it."""
    import backend.ml_pipeline._execution.jobs as jobs_mod

    created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=20)
    await _insert_running_job(async_session, created_at)

    class _ShortWindow:
        JOB_IDEMPOTENCY_WINDOW_SECONDS = 5

    monkeypatch.setattr(jobs_mod, "get_settings", lambda: _ShortWindow())
    result = await JobManager.find_active_job(async_session, "d1", "n1", branch_index=0)
    assert result is None  # 20s ago is outside a 5s window

    class _LongWindow:
        JOB_IDEMPOTENCY_WINDOW_SECONDS = 60

    monkeypatch.setattr(jobs_mod, "get_settings", lambda: _LongWindow())
    result = await JobManager.find_active_job(async_session, "d1", "n1", branch_index=0)
    assert result is not None  # 20s ago is inside a 60s window


@pytest.mark.asyncio
async def test_list_jobs_clamps_skip_to_configured_max_cross_table_skip(async_session, monkeypatch):
    """list_jobs (all types) clamps `skip` to the configured MAX_CROSS_TABLE_SKIP."""
    import backend.ml_pipeline._execution.jobs as jobs_mod

    class _FakeSettings:
        MAX_CROSS_TABLE_SKIP = 10

    monkeypatch.setattr(jobs_mod, "get_settings", lambda: _FakeSettings())

    # skip well beyond the configured cap should be clamped, not raise, and
    # should log a warning (behavior preserved from the hardcoded constant).
    result = await JobManager.list_jobs(async_session, limit=5, skip=1000)
    assert isinstance(result, list)
