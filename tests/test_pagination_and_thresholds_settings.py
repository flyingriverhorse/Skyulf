"""Regression tests for config-centralization of scattered pagination/threshold
magic numbers (069 cleanup):

- Various ``limit`` defaults across ml_pipeline/data_ingestion/database
  modules now fall back to ``Settings.DEFAULT_PAGE_SIZE`` (50) instead of
  hardcoded 20/50/100 values when the caller passes ``None``.
- Data-ingestion sample-preview ``limit`` defaults now fall back to
  ``Settings.DEFAULT_SAMPLE_ROWS`` (5) instead of a hardcoded ``5``.
- ``AsyncJSONSafeSerializer`` DataFrame/Series/records "yield to event loop"
  row-count threshold now reads ``Settings.SERIALIZATION_YIELD_THRESHOLD_ROWS``
  instead of a hardcoded ``1000``.
- ``FilterRequest.column`` length validation in ``backend.eda.router`` now
  reads ``Settings.MAX_COLUMN_NAME_LENGTH`` instead of a hardcoded ``255``.
- Parallel-branch ``ThreadPoolExecutor`` pools in ``run_pipeline.py`` and
  ``tasks.py`` now cap concurrency at ``Settings.MAX_PARALLEL_BRANCH_WORKERS``
  instead of being unbounded (``max_workers=len(branches)``).
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.data_ingestion.serialization import AsyncJSONSafeSerializer
from backend.database.models import (
    AdvancedTuningJob,
    Base,
    BasicTrainingJob,
    DataSource,
    Deployment,
)
from backend.database.repository import BaseRepository
from backend.eda.router import FilterRequest
from backend.ml_pipeline._execution.advanced_tuning_manager import AdvancedTuningManager
from backend.ml_pipeline._execution.basic_training_manager import BasicTrainingManager
from backend.ml_pipeline._execution.schemas import JobStatus
from backend.ml_pipeline.deployment.service import DeploymentService
from backend.ml_pipeline.model_registry.service import ModelRegistryService

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


def _make_training_job(idx: int) -> BasicTrainingJob:
    return BasicTrainingJob(
        id=str(uuid.uuid4()),
        pipeline_id=f"p{idx}",
        node_id=f"n{idx}",
        dataset_source_id="d1",
        status="completed",
        model_type="classifier",
        graph={},
        version=1,
    )


def _make_tuning_job(idx: int, model_type: str = "classifier") -> AdvancedTuningJob:
    return AdvancedTuningJob(
        id=str(uuid.uuid4()),
        pipeline_id=f"p{idx}",
        node_id=f"n{idx}",
        dataset_source_id="d1",
        status=JobStatus.COMPLETED.value,
        model_type=model_type,
        graph={},
        run_number=1,
        search_strategy="random",
        finished_at=datetime.now(UTC).replace(tzinfo=None),
    )


def _make_deployment(idx: int) -> Deployment:
    return Deployment(
        job_id=f"job-{idx}",
        model_type="classifier",
        artifact_uri=f"uri-{idx}",
        is_active=False,
    )


# ── Part A: pagination defaults ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_training_jobs_uses_default_page_size(async_session):
    """No explicit limit should fall back to Settings.DEFAULT_PAGE_SIZE (50)."""
    for i in range(3):
        async_session.add(_make_training_job(i))
    await async_session.commit()

    class _SmallPage:
        DEFAULT_PAGE_SIZE = 2

    with patch(
        "backend.ml_pipeline._execution.basic_training_manager.get_settings",
        lambda: _SmallPage(),
    ):
        jobs = await BasicTrainingManager.list_training_jobs(async_session, limit=None)
    assert len(jobs) == 2


@pytest.mark.asyncio
async def test_get_tuning_jobs_for_model_uses_default_page_size(async_session):
    """No explicit limit should fall back to Settings.DEFAULT_PAGE_SIZE (50)."""
    for i in range(3):
        async_session.add(_make_tuning_job(i))
    await async_session.commit()

    class _SmallPage:
        DEFAULT_PAGE_SIZE = 2

    with patch(
        "backend.ml_pipeline._execution.advanced_tuning_manager.get_settings",
        lambda: _SmallPage(),
    ):
        jobs = await AdvancedTuningManager.get_tuning_jobs_for_model(
            async_session, "classifier", limit=None
        )
    assert len(jobs) == 2


@pytest.mark.asyncio
async def test_list_deployments_uses_default_page_size(async_session):
    """No explicit limit should fall back to Settings.DEFAULT_PAGE_SIZE (50)."""
    for i in range(3):
        async_session.add(_make_deployment(i))
    await async_session.commit()

    class _SmallPage:
        DEFAULT_PAGE_SIZE = 2

    with patch("backend.ml_pipeline.deployment.service.get_settings", lambda: _SmallPage()):
        deployments = await DeploymentService.list_deployments(async_session, limit=None)
    assert len(deployments) == 2


@pytest.mark.asyncio
async def test_model_registry_list_models_uses_default_page_size(async_session):
    """No explicit limit should fall back to Settings.DEFAULT_PAGE_SIZE (50)."""
    for i in range(3):
        async_session.add(_make_training_job(i))
    await async_session.commit()

    class _SmallPage:
        DEFAULT_PAGE_SIZE = 1

    with patch("backend.ml_pipeline.model_registry.service.get_settings", lambda: _SmallPage()):
        results = await ModelRegistryService.list_models(async_session, limit=None)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_repository_get_multi_uses_default_page_size(async_session):
    """No explicit limit should fall back to Settings.DEFAULT_PAGE_SIZE (50)."""
    for i in range(3):
        async_session.add(_make_deployment(i))
    await async_session.commit()

    repo = BaseRepository(async_session, Deployment)

    class _SmallPage:
        DEFAULT_PAGE_SIZE = 2

    with patch("backend.database.repository.get_settings", lambda: _SmallPage()):
        records = await repo.get_multi(limit=None)
    assert len(records) == 2


@pytest.mark.asyncio
async def test_data_ingestion_list_sources_uses_default_page_size(async_session):
    """No explicit limit should fall back to Settings.DEFAULT_PAGE_SIZE (50)."""
    from backend.data_ingestion.service import DataIngestionService

    for i in range(3):
        async_session.add(DataSource(name=f"src-{i}", type="file", config={}, is_active=True))
    await async_session.commit()

    service = DataIngestionService(async_session, upload_dir="uploads/data_test")

    class _SmallPage:
        DEFAULT_PAGE_SIZE = 2

    with patch("backend.data_ingestion.service.get_settings", lambda: _SmallPage()):
        sources = await service.list_sources(limit=None)
    assert len(sources) == 2


# ── Part A: sample preview default ───────────────────────────────────────


@pytest.mark.asyncio
async def test_data_ingestion_get_sample_uses_default_sample_rows(async_session):
    """No explicit limit should fall back to Settings.DEFAULT_SAMPLE_ROWS (5)."""
    from backend.data_ingestion.service import DataIngestionService

    source = DataSource(
        name="src",
        type="file",
        config={"file_path": "/tmp/does-not-matter.csv"},
        is_active=True,
    )
    async_session.add(source)
    await async_session.commit()

    service = DataIngestionService(async_session, upload_dir="uploads/data_test")

    captured: dict = {}

    class _FakeDataFrame:
        def to_dicts(self):
            return [{"a": 1}]

    class _FakeConnector:
        def __init__(self, *_args, **_kwargs):
            pass

        async def connect(self):
            return None

        async def fetch_data(self, limit):
            captured["limit"] = limit
            return _FakeDataFrame()

    class _CustomSampleRows:
        DEFAULT_SAMPLE_ROWS = 7

    with (
        patch("backend.data_ingestion.service.get_settings", lambda: _CustomSampleRows()),
        patch(
            "backend.data_ingestion.connectors.s3.S3Connector",
            _FakeConnector,
        ),
    ):
        # Force the S3 code path so we can capture the effective limit
        # without touching the local filesystem connector.
        source.config = {"file_path": "s3://bucket/key.csv"}
        await async_session.commit()
        await service.get_sample(source.id, limit=None)

    assert captured["limit"] == 7


# ── Part B: serialization yield threshold ────────────────────────────────


@pytest.mark.asyncio
async def test_dataframe_serialization_yields_at_configured_threshold():
    """DataFrame serialization should only yield control when the row count
    exceeds the configured SERIALIZATION_YIELD_THRESHOLD_ROWS, not a hardcoded 1000.
    """
    df = pd.DataFrame({"a": range(5)})

    class _LowThreshold:
        SERIALIZATION_YIELD_THRESHOLD_ROWS = 3

    with (
        patch("backend.data_ingestion.serialization.get_settings", lambda: _LowThreshold()),
        patch("backend.data_ingestion.serialization.asyncio.sleep") as mock_sleep,
    ):
        await AsyncJSONSafeSerializer._handle_dataframe(df)
    mock_sleep.assert_awaited_once_with(0)


@pytest.mark.asyncio
async def test_dataframe_serialization_does_not_yield_below_configured_threshold():
    """Below the configured threshold, no yield/sleep should occur."""
    df = pd.DataFrame({"a": range(2)})

    class _HighThreshold:
        SERIALIZATION_YIELD_THRESHOLD_ROWS = 1000

    with (
        patch("backend.data_ingestion.serialization.get_settings", lambda: _HighThreshold()),
        patch("backend.data_ingestion.serialization.asyncio.sleep") as mock_sleep,
    ):
        await AsyncJSONSafeSerializer._handle_dataframe(df)
    mock_sleep.assert_not_called()


# ── Part C: EDA column name length ───────────────────────────────────────


def test_filter_request_column_length_uses_configured_setting():
    """Column names longer than Settings.MAX_COLUMN_NAME_LENGTH should be rejected."""

    class _ShortColumnLimit:
        MAX_COLUMN_NAME_LENGTH = 5

    with patch("backend.eda.router.get_settings", lambda: _ShortColumnLimit()):
        with pytest.raises(ValueError, match="column name too long"):
            FilterRequest(column="too_long_name", operator="==", value=1)

        # A name within the (shortened) limit should still pass.
        FilterRequest(column="ok", operator="==", value=1)


def test_filter_request_column_length_default_matches_255():
    """Default settings should still allow up to 255 characters (unchanged behavior)."""
    FilterRequest(column="a" * 255, operator="==", value=1)
    with pytest.raises(ValueError, match="column name too long"):
        FilterRequest(column="a" * 256, operator="==", value=1)


# ── Part D: parallel branch worker cap ───────────────────────────────────


def test_tasks_run_pipeline_batch_caps_thread_pool_workers(monkeypatch):
    """ThreadPoolExecutor max_workers should be capped at
    Settings.MAX_PARALLEL_BRANCH_WORKERS, while every branch is still processed
    (bounded concurrency, not bounded work).
    """
    import backend.ml_pipeline.tasks as tasks_mod

    class _SmallCap:
        MAX_PARALLEL_BRANCH_WORKERS = 2

    monkeypatch.setattr(tasks_mod, "get_settings", lambda: _SmallCap())
    monkeypatch.setattr(tasks_mod, "get_db_session", lambda: _FakeSession())
    monkeypatch.setattr(tasks_mod, "execute_pipeline", lambda job_id, cfg, session: None)

    captured: dict = {}
    real_executor = tasks_mod.ThreadPoolExecutor

    class _SpyExecutor(real_executor):
        def __init__(self, max_workers=None, *args, **kwargs):
            captured["max_workers"] = max_workers
            super().__init__(*args, max_workers=max_workers, **kwargs)

    monkeypatch.setattr(tasks_mod, "ThreadPoolExecutor", _SpyExecutor)

    branches = [(f"job-{i}", {}) for i in range(5)]
    tasks_mod.run_pipeline_batch_task(branches)

    assert captured["max_workers"] == 2  # capped, not len(branches) == 5


class _FakeSession:
    """Minimal stand-in for a sync SQLAlchemy session used in tasks.py tests."""

    def close(self):
        return None
