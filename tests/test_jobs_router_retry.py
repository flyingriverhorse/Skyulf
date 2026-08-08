"""HTTP-level tests for `POST /api/pipeline/jobs/{job_id}/retry` (OPS-001).

Covers: retry-supported (failed/cancelled training/tuning jobs with a stored
graph) succeeds and creates a fresh job; retry-unavailable outcomes (missing
job, wrong job type, non-terminal status, missing graph) return 400 with an
explanatory detail rather than silently no-oping.
"""

from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database.engine import get_async_session
from backend.database.models import Base
from backend.main import app

BASE = "/api/pipeline"
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

_STORED_GRAPH = (
    '{"pipeline_id": "pipe-1", "nodes": ['
    '{"node_id": "node-1", "step_type": "training", '
    '"params": {"algorithm": "random_forest"}, "inputs": ["loader"]}], '
    '"metadata": {}}'
)


async def _insert_job(
    session: AsyncSession,
    job_id: str,
    *,
    status: str = "failed",
    graph: str | None = _STORED_GRAPH,
) -> None:
    """Inserts a minimal `training_jobs` row via raw SQL (ORM defaults don't apply to raw INSERT)."""
    await session.execute(
        text(
            """
        INSERT INTO training_jobs (id, pipeline_id, node_id, dataset_source_id, user_id, status, run_mode, version, model_type, graph, artifact_uri, error_message, progress, current_step, started_at, finished_at, created_at, updated_at)
        VALUES (:id, :pipeline_id, :node_id, :ds_id, :user_id, :status, :run_mode, :version, :model_type, :graph, :artifact_uri, :error_message, :progress, :current_step, :started_at, :finished_at, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        ),
        {
            "id": job_id,
            "pipeline_id": "pipe-1",
            "node_id": "node-1",
            "ds_id": "ds-1",
            "user_id": None,
            "status": status,
            "run_mode": "fixed",
            "version": 1,
            "model_type": "random_forest",
            "graph": graph,
            "artifact_uri": job_id,
            "error_message": "boom" if status == "failed" else None,
            "progress": 100,
            "current_step": None,
            "started_at": None,
            "finished_at": None,
        },
    )
    await session.commit()


@pytest_asyncio.fixture
async def async_session():
    """Provides an in-memory async SQLite session with all tables created."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session_maker() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def client(async_session):
    """A TestClient wired to override `get_async_session` with the in-memory test session."""

    async def _override_get_async_session():
        yield async_session

    app.dependency_overrides[get_async_session] = _override_get_async_session
    with TestClient(app, base_url="http://testserver") as c:
        yield c
    app.dependency_overrides.pop(get_async_session, None)


def test_retry_returns_404_for_missing_job(client):
    """POST .../retry returns 404 when the job doesn't exist."""
    response = client.post(f"{BASE}/jobs/nonexistent/retry")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_retry_returns_400_for_non_terminal_job(async_session, client):
    """POST .../retry returns 400 for a still-running job (nothing to recover from yet)."""
    await _insert_job(async_session, "job-1", status="running")

    response = client.post(f"{BASE}/jobs/job-1/retry")
    assert response.status_code == 400
    assert "failed or cancelled" in response.json()["message"].lower()


@pytest.mark.asyncio
async def test_retry_returns_400_when_graph_missing(async_session, client):
    """POST .../retry returns 400 with an explanatory detail when no graph snapshot was stored."""
    await _insert_job(async_session, "job-1", status="failed", graph="{}")

    response = client.post(f"{BASE}/jobs/job-1/retry")
    assert response.status_code == 400
    assert "graph" in response.json()["message"].lower()


@pytest.mark.asyncio
async def test_retry_succeeds_for_failed_training_job(async_session, client):
    """POST .../retry on a failed training job with a stored graph creates a fresh job id."""
    await _insert_job(async_session, "job-1", status="failed")

    with patch(
        "backend.ml_pipeline._internal._routers.run_pipeline.run_pipeline_task"
    ) as mock_run_task:
        response = client.post(f"{BASE}/jobs/job-1/retry")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] != "job-1"
    assert "retry" in body["message"].lower()
    # BackgroundTasks defers execution until after the response is sent by
    # TestClient's context manager, so the task runs before this assertion.
    mock_run_task.assert_called_once()
    new_job_id, payload = mock_run_task.call_args[0]
    assert new_job_id == body["job_id"]
    assert payload["nodes"][0]["node_id"] == "node-1"


@pytest.mark.asyncio
async def test_retry_succeeds_for_cancelled_tuning_job(async_session, client):
    """POST .../retry also supports cancelled tuning jobs, not only failed training jobs."""
    graph = _STORED_GRAPH.replace('"training"', '"training"')
    await _insert_job(async_session, "job-tune", status="cancelled", graph=graph)

    with patch("backend.ml_pipeline._internal._routers.run_pipeline.run_pipeline_task"):
        response = client.post(f"{BASE}/jobs/job-tune/retry")

    assert response.status_code == 200
    assert response.json()["job_id"] != "job-tune"


@pytest.mark.asyncio
async def test_concurrent_retries_create_only_one_job(async_session):
    """Two near-simultaneous retries of the same failed job must dedupe to one new job.

    Exercises `resubmit_job_from_graph` directly (rather than through
    `TestClient`, which serialises requests) so the two calls genuinely race
    through the same submit-lock/dedupe path `_submit_or_dedupe_branch_job`
    already provides for normal `/run` submissions.
    """
    import asyncio

    from backend.ml_pipeline._execution.jobs import JobManager
    from backend.ml_pipeline._internal._routers.run_pipeline import resubmit_job_from_graph

    await _insert_job(async_session, "job-race", status="failed")
    job = await JobManager.get_job(async_session, "job-race")
    assert job is not None

    background_tasks_a = BackgroundTasks()
    background_tasks_b = BackgroundTasks()

    with patch("backend.ml_pipeline._internal._routers.run_pipeline.run_pipeline_task"):
        job_id_a, job_id_b = await asyncio.gather(
            resubmit_job_from_graph(async_session, job, background_tasks_a),
            resubmit_job_from_graph(async_session, job, background_tasks_b),
        )

    assert job_id_a == job_id_b

    result = await async_session.execute(
        text("SELECT COUNT(*) FROM training_jobs WHERE node_id = 'node-1' AND id != 'job-race'")
    )
    assert result.scalar_one() == 1
