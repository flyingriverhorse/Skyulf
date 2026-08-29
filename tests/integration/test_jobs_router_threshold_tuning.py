"""HTTP-level tests for the threshold-tuning endpoints on the jobs router.

Covers `POST /api/pipeline/jobs/{job_id}/thresholds/preview`,
`POST .../thresholds/save`, `POST .../thresholds/toggle`, and
`DELETE .../thresholds`.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.database.engine import get_async_session
from backend.database.models import Base
from backend.main import app

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

BASE = "/api/pipeline"


def _fake_evaluation_data() -> dict:
    """Builds a raw (undecoded) 3-class evaluation payload matching EvaluationService's shape."""
    return {
        "job_id": "job-1",
        "problem_type": "classification",
        "splits": {
            "validation": {
                "y_true": [0, 1, 2, 2, 1],
                "y_pred": [0, 1, 2, 2, 0],
                "y_proba": {
                    "classes": ["0", "1", "2"],
                    "values": [
                        [0.5, 0.3, 0.2],
                        [0.2, 0.6, 0.2],
                        [0.34, 0.33, 0.33],
                        [0.1, 0.1, 0.8],
                        [0.4, 0.4, 0.2],
                    ],
                },
            },
            "test": None,
        },
    }


async def _insert_job(session: AsyncSession, job_id: str) -> None:
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
            "status": "completed",
            "run_mode": "fixed",
            "version": 1,
            "model_type": "random_forest",
            "graph": "{}",
            "artifact_uri": job_id,
            "error_message": None,
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


def test_preview_endpoint_returns_400_for_missing_job(client):
    """POST .../thresholds/preview returns 400 when the job doesn't exist."""
    response = client.post(f"{BASE}/jobs/nonexistent/thresholds/preview", json={"metric": "f1"})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_preview_endpoint_returns_400_for_unsupported_metric(async_session, client):
    """POST .../thresholds/preview returns 400 for an unsupported metric."""
    await _insert_job(async_session, "job-1")

    response = client.post(f"{BASE}/jobs/job-1/thresholds/preview", json={"metric": "not_a_metric"})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_toggle_endpoint_returns_400_when_no_saved_thresholds(async_session, client):
    """POST .../thresholds/toggle returns 400 when the job has no saved thresholds yet."""
    await _insert_job(async_session, "job-1")

    response = client.post(f"{BASE}/jobs/job-1/thresholds/toggle", json={"enabled": True})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_full_threshold_tuning_http_flow(async_session, client):
    """preview -> save -> toggle -> clear works end-to-end over real HTTP requests."""
    await _insert_job(async_session, "job-1")

    with patch(
        "backend.ml_pipeline._services.threshold_tuning_service.EvaluationService"
        "._load_raw_evaluation_data",
        new=AsyncMock(return_value=(_fake_evaluation_data(), None)),
    ):
        preview = client.post(f"{BASE}/jobs/job-1/thresholds/preview", json={"metric": "f1"})
    assert preview.status_code == 200
    body = preview.json()
    assert set(body["classes"]) == {0, 1, 2}
    assert body["metric"] == "f1"
    assert body["split_used"] == "validation"

    save = client.post(
        f"{BASE}/jobs/job-1/thresholds/save",
        json={
            "thresholds": body["thresholds"],
            "classes": body["classes"],
            "metric": body["metric"],
            "split_used": body["split_used"],
        },
    )
    assert save.status_code == 200
    assert save.json() == {"status": "saved"}

    toggle = client.post(f"{BASE}/jobs/job-1/thresholds/toggle", json={"enabled": False})
    assert toggle.status_code == 200
    assert toggle.json() == {"status": "toggled", "enabled": False}

    clear = client.delete(f"{BASE}/jobs/job-1/thresholds")
    assert clear.status_code == 200
    assert clear.json() == {"status": "cleared"}


def test_get_endpoint_returns_400_for_missing_job(client):
    """GET .../thresholds returns 400 when the job doesn't exist."""
    response = client.get(f"{BASE}/jobs/nonexistent/thresholds")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_endpoint_returns_disabled_shell_before_any_save(async_session, client):
    """GET .../thresholds returns an all-null/disabled shell before anything is saved."""
    await _insert_job(async_session, "job-1")

    response = client.get(f"{BASE}/jobs/job-1/thresholds")
    assert response.status_code == 200
    assert response.json() == {
        "thresholds": None,
        "classes": None,
        "metric": None,
        "split_used": None,
        "computed_at": None,
        "source": None,
        "enabled": False,
    }


@pytest.mark.asyncio
async def test_get_endpoint_reflects_state_after_save_and_toggle(async_session, client):
    """GET .../thresholds reflects the saved thresholds and enabled flag after save/toggle."""
    await _insert_job(async_session, "job-1")

    with patch(
        "backend.ml_pipeline._services.threshold_tuning_service.EvaluationService"
        "._load_raw_evaluation_data",
        new=AsyncMock(return_value=(_fake_evaluation_data(), None)),
    ):
        preview = client.post(f"{BASE}/jobs/job-1/thresholds/preview", json={"metric": "f1"})
    body = preview.json()

    client.post(
        f"{BASE}/jobs/job-1/thresholds/save",
        json={
            "thresholds": body["thresholds"],
            "classes": body["classes"],
            "metric": body["metric"],
            "split_used": body["split_used"],
        },
    )

    get_response = client.get(f"{BASE}/jobs/job-1/thresholds")
    assert get_response.status_code == 200
    saved = get_response.json()
    assert saved["enabled"] is True
    assert saved["metric"] == "f1"
    assert saved["source"] is None
    assert set(saved["thresholds"].keys()) == {"0", "1", "2"}

    client.post(f"{BASE}/jobs/job-1/thresholds/toggle", json={"enabled": False})
    get_response = client.get(f"{BASE}/jobs/job-1/thresholds")
    assert get_response.json()["enabled"] is False


@pytest.mark.asyncio
async def test_get_endpoint_exposes_training_seed_source(async_session, client):
    """GET .../thresholds must surface the seeded 'training' source so the UI
    can label training-time thresholds (the response model used to strip it)."""
    await _insert_job(async_session, "job-1")
    seeded = json.dumps(
        {
            "thresholds": {"0": 0.6, "1": 0.4},
            "classes": ["0", "1"],
            "metric": "f1",
            "split_used": "validation",
            "computed_at": "2026-08-28T21:45:32+00:00",
            "source": "training",
        }
    )
    await async_session.execute(
        text(
            "UPDATE training_jobs SET tuned_thresholds = :t, tuned_thresholds_enabled = 1"
            " WHERE id = 'job-1'"
        ),
        {"t": seeded},
    )
    await async_session.commit()

    response = client.get(f"{BASE}/jobs/job-1/thresholds")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "training"
    assert body["enabled"] is True
    assert body["thresholds"] == {"0": 0.6, "1": 0.4}


def test_save_endpoint_returns_400_for_missing_job(client):
    """POST .../thresholds/save returns 400 when the job doesn't exist."""
    response = client.post(
        f"{BASE}/jobs/nonexistent/thresholds/save",
        json={"thresholds": {"0": 0.5}, "classes": [0], "metric": "f1", "split_used": "validation"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_save_endpoint_returns_400_for_invalid_payload(async_session, client):
    """POST .../thresholds/save returns 400 for a payload predict-time cannot honor (F-40)."""
    await _insert_job(async_session, "job-1")

    response = client.post(
        f"{BASE}/jobs/job-1/thresholds/save",
        json={
            "thresholds": {"0": 0.5},
            "classes": [0, 1],
            "metric": "f1",
            "split_used": "validation",
        },
    )
    assert response.status_code == 400


def test_clear_endpoint_returns_400_for_missing_job(client):
    """DELETE .../thresholds returns 400 when the job doesn't exist."""
    response = client.delete(f"{BASE}/jobs/nonexistent/thresholds")
    assert response.status_code == 400
