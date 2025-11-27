import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from core.feature_engineering import routes as fe_routes
    from core.feature_engineering.schemas import FeatureGraph, TrainingJobStatus
except ImportError:  # pragma: no cover - fallback for editable installs
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from core.feature_engineering import routes as fe_routes
    from core.feature_engineering.schemas import FeatureGraph, TrainingJobStatus


@dataclass
class StubTrainingJob:
    """Lightweight fake TrainingJob for route serialization."""

    id: str
    dataset_source_id: str
    pipeline_id: str
    node_id: str
    version: int
    model_type: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    job_metadata: Dict[str, Any] = field(default_factory=dict)
    graph: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {"nodes": [], "edges": []})
    status: TrainingJobStatus = TrainingJobStatus.QUEUED
    user_id: int | None = None
    metrics: Dict[str, Any] | None = None
    artifact_uri: str | None = None
    error_message: str | None = None
    created_at: Any | None = None
    updated_at: Any | None = None
    started_at: Any | None = None
    finished_at: Any | None = None


@pytest.fixture
def fastapi_app():
    app = FastAPI()
    app.include_router(fe_routes.router)

    class FakeSession:
        pass

    async def session_override():
        session = FakeSession()
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[fe_routes.get_async_session] = session_override
    try:
        yield app
    finally:
        app.dependency_overrides.clear()


def _build_training_request(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "dataset_source_id": "dataset-123",
        "pipeline_id": "pipeline-abc",
        "node_id": "node-1",
        "model_types": ["xgboost_classifier"],
        "hyperparameters": {"n_estimators": 25},
        "graph": FeatureGraph(nodes=[], edges=[]).model_dump(),
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_enqueue_training_job_creates_jobs_and_dispatches(monkeypatch, fastapi_app):
    app = fastapi_app

    created_jobs: List[StubTrainingJob] = []
    dispatch_calls: List[str] = []

    from core.feature_engineering.api import training as training_api

    async def fake_create_training_job_record(session, payload, *, user_id, model_type_override):
        assert payload.model_types == [model_type_override]
        job = StubTrainingJob(
            id=f"job-{model_type_override}",
            dataset_source_id=payload.dataset_source_id,
            pipeline_id=payload.pipeline_id,
            node_id=payload.node_id,
            version=len(created_jobs) + 1,
            model_type=model_type_override,
            hyperparameters=payload.hyperparameters or {},
            job_metadata=payload.metadata or {},
            graph=payload.graph.model_dump(),
        )
        created_jobs.append(job)
        return job

    def fake_dispatch_training_job(job_id: str) -> None:
        dispatch_calls.append(job_id)

    monkeypatch.setattr(training_api, "create_training_job_record", fake_create_training_job_record)
    monkeypatch.setattr(training_api, "dispatch_training_job", fake_dispatch_training_job)

    request_payload = _build_training_request(model_types=["random_forest", "lightgbm"])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/ml-workflow/api/training-jobs", json=request_payload)

    assert response.status_code == 202
    payload = response.json()

    assert payload["total"] == 2
    job_ids = [job["id"] for job in payload["jobs"]]
    assert job_ids == ["job-random_forest", "job-lightgbm"]
    assert dispatch_calls == job_ids
    assert [job["model_type"] for job in payload["jobs"]] == ["random_forest", "lightgbm"]


@pytest.mark.asyncio
async def test_enqueue_training_job_skips_dispatch_when_disabled(monkeypatch, fastapi_app):
    app = fastapi_app

    from core.feature_engineering.api import training as training_api

    async def fake_create_training_job_record(session, payload, *, user_id, model_type_override):
        return StubTrainingJob(
            id="job-only",
            dataset_source_id=payload.dataset_source_id,
            pipeline_id=payload.pipeline_id,
            node_id=payload.node_id,
            version=1,
            model_type=model_type_override,
            graph=payload.graph.model_dump(),
        )

    dispatch_calls: List[str] = []

    def fake_dispatch_training_job(job_id: str) -> None:
        dispatch_calls.append(job_id)

    monkeypatch.setattr(training_api, "create_training_job_record", fake_create_training_job_record)
    monkeypatch.setattr(training_api, "dispatch_training_job", fake_dispatch_training_job)

    request_payload = _build_training_request(run_training=False)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/ml-workflow/api/training-jobs", json=request_payload)

    assert response.status_code == 202
    payload = response.json()

    assert payload["total"] == 1
    assert not dispatch_calls


@pytest.mark.asyncio
async def test_enqueue_training_job_updates_status_on_dispatch_failure(monkeypatch, fastapi_app):
    app = fastapi_app

    from core.feature_engineering.api import training as training_api

    job = StubTrainingJob(
        id="job-dispatch",
        dataset_source_id="dataset-123",
        pipeline_id="pipeline-abc",
        node_id="node-1",
        version=1,
        model_type="random_forest",
        graph={"nodes": [], "edges": []},
    )

    async def fake_create_training_job_record(session, payload, *, user_id, model_type_override):
        return job

    def fake_dispatch_training_job(job_id: str) -> None:
        raise RuntimeError("celery down")

    update_calls: List[Dict[str, Any]] = []

    async def fake_update_job_status(
        session,
        target_job,
        *,
        status,
        metrics=None,
        artifact_uri=None,
        error_message=None,
        metadata=None,
    ):
        update_calls.append({
            "job_id": target_job.id,
            "status": status,
            "error_message": error_message,
        })
        return target_job

    monkeypatch.setattr(training_api, "create_training_job_record", fake_create_training_job_record)
    monkeypatch.setattr(training_api, "dispatch_training_job", fake_dispatch_training_job)
    monkeypatch.setattr(training_api, "update_job_status", fake_update_job_status)

    request_payload = _build_training_request()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/ml-workflow/api/training-jobs", json=request_payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Failed to enqueue training job"
    assert update_calls == [{
        "job_id": "job-dispatch",
        "status": TrainingJobStatus.FAILED,
        "error_message": "Failed to enqueue training job",
    }]
