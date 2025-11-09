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
    from core.feature_engineering.schemas import FeatureGraph, HyperparameterTuningJobStatus
except ImportError:  # pragma: no cover - allow running without installation
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from core.feature_engineering import routes as fe_routes
    from core.feature_engineering.schemas import FeatureGraph, HyperparameterTuningJobStatus


@dataclass
class StubTuningJob:
    id: str
    dataset_source_id: str
    pipeline_id: str
    node_id: str
    run_number: int
    model_type: str
    search_strategy: str
    status: HyperparameterTuningJobStatus = HyperparameterTuningJobStatus.QUEUED
    user_id: int | None = None
    search_space: Dict[str, Any] = field(default_factory=dict)
    baseline_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    n_iterations: int | None = None
    scoring: str | None = None
    random_state: int | None = None
    cross_validation: Dict[str, Any] = field(default_factory=dict)
    job_metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] | None = None
    results: List[Dict[str, Any]] | None = None
    best_params: Dict[str, Any] | None = None
    best_score: float | None = None
    artifact_uri: str | None = None
    error_message: str | None = None
    created_at: Any | None = None
    updated_at: Any | None = None
    started_at: Any | None = None
    finished_at: Any | None = None
    graph: FeatureGraph = field(default_factory=lambda: FeatureGraph(nodes=[], edges=[]))


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


def _build_tuning_request(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "dataset_source_id": "dataset-123",
        "pipeline_id": "pipeline-abc",
        "node_id": "node-1",
        "model_types": ["xgboost_classifier"],
        "search_space": {"n_estimators": [10, 20]},
        "graph": FeatureGraph(nodes=[], edges=[]).model_dump(),
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_enqueue_tuning_jobs_dispatches(monkeypatch, fastapi_app):
    created_jobs: List[StubTuningJob] = []
    dispatch_calls: List[str] = []

    async def fake_create_job(session, payload, *, user_id, model_type_override):
        job = StubTuningJob(
            id=f"tuning-{model_type_override}",
            dataset_source_id=payload.dataset_source_id,
            pipeline_id=payload.pipeline_id,
            node_id=payload.node_id,
            run_number=len(created_jobs) + 1,
            model_type=model_type_override,
            search_strategy=payload.search_strategy,
            search_space=payload.search_space,
            n_iterations=payload.n_iterations,
            baseline_hyperparameters=payload.baseline_hyperparameters or {},
        )
        created_jobs.append(job)
        return job

    def fake_dispatch(job_id: str) -> None:
        dispatch_calls.append(job_id)

    monkeypatch.setattr(fe_routes, "create_hyperparameter_tuning_job_record", fake_create_job)
    monkeypatch.setattr(fe_routes, "dispatch_hyperparameter_tuning_job", fake_dispatch)

    request_payload = _build_tuning_request(model_types=["random_forest", "lightgbm"])

    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/ml-workflow/api/hyperparameter-tuning-jobs", json=request_payload)

    assert response.status_code == 202
    payload = response.json()
    assert payload["total"] == 2
    ids = [item["id"] for item in payload["jobs"]]
    assert ids == ["tuning-random_forest", "tuning-lightgbm"]
    assert dispatch_calls == ids


@pytest.mark.asyncio
async def test_enqueue_tuning_jobs_skips_dispatch_when_disabled(monkeypatch, fastapi_app):
    async def fake_create_job(session, payload, *, user_id, model_type_override):
        return StubTuningJob(
            id="tuning-only",
            dataset_source_id=payload.dataset_source_id,
            pipeline_id=payload.pipeline_id,
            node_id=payload.node_id,
            run_number=1,
            model_type=model_type_override,
            search_strategy=payload.search_strategy,
        )

    dispatch_calls: List[str] = []

    def fake_dispatch(job_id: str) -> None:
        dispatch_calls.append(job_id)

    monkeypatch.setattr(fe_routes, "create_hyperparameter_tuning_job_record", fake_create_job)
    monkeypatch.setattr(fe_routes, "dispatch_hyperparameter_tuning_job", fake_dispatch)

    request_payload = _build_tuning_request(run_tuning=False)

    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/ml-workflow/api/hyperparameter-tuning-jobs", json=request_payload)

    assert response.status_code == 202
    payload = response.json()
    assert payload["total"] == 1
    assert dispatch_calls == []


@pytest.mark.asyncio
async def test_enqueue_tuning_jobs_handles_dispatch_failure(monkeypatch, fastapi_app):
    job = StubTuningJob(
        id="tuning-dispatch",
        dataset_source_id="dataset-123",
        pipeline_id="pipeline-abc",
        node_id="node-1",
        run_number=1,
        model_type="random_forest",
        search_strategy="grid",
    )

    async def fake_create_job(session, payload, *, user_id, model_type_override):
        return job

    def fake_dispatch(job_id: str) -> None:
        raise RuntimeError("celery unavailable")

    update_calls: List[Dict[str, Any]] = []

    async def fake_update(
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

    monkeypatch.setattr(fe_routes, "create_hyperparameter_tuning_job_record", fake_create_job)
    monkeypatch.setattr(fe_routes, "dispatch_hyperparameter_tuning_job", fake_dispatch)
    monkeypatch.setattr(fe_routes, "update_tuning_job_status", fake_update)

    request_payload = _build_tuning_request()

    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/ml-workflow/api/hyperparameter-tuning-jobs", json=request_payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Failed to enqueue tuning job"
    assert update_calls == [{
        "job_id": "tuning-dispatch",
        "status": HyperparameterTuningJobStatus.FAILED,
        "error_message": "Failed to enqueue tuning job",
    }]
