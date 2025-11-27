import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from core.feature_engineering import routes as fe_routes
    from core.feature_engineering.schemas import (
        FeatureGraph,
        ModelEvaluationSplitPayload,
    )
except ImportError:  # pragma: no cover - allow tests without package install
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from core.feature_engineering import routes as fe_routes
    from core.feature_engineering.schemas import (
        FeatureGraph,
        ModelEvaluationSplitPayload,
    )


@dataclass
class StubTrainingJob:
    id: str
    dataset_source_id: str
    pipeline_id: Optional[str]
    node_id: Optional[str]
    artifact_uri: str
    job_metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    graph: FeatureGraph = field(default_factory=lambda: FeatureGraph(nodes=[], edges=[]))


class FakeSession:
    def add(self, _obj: Any) -> None:
        pass

    async def commit(self) -> None:  # pragma: no cover - trivial
        pass

    async def refresh(self, _obj: Any) -> None:  # pragma: no cover - trivial
        pass


@pytest.fixture
def fastapi_app(tmp_path):
    app = FastAPI()
    app.include_router(fe_routes.router)

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


@pytest.mark.asyncio
async def test_evaluate_training_job_route(monkeypatch, fastapi_app, tmp_path):
    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_bytes(b"binary-placeholder")

    job = StubTrainingJob(
        id="job-123",
        dataset_source_id="dataset-1",
        pipeline_id="pipeline-1",
        node_id="node-1",
        artifact_uri=str(artifact_path),
        job_metadata={"resolved_problem_type": "classification"},
    )

    async def fake_fetch_training_job(_session, job_id):
        assert job_id == job.id
        return job

    async def fake_resolve_training_inputs(_session, _job):
        frame = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
        node_config = {"target_column": "target"}
        dataset_meta: Dict[str, Any] = {}
        upstream_order: List[str] = ["node-a", "node-b"]
        return frame, node_config, dataset_meta, upstream_order

    def fake_prepare_training_data(_frame, _target_column):
        features = pd.DataFrame({"feature": [1, 2, 3]})
        target = pd.Series([0, 1, 0])
        return (
            features,
            target,
            features,
            target,
            features,
            target,
            ["feature"],
            {"dtype": "categorical", "categories": [0, 1]},
        )

    def fake_joblib_load(_path):
        return {"model": object(), "problem_type": "classification"}

    def fake_build_classification_split_report(
        _model,
        *,
        split_name: str,
        features,
        target,
        label_names,
        include_confusion,
        include_curves,
        max_curve_points,
    ) -> ModelEvaluationSplitPayload:
        assert split_name == "test"
        return ModelEvaluationSplitPayload(
            split=split_name,
            row_count=len(target) if target is not None else 0,
            metrics={"accuracy": 0.75},
            notes=["evaluated"],
        )

    from core.feature_engineering.api import training as training_api

    monkeypatch.setattr(training_api, "fetch_training_job", fake_fetch_training_job)
    monkeypatch.setattr(training_api, "_resolve_training_inputs", fake_resolve_training_inputs)
    monkeypatch.setattr(training_api, "_prepare_training_data", fake_prepare_training_data)
    monkeypatch.setattr(training_api.joblib, "load", fake_joblib_load)
    monkeypatch.setattr(
        training_api,
        "build_classification_split_report",
        fake_build_classification_split_report,
    )

    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            f"/ml-workflow/api/training-jobs/{job.id}/evaluate",
            json={"splits": ["test"], "include_curves": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == job.id
    assert payload["problem_type"] == "classification"
    assert "test" in payload["splits"]
    assert payload["splits"]["test"]["metrics"] == {"accuracy": 0.75}

    # Ensure evaluation metadata persisted on the job stub.
    evaluation_metadata = job.metrics.get("evaluation")
    assert evaluation_metadata is not None
    assert evaluation_metadata["problem_type"] == "classification"
    assert evaluation_metadata["splits"]["test"]["split"] == "test"
