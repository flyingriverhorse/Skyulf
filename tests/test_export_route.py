import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from core.feature_engineering import routes as fe_routes
    from core.feature_engineering.schemas import (
        PipelinePreviewSignals,
        TrainModelDraftReadinessSnapshot,
    )
except ImportError:  # pragma: no cover - fallback when tests run without package installed
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from core.feature_engineering import routes as fe_routes
    from core.feature_engineering.schemas import (
        PipelinePreviewSignals,
        TrainModelDraftReadinessSnapshot,
    )


@pytest.mark.asyncio
async def test_export_training_job_bundle(tmp_path, monkeypatch):
    app = FastAPI()
    app.include_router(fe_routes.router)

    session_container = {}

    class FakeSession:
        def __init__(self) -> None:
            self.committed = False
            self.refreshed = False

        async def commit(self) -> None:
            self.committed = True

        async def refresh(self, obj) -> None:  # pragma: no cover - simple flag mutation
            self.refreshed = True

        def add(self, obj) -> None:
            pass

    async def session_override():
        session = FakeSession()
        session_container["session"] = session
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[fe_routes.get_async_session] = session_override

    class FakeJob:
        def __init__(self) -> None:
            self.id = "job123"
            self.dataset_source_id = "dataset_abc"
            self.pipeline_id = "pipe_001"
            self.version = 2
            self.artifact_uri = str(tmp_path / "model.joblib")
            (tmp_path / "model.joblib").write_text("stub", encoding="utf-8")
            self.graph = {
                "nodes": [
                    {
                        "id": "dataset-source",
                        "data": {"catalogType": "dataset", "label": "Dataset"},
                    }
                ],
                "edges": [],
            }
            self.job_metadata = {"existing": True}

    job = FakeJob()

    async def fake_fetch_training_job(session, job_id):
        assert job_id == job.id
        return job

    from core.feature_engineering.api import training as training_api
    monkeypatch.setattr(training_api, "fetch_training_job", fake_fetch_training_job)

    async def fake_load_dataset_frame(
        session,
        dataset_source_id,
        *,
        sample_size,
        execution_mode,
        allow_empty_sample,
    ):
        assert dataset_source_id == job.dataset_source_id
        frame = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
        preview_meta = {"total_rows": 2, "sample_size": 2}
        return frame, preview_meta

    monkeypatch.setattr(training_api, "load_dataset_frame", fake_load_dataset_frame)

    def fake_collect_pipeline_signals(
        frame,
        execution_order,
        node_map,
        *,
        pipeline_id,
        existing_signals=None,
        preserve_split_column,
    ):
        assert pipeline_id == job.pipeline_id
        modeling_snapshot = TrainModelDraftReadinessSnapshot(
            row_count=len(frame),
            feature_count=frame.shape[1],
            feature_columns=list(frame.columns),
            warnings=["check data"],
        )
        signals = PipelinePreviewSignals()
        applied_steps = ["applied node"]
        return frame.copy(), signals, modeling_snapshot, applied_steps

    monkeypatch.setattr(training_api, "collect_pipeline_signals", fake_collect_pipeline_signals)

    captured_export = {}

    def fake_export_project_bundle(*, artifact_path, output_directory, job_id, pipeline_id, job_metadata):
        captured_export.update(
            artifact_path=str(artifact_path),
            output_directory=str(output_directory),
            job_id=job_id,
            pipeline_id=pipeline_id,
            job_metadata=job_metadata,
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = output_directory / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            manifest_payload={"scaffold_root": str(output_directory)},
            artefact_entries=[
                {
                    "index": 0,
                    "name": "pipeline/pipeline_overview",
                    "filename": "pipeline_overview.json",
                    "content_type": "application/json",
                    "generator": "pipeline",
                }
            ],
            output_directory=output_directory,
            manifest_path=manifest_path,
        )

    monkeypatch.setattr(training_api, "export_project_bundle", fake_export_project_bundle)

    class SettingsStub:
        PIPELINE_EXPORT_DIR = str(tmp_path / "exports")

    # monkeypatch.setattr(fe_routes, "get_settings", lambda: SettingsStub())

    request_body = {
        "sample_size": 16,
        "project_name": "Demo Project",
        "project_description": "Demo description",
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            f"/ml-workflow/api/training-jobs/{job.id}/export",
            json=request_body,
        )

    assert response.status_code == 201
    payload = response.json()

    assert payload["project_name"] == "Demo Project"
    assert payload["project_slug"] == "demo-project"
    assert payload["sample_size"] == 16
    assert (
        payload["manifest"]["scaffold_root"]
        == captured_export["output_directory"]
    )
    assert "check data" in payload["warnings"]
    assert payload["artefacts"][0]["name"] == "pipeline/pipeline_overview"

    assert captured_export["pipeline_id"] == job.pipeline_id
    assert captured_export["job_metadata"]["project_name"] == "Demo Project"
    assert "pipeline_signals" in captured_export["job_metadata"]
    assert "last_export" in job.job_metadata

    stored_session = session_container["session"]
    assert stored_session.committed
    assert stored_session.refreshed

    app.dependency_overrides.clear()
