import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from core.feature_engineering import routes as fe_routes
    from core.feature_engineering.schemas import (
        PipelinePreviewMetrics,
        PipelinePreviewResponse,
        PipelinePreviewSignals,
        TrainModelDraftReadinessSnapshot,
    )
except ImportError:  # pragma: no cover - when running tests without package installation
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from core.feature_engineering import routes as fe_routes
    from core.feature_engineering.schemas import (
        PipelinePreviewMetrics,
        PipelinePreviewResponse,
        PipelinePreviewSignals,
        TrainModelDraftReadinessSnapshot,
    )


@pytest.mark.asyncio
async def test_preview_pipeline_full_dataset_signal(monkeypatch):
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

    async def fake_load_dataset_frame(
        session,
        dataset_source_id,
        *,
        sample_size,
        execution_mode,
        allow_empty_sample,
    ):
        assert dataset_source_id == "dataset-123"
        assert allow_empty_sample is False
        frame = pd.DataFrame({"feature": [1, 2], "value": [3.5, 4.2]})
        meta = {"total_rows": 2, "sample_size": 0}
        return frame, meta

    import core.feature_engineering.execution.preview as preview_module
    from core.feature_engineering.api import pipeline as pipeline_api

    monkeypatch.setattr(preview_module, "load_dataset_frame", fake_load_dataset_frame)

    def fake_run_pipeline_execution(
        frame,
        execution_order,
        node_map,
        *,
        pipeline_id=None,
        collect_signals=True,
        existing_signals=None,
        preserve_split_column=False,
    ):
        assert execution_order == ["dataset-source", "node-1"]
        signals = PipelinePreviewSignals() if collect_signals else None
        readiness = TrainModelDraftReadinessSnapshot(
            row_count=len(frame),
            feature_count=frame.shape[1],
            feature_columns=list(frame.columns),
        )
        return frame.copy(), ["Applied node-1"], signals, readiness

    monkeypatch.setattr(pipeline_api, "run_pipeline_execution", fake_run_pipeline_execution)

    captured_snapshot = {}

    def fake_build_data_snapshot_response(
        working_frame,
        *,
        target_node_id,
        preview_rows,
        preview_total_rows,
        initial_sample_rows,
        applied_steps,
        metrics_requested_sample_size,
        modeling_signals,
        signals,
        include_signals,
        **_kwargs,
    ):
        captured_snapshot.update(
            preview_rows=preview_rows,
            preview_total_rows=preview_total_rows,
            initial_sample_rows=initial_sample_rows,
            applied_steps=list(applied_steps),
            metrics_requested_sample_size=metrics_requested_sample_size,
            include_signals=include_signals,
            signals=signals,
        )
        return PipelinePreviewResponse(
            node_id=target_node_id,
            columns=list(working_frame.columns),
            sample_rows=working_frame.to_dict(orient="records"),
            metrics=PipelinePreviewMetrics(
                row_count=preview_total_rows,
                column_count=working_frame.shape[1],
                duplicate_rows=0,
                missing_cells=0,
                preview_rows=preview_rows,
                total_rows=preview_total_rows,
                requested_sample_size=metrics_requested_sample_size,
            ),
            applied_steps=list(applied_steps),
            modeling_signals=modeling_signals,
            signals=signals,
        )

    monkeypatch.setattr(pipeline_api, "build_data_snapshot_response", fake_build_data_snapshot_response)

    request_body = {
        "dataset_source_id": "dataset-123",
        "graph": {
            "nodes": [
                {
                    "id": "dataset-source",
                    "data": {"catalogType": "dataset", "label": "Dataset"},
                },
                {
                    "id": "node-1",
                    "data": {"catalogType": "transform", "label": "Transform"},
                },
            ],
            "edges": [
                {"source": "dataset-source", "target": "node-1"},
            ],
        },
        "target_node_id": "node-1",
        "sample_size": 0,
        "include_preview_rows": True,
        "include_signals": True,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/ml-workflow/api/pipelines/preview",
            json=request_body,
        )

    assert response.status_code == 200
    payload = response.json()

    assert payload["columns"] == ["feature", "value"]
    assert len(payload["sample_rows"]) == 2
    assert payload["metrics"]["preview_rows"] == 2
    assert payload["metrics"]["total_rows"] == 2
    assert payload["signals"]["full_execution"]["reason"] == "Preview executed against full dataset."
    assert captured_snapshot["include_signals"] is True
    assert captured_snapshot["preview_rows"] == 2
    assert captured_snapshot["metrics_requested_sample_size"] == 0
    assert captured_snapshot["applied_steps"] == ["Applied node-1"]

    app.dependency_overrides.clear()
