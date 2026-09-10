"""Raw uploads must be compared with the saved source, not transformed split data."""

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
from fastapi import UploadFile
from starlette.requests import Request

from backend.data.catalog import FileSystemCatalog
from backend.exceptions.core import SkyulfException
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.monitoring.router import calculate_drift


def _graph():
    """Represent two preprocessing branches that share one source before training."""
    return {
        "nodes": [
            {"node_id": "source", "step_type": "data_loader", "params": {}, "inputs": []},
            {
                "node_id": "log",
                "step_type": "GeneralTransformation",
                "params": {"transformations": [{"column": "length", "method": "log"}]},
                "inputs": ["source"],
            },
            {
                "node_id": "drop",
                "step_type": "DropMissingColumns",
                "params": {"columns": ["Id"], "missing_threshold": 0},
                "inputs": ["source"],
            },
            {
                "node_id": "split",
                "step_type": "TrainTestSplitter",
                "params": {"target_column": "Species", "test_size": 0.2, "random_state": 42},
                "inputs": ["drop", "log"],
            },
            {
                "node_id": "model",
                "step_type": "training",
                "params": {
                    "target_column": "Species",
                    "algorithm": "logistic_regression",
                    "evaluate": False,
                },
                "inputs": ["split"],
            },
        ]
    }


def _raw():
    """Use a full source population whose logarithms have a disjoint distribution."""
    return pd.DataFrame(
        {
            "Id": range(150),
            "length": np.tile([4.0, 5.0, 6.0, 7.0, 8.0], 30),
            "width": np.tile([2.0, 3.0, 4.0], 50),
            "Species": [0, 1] * 75,
        }
    )


async def _report(store, graph, current, recorded=None):
    """Exercise real artifact loading, upload parsing and metrics with DB writes isolated."""
    job = SimpleNamespace(graph=graph, node_id="model")
    with (
        patch("backend.monitoring.router.ArtifactFactory") as factory,
        patch(
            "backend.monitoring.router._fetch_drift_job_rows",
            new=AsyncMock(return_value={"job-1": job}),
        ),
        patch(
            "backend.monitoring.router._find_deployment_context",
            new=AsyncMock(return_value=(None, None)),
        ),
        patch(
            "backend.monitoring.router._get_or_create_threshold_version",
            new=AsyncMock(return_value=SimpleNamespace(version=1)),
        ),
        patch(
            "backend.monitoring.router._save_drift_alert",
            new=recorded or AsyncMock(return_value=None),
        ),
        patch(
            "backend.monitoring.router._load_feature_importances", new=AsyncMock(return_value=None)
        ),
    ):
        factory.get_discovery.return_value.get_store_for_job.return_value = store
        return await calculate_drift(
            request=Request({"type": "http", "path": "/monitoring/drift/calculate"}),
            job_id="job-1",
            dataset_name="ds",
            file=UploadFile(
                file=io.BytesIO(current.to_csv(index=False).encode()), filename="same.csv"
            ),
            threshold_psi=None,
            threshold_ks=None,
            threshold_wasserstein=None,
            threshold_kl=None,
            db=AsyncMock(),
        )


@pytest.fixture
def saved_job(tmp_path):
    """Keep a legacy transformed reference beside the original saved loader snapshot."""
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    raw = _raw()
    transformed = raw.drop(columns="Id").iloc[:120].copy()
    transformed["length"] = np.log1p(transformed["length"])
    store.save("source", raw)
    store.save("reference_data_ds_job-1", transformed)
    return store, _graph(), raw


async def test_training_with_pre_split_log_accepts_the_same_raw_upload(tmp_path, monkeypatch):
    """The production training-to-drift path must not report the preprocessing itself as drift."""
    from backend.config import get_settings

    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas")
    raw = _raw()
    source = tmp_path / "source.csv"
    raw.to_csv(source, index=False)
    graph = _graph()
    graph["nodes"][0]["params"] = {"source": "csv", "path": str(source)}
    config = PipelineConfig(
        pipeline_id="drift-regression", nodes=[NodeConfig(**node) for node in graph["nodes"]]
    )
    store = LocalArtifactStore(str(tmp_path / "trained"))
    engine = PipelineEngine(store, catalog=FileSystemCatalog())
    training = engine.run(config, job_id="job-1", dataset_name="ds")
    assert training.status == "success", training
    reference_before = store.load("reference_data_ds_job-1")

    report = await _report(store, graph, raw)

    assert report.reference_rows == report.current_rows == 150
    assert set(report.column_drifts) == {"length", "width"}
    assert report.drifted_columns_count == 0
    assert report.new_columns == report.missing_columns == []
    assert all(
        metric["value"] == 0
        for column in report.column_drifts.values()
        for metric in column["metrics"]
        if metric["metric"] == "psi"
    )
    pd.testing.assert_frame_equal(store.load("reference_data_ds_job-1"), reference_before)
    pd.testing.assert_frame_equal(store.load("source"), raw)


@pytest.mark.parametrize("source_engine", ["pandas", "polars"])
async def test_legacy_saved_job_uses_raw_source_without_retraining(saved_job, source_engine):
    """Saved loader snapshots must repair existing jobs independently of their frame engine."""
    store, graph, raw = saved_job
    if source_engine == "polars":
        store.save("source", pl.from_pandas(raw))
    report = await _report(store, graph, raw)
    assert report.reference_rows == 150
    assert report.drifted_columns_count == 0
    assert set(report.column_drifts) == {"length", "width"}


async def test_raw_source_still_detects_distribution_and_schema_changes(saved_job):
    """Correcting the comparison space must retain real shifted, missing and new inputs."""
    store, graph, raw = saved_job
    current = raw.drop(columns="width").assign(length=raw["length"] + 100, extra=1)
    report = await _report(store, graph, current)
    assert report.column_drifts["length"]["drift_detected"] is True
    assert report.missing_columns == ["width"]
    assert report.new_columns == ["extra"]
    assert report.drifted_columns_count == 3


async def test_encoded_raw_categories_remain_monitored(saved_job):
    """Model-space feature names must not hide categorical inputs that were encoded."""
    store, graph, raw = saved_job
    raw["color"] = ["red", "blue"] * 75
    store.save("source", raw)
    store.save("reference_data_ds_job-1", pd.DataFrame({"color_red": [0, 1] * 60}))
    report = await _report(store, graph, raw.assign(color="blue"))
    assert report.column_drifts["color"]["drift_detected"] is True
    assert report.missing_columns == report.new_columns == []
    assert set(report.column_drifts) == {"length", "width", "color"}


async def test_only_selected_model_ancestors_choose_source_and_drops(saved_job):
    """Unrelated loaders and drop nodes must not change the selected model's reference."""
    store, graph, raw = saved_job
    graph["nodes"].insert(
        0, {"node_id": "other", "step_type": "data_loader", "params": {}, "inputs": []}
    )
    graph["nodes"].append(
        {
            "node_id": "other-drop",
            "step_type": "DropMissingColumns",
            "params": {"columns": ["length"]},
            "inputs": ["other"],
        }
    )
    store.save("other", raw.assign(length=1000))
    report = await _report(store, graph, raw)
    assert report.drifted_columns_count == 0
    assert "length" in report.column_drifts


@pytest.mark.parametrize("failure", ["missing", "multiple", "broken_graph"])
async def test_unavailable_or_ambiguous_source_records_failure(saved_job, failure):
    """An untrusted source must fail explicitly instead of publishing invented drift metrics."""
    store, graph, raw = saved_job
    if failure == "missing":
        graph["nodes"][0]["node_id"] = "unavailable"
        graph["nodes"][1]["inputs"] = ["unavailable"]
        graph["nodes"][2]["inputs"] = ["unavailable"]
    elif failure == "multiple":
        graph["nodes"].append(
            {"node_id": "other", "step_type": "data_loader", "params": {}, "inputs": []}
        )
        graph["nodes"][3]["inputs"].append("other")
        store.save("other", raw)
    else:
        graph["nodes"][1]["inputs"] = ["unknown-node"]
    recorded = AsyncMock(return_value=None)
    with pytest.raises(SkyulfException):
        await _report(store, graph, raw, recorded)
    outcomes = [call.kwargs["evaluation_status"] for call in recorded.await_args_list]
    assert outcomes == ["failed"]


async def test_graphless_legacy_reference_remains_usable(saved_job):
    """Legacy manually saved references without graph metadata keep their existing contract."""
    store, _graph_data, raw = saved_job
    store.save("reference_data_ds_job-1", raw)
    report = await _report(store, {}, raw)
    assert report.reference_rows == 150
    assert report.drifted_columns_count == 0
