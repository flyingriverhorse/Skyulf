"""Exported projects keep rows removed by fitted outlier filters, like Databricks scoring."""

import importlib.util
import io
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.deployment.service import DeploymentService
from backend.ml_pipeline.project_export.service import build_project

FEATURES = ["a", "b", "c"]
# Each filter with parameters and one raw row it must exclude at prediction time.
FILTERS = {
    "IQR": ({"columns": FEATURES}, {"a": 1e6}, "outlier"),
    "ZScore": ({"columns": FEATURES}, {"a": 1e6}, "outlier"),
    "EllipticEnvelope": ({"columns": FEATURES}, {"a": 1e6}, "outlier"),
    "IsolationForest": ({"columns": FEATURES}, {"a": 1e6, "b": -1e6}, "outlier"),
    "LocalOutlierFactor": ({"columns": FEATURES}, {"a": 1e6, "b": -1e6}, "outlier"),
    "ManualBounds": ({"bounds": {"c": {"lower": -3.0, "upper": 3.0}}}, {"c": 50.0}, "outlier"),
    "PowerTransformer": ({"columns": ["b"], "method": "box-cox"}, {"b": -5.0}, "invalid"),
}


def _frame(rows=240):
    """Separable data with strictly positive b so box-cox can be fitted."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "a": rng.normal(10, 2, rows),
            "b": rng.uniform(1, 50, rows),
            "c": rng.normal(0, 1, rows),
        }
    )
    frame["label"] = np.where(frame["b"] + rng.normal(0, 5, rows) > 25, "yes", "no")
    return frame


def _train(tmp_path, filter_type, params, model="logistic_regression"):
    """Fit loader -> split -> filter -> model through the real engine."""
    source = tmp_path / "train.csv"
    frame = _frame()
    if model != "logistic_regression":
        frame["label"] = frame["a"] * 2 + frame["c"]
    frame.to_csv(source, index=False)
    graph = [
        NodeConfig("load", "data_loader", {"path": str(source), "_display_name": "Dataset"}),
        NodeConfig("split", "TrainTestSplitter", {"target_column": "label"}, ["load"]),
        NodeConfig("filter", filter_type, {**params, "_display_name": "Outliers"}, ["split"]),
        NodeConfig(
            "train",
            "training",
            {"model_type": model, "target_column": "label", "cv_enabled": False},
            ["filter"],
        ),
    ]
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    result = PipelineEngine(store, FileSystemCatalog(str(tmp_path))).run(
        PipelineConfig("exclusions", graph), job_id="exclusion-job"
    )
    assert result.status == "success", result
    return store.load("exclusion-job")


def _export(tmp_path, bundle, model_type="classification"):
    """Build the real ZIP and import its scorer and batch entry point."""
    job = SimpleNamespace(
        id="exclusion-job",
        version=1,
        model_type=model_type,
        tuned_thresholds_enabled=False,
        tuned_thresholds=None,
    )
    content, _ = build_project(bundle, job, "both")
    project = tmp_path / "project"
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        archive.extractall(project)
    sys.path.insert(0, str(project))
    try:
        modules = {}
        for name in ("scoring", "batch"):
            spec = importlib.util.spec_from_file_location(name, project / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        sys.path.remove(str(project))
    return modules["scoring"].Predictor(project), modules["batch"], project


def _requests(outlier):
    """Two ordinary rows around one row that the fitted filter rejects."""
    rows = _frame(rows=3)[FEATURES].reset_index(drop=True)
    rows.loc[1, list(outlier)] = list(outlier.values())
    return rows


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("filter_type", sorted(FILTERS))
def test_excluded_rows_keep_position_and_reason(tmp_path, monkeypatch, engine, filter_type):
    """A filtered row must stay in place with a null prediction, never vanish or shift rows."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", engine)
    params, outlier, prefix = FILTERS[filter_type]
    bundle = _train(tmp_path, filter_type, params)
    predictor, _, _ = _export(tmp_path, bundle)
    rows = _requests(outlier)

    result = predictor.predict(rows.copy())

    kept = rows.drop(index=1).reset_index(drop=True)
    expected, _ = DeploymentService._predict_with_bundled_artifact(bundle, kept.copy())
    assert result["predictions"] == [expected[0], None, expected[1]]
    assert result["scoring_status"] == ["predicted", "excluded", "predicted"]
    assert result["exclusion_reason"][0] is None and result["exclusion_reason"][2] is None
    assert result["exclusion_reason"][1] == f"{prefix}:{filter_type}"
    assert result["probabilities"][1] is None
    assert set(result["probabilities"][0]) == {"no", "yes"}


def test_batch_writes_typed_columns_when_first_chunk_is_excluded(tmp_path):
    """An all-excluded first chunk must not fix a null column type for later chunks."""
    params, outlier, _ = FILTERS["IQR"]
    bundle = _train(tmp_path, "IQR", params)
    predictor, batch, project = _export(tmp_path, bundle)
    rows = _requests(outlier).iloc[[1, 0, 2]].reset_index(drop=True)
    expected = predictor.predict(rows.copy())
    for suffix in (".csv", ".parquet"):
        source = tmp_path / f"input{suffix}"
        destination = tmp_path / f"output{suffix}"
        if suffix == ".csv":
            rows.to_csv(source, index=False)
        else:
            rows.to_parquet(source, index=False)

        batch.score_file(source, destination, chunk_size=1)

        output = pd.read_csv(destination) if suffix == ".csv" else pd.read_parquet(destination)
        assert output["scoring_status"].tolist() == ["excluded", "predicted", "predicted"]
        assert output["prediction"].isna().tolist() == [True, False, False]
        assert output["prediction"].iloc[1:].tolist() == expected["predictions"][1:]
        assert output["exclusion_reason"].iloc[0].startswith("outlier:")
        assert output["prediction_proba_yes"].isna().tolist() == [True, False, False]
    assert (project / "batch.py").exists()


def test_regression_batch_keeps_float_predictions_with_nulls(tmp_path):
    """Regression outputs stay numeric when a filtered row has no prediction."""
    params, outlier, _ = FILTERS["ZScore"]
    bundle = _train(tmp_path, "ZScore", params, model="linear_regression")
    predictor, batch, _ = _export(tmp_path, bundle, model_type="regression")
    rows = _requests(outlier)
    source, destination = tmp_path / "input.parquet", tmp_path / "output.parquet"
    rows.to_parquet(source, index=False)

    batch.score_file(source, destination, chunk_size=2)

    output = pd.read_parquet(destination)
    expected, _ = DeploymentService._predict_with_bundled_artifact(
        bundle, rows.drop(index=1).reset_index(drop=True)
    )
    assert output["prediction"].dtype == "float64"
    assert output["prediction"].isna().tolist() == [False, True, False]
    np.testing.assert_allclose(output["prediction"].iloc[[0, 2]], expected)
    assert "probabilities" not in predictor.predict(rows.copy())


@pytest.mark.parametrize("step_type", ["ColumnFunction", "FittedFunction", "RowFilterFunction"])
def test_project_code_steps_are_refused_even_with_a_row_promise(step_type):
    """A saved module reference would fail with ImportError inside the standalone project."""
    from backend.ml_pipeline.project_export.service import _prediction_step

    step = {"name": "custom", "type": step_type, "applier": None, "artifact": {}}
    with pytest.raises(ValueError, match="project code"):
        _prediction_step(step)


def test_projects_without_filters_keep_their_response_shape(tmp_path):
    """Exclusion columns appear only when the exported pipeline can exclude rows."""
    bundle = _train(tmp_path, "StandardScaler", {"columns": FEATURES})
    predictor, _, _ = _export(tmp_path, bundle)

    result = predictor.predict(_requests({"a": 1e6}).copy())

    assert "scoring_status" not in result and "exclusion_reason" not in result
    assert None not in result["predictions"]
