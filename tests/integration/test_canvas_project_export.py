"""Real canvas parameters must survive training, artifact reload and project export."""

import importlib.util
import io
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


def _canvas_nodes(source):
    """Mirror the serialized defaults from the failing browser canvas graph."""
    steps = [
        ("load", "data_loader", {"path": str(source), "_display_name": "Dataset"}),
        (
            "split",
            "TrainTestSplitter",
            {
                "test_size": 0.2,
                "validation_size": 0,
                "random_state": 42,
                "stratify": False,
                "shuffle": True,
                "target_column": "churn",
                "_display_name": "Train-Test Split",
            },
        ),
        (
            "mean",
            "SimpleImputer",
            {
                "columns": ["age", "income"],
                "strategy": "mean",
                "fill_value": 0,
                "_display_name": "Imputation",
            },
        ),
        (
            "mode",
            "SimpleImputer",
            {
                "columns": ["city"],
                "strategy": "most_frequent",
                "fill_value": 0,
                "_display_name": "Imputation",
            },
        ),
        (
            "dates",
            "FeatureMath",
            {
                "operations": [
                    {
                        "operation_type": "datetime_extract",
                        "method": "year",
                        "input_columns": ["signup_date"],
                        "output_column": "",
                        "datetime_features": ["year", "month"],
                    }
                ],
                "_display_name": "Feature Generation",
            },
        ),
        (
            "encode",
            "OneHotEncoder",
            {
                "columns": ["city", "plan"],
                "drop_first": False,
                "drop_original": True,
                "handle_unknown": "ignore",
                "max_categories": 20,
                "include_missing": False,
                "_display_name": "Encoding",
            },
        ),
        (
            "drop",
            "DropMissingColumns",
            {"columns": ["signup_date"], "missing_threshold": 0, "_display_name": "Drop Columns"},
        ),
        (
            "scale",
            "StandardScaler",
            {
                "catalogType": "scale_numeric_features",
                "method": "standard",
                "columns": ["age", "income", "monthly_visits", "support_tickets"],
                "_display_name": "Scaling",
            },
        ),
        (
            "train",
            "training",
            {
                "run_mode": "fixed",
                "target_column": "churn",
                "model_type": "random_forest_classifier",
                "hyperparameters": {"n_estimators": 12, "random_state": 42},
                "cv_enabled": False,
                "_display_name": "Classification",
            },
        ),
    ]
    return [
        NodeConfig(name, kind, params, [steps[index - 1][0]] if index else [])
        for index, (name, kind, params) in enumerate(steps)
    ]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_canvas_pipeline_exports_saved_raw_input_prediction_parity(tmp_path, monkeypatch, engine):
    """The genuine canvas labels and hidden defaults must produce a usable fitted ZIP."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", engine)
    index = np.arange(80)
    frame = pd.DataFrame(
        {
            "age": np.where(index % 13 == 0, np.nan, 20 + index % 40),
            "income": np.where(index % 17 == 0, np.nan, 30000 + index * 1000),
            "city": np.where(index % 11 == 0, None, np.where(index % 2 == 0, "London", "Paris")),
            "plan": np.where(index % 3 == 0, "premium", "basic"),
            "signup_date": pd.date_range("2023-01-01", periods=80).strftime("%Y-%m-%d"),
            "monthly_visits": index % 20,
            "support_tickets": index % 4,
            "churn": np.where(index % 3 == 0, "yes", "no"),
        }
    )
    source = tmp_path / "canvas.csv"
    frame.to_csv(source, index=False)
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    result = PipelineEngine(store, FileSystemCatalog(str(tmp_path))).run(
        PipelineConfig("canvas-export", _canvas_nodes(source)), job_id="canvas-job"
    )
    assert result.status == "success", result
    bundle = store.load("canvas-job")
    job = SimpleNamespace(
        id="canvas-job",
        version=1,
        model_type="classification",
        tuned_thresholds_enabled=False,
        tuned_thresholds=None,
    )
    content, filename = build_project(bundle, job, "both")
    project = tmp_path / "project"
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        archive.extractall(project)
    spec = importlib.util.spec_from_file_location("canvas_scoring", project / "scoring.py")
    assert spec and spec.loader
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    rows = frame.drop(columns="churn").iloc[:20].copy()
    rows.loc[rows.index[-1], "city"] = "unseen"
    expected, _ = DeploymentService._predict_with_bundled_artifact(bundle, rows)
    actual = scoring.Predictor(project).predict(rows)["predictions"]
    assert filename == "kayra_model_canvas-job.zip"
    assert len(actual) == len(rows)
    assert set(actual) == {"yes", "no"}
    assert actual == expected
