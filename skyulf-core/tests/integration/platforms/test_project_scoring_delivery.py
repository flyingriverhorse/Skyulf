"""Saved scoring policies must survive delivery without changing training evaluation."""

import json
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import (
    load_local_pipeline,
    predict_local_pipeline,
    save_local_pipeline,
)
from skyulf.integrations.databricks.project import load_project_workflow
from skyulf.pipeline import SkyulfPipeline

SOURCE = """
import pandas as pd

def build_preprocessing():
    return []

def eligible(frame, params):
    return pd.Series(["negative_input" if x < params["minimum"] else None for x in frame.x], index=frame.index, dtype="string")

def band(frame, predictions, params):
    return pd.DataFrame({"band": ["high" if x >= params["threshold"] else "low" for x in predictions.prediction]}, index=frame.index)

def build_scoring():
    return {
        "eligibility": [{"name": "nonnegative", "version": "1", "function": "eligible", "params": {"minimum": 0}}],
        "outputs": [{"name": "prediction_band", "version": "1", "function": "band", "params": {"threshold": 10}, "columns": [{"name": "band", "dtype": "string"}]}],
    }
"""


def fitted_scoring_artifact(tmp_path, engine="pandas"):
    """Save a real model with an editable project scoring hook."""
    source = tmp_path / "preprocessing.py"
    source.write_text(SOURCE, encoding="utf-8")
    workflow = load_project_workflow(
        {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}, source
    )
    data = pd.DataFrame({"x": np.arange(20, dtype=float), "target": 2 * np.arange(20, dtype=float)})
    if engine == "polars":
        data = pl.from_pandas(data)
    pipeline = SkyulfPipeline(workflow["pipeline"])
    pipeline.fit(SplitDataset(train=data[:16], test=data[16:]), target_column="target")
    destination = tmp_path / "artifact"
    save_local_pipeline(pipeline, destination)
    source.write_text("raise RuntimeError('edited project must not run')", encoding="utf-8")
    return load_local_pipeline(destination)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_scoring_preserves_membership_and_raw_evaluation(tmp_path, engine):
    """Scoring emits exclusions while heldout evaluation still measures the entire population."""
    from skyulf.inference.local_scoring import score_local_pipeline, scoring_output_schema

    artifact = fitted_scoring_artifact(tmp_path, engine)
    frame = pd.DataFrame({"x": [-1.0, 2.0, 8.0]}, index=[8, 3, 8])
    scored = score_local_pipeline(frame, artifact)
    assert scored.index.tolist() == [8, 3, 8]
    assert scored.scoring_status.tolist() == ["excluded", "predicted", "predicted"]
    assert scored.exclusion_reason.iloc[0] == "negative_input"
    assert pd.isna(scored.prediction.iloc[0])
    assert scored.band.iloc[1:].tolist() == ["low", "high"]
    np.testing.assert_allclose(scored.prediction.iloc[1:].astype(float), [4, 16])
    assert len(predict_local_pipeline(frame, artifact)) == 3
    assert tuple(scored.columns) == tuple(c.name for c in scoring_output_schema(artifact))


def test_all_excluded_does_not_invoke_model(tmp_path):
    """An excluded-only batch must remain publishable and never call the estimator."""
    from skyulf.inference.local_scoring import score_local_pipeline

    artifact = fitted_scoring_artifact(tmp_path)
    with patch.object(artifact.pipeline, "predict", side_effect=AssertionError("model called")):
        result = score_local_pipeline(pd.DataFrame({"x": [-2.0, -1.0]}), artifact)
    assert result.scoring_status.tolist() == ["excluded", "excluded"]
    assert result.prediction.isna().all()


def test_empty_scoring_keeps_schema(tmp_path):
    """An empty request has the saved output schema without evaluating callbacks."""
    from skyulf.inference.local_scoring import score_local_pipeline, scoring_output_schema

    artifact = fitted_scoring_artifact(tmp_path)
    result = score_local_pipeline(pd.DataFrame({"x": pd.Series([], dtype="float64")}), artifact)
    assert result.empty
    assert tuple(result.columns) == tuple(c.name for c in scoring_output_schema(artifact))


def test_mlflow_signature_and_pyfunc_apply_saved_rules(tmp_path):
    """MLflow advertises nullable policy outputs and runs exactly the saved policy."""
    pytest.importorskip("mlflow")
    from types import SimpleNamespace

    from skyulf.integrations.mlflow.local_model import SkyulfLocalPythonModel, _signature

    artifact = fitted_scoring_artifact(tmp_path)
    model = SkyulfLocalPythonModel()
    model.load_context(SimpleNamespace(artifacts={"local_pipeline": str(tmp_path / "artifact")}))
    result = model.predict(None, pd.DataFrame({"x": [-1.0, 8.0]}))
    assert result.scoring_status.tolist() == ["excluded", "predicted"]
    assert _signature(artifact).outputs.input_names() == list(result.columns)


def test_scoring_configuration_is_bound_to_artifact(tmp_path):
    """The recipe snapshot includes exact rule parameters and survives editable source changes."""
    artifact = fitted_scoring_artifact(tmp_path)
    policy = artifact.pipeline.config["project_scoring"]
    assert policy["eligibility"][0]["version"] == "1"
    assert policy["outputs"][0]["params"] == {"threshold": 10}
    assert json.loads((tmp_path / "artifact" / "manifest.json").read_text())[
        "project_source_sha256"
    ]


def test_preflight_probe_exercises_saved_output_rule(tmp_path):
    """A model-only probe must not hide an invalid declared scoring output."""
    from skyulf.inference.project_code import load_project_module
    from skyulf.integrations.databricks.local_sdk import (
        InputSource,
        LocalWorkflowConfig,
        ModelSelection,
        OutputSink,
        preflight_local,
    )

    artifact = fitted_scoring_artifact(tmp_path)
    module = load_project_module(artifact.pipeline.config["project_python_source"])
    original = module.band

    def invalid_output(frame, predictions, params):
        """Produce a deliberately undeclared column without changing row identity."""
        return pd.DataFrame({"wrong": ["x"] * len(frame)}, index=frame.index)

    invalid_output.__module__ = module.__name__
    module.__dict__["band"] = invalid_output
    try:
        config = LocalWorkflowConfig(
            runtime="local",
            engine="pandas",
            source=InputSource(kind="caller_frame"),
            model=ModelSelection(kind="local_pipeline", path=str(tmp_path / "artifact")),
            sink=OutputSink(kind="return_frame"),
        )
        result = preflight_local(config, artifact=artifact, probe_frame=pd.DataFrame({"x": [2.0]}))
        assert "prediction_probe_failed" in [issue.code for issue in result.issues]
    finally:
        module.__dict__["band"] = original


def test_all_excluded_keeps_temporal_session_state(tmp_path):
    """Skipping every estimate must not discard the incoming continuation tail."""
    from skyulf.inference.local_scoring import score_local_pipeline
    from skyulf.preprocessing.time_series.history import TemporalHistorySession

    artifact = fitted_scoring_artifact(tmp_path)
    params = {"history_mode": "carry", "history_id": "test", "history_seed": [{"t": 1, "x": 2.0}]}
    artifact.pipeline.feature_engineer.fitted_steps.append({"name": "unused", "artifact": params})
    previous = {"version": 1, "model_id": "model", "steps": {"test": [{"t": 2, "x": 4.0}]}}
    with TemporalHistorySession("model", previous) as history:
        scored = score_local_pipeline(pd.DataFrame({"x": [-1.0]}), artifact)
    assert history.state == previous
    assert scored.scoring_status.tolist() == ["excluded"]


def test_all_excluded_still_validates_saved_input_schema(tmp_path):
    """Eligibility cannot silently admit a different raw input type contract."""
    from skyulf.inference.local_scoring import score_local_pipeline

    artifact = fitted_scoring_artifact(tmp_path)
    artifact.pipeline.config["project_scoring"]["eligibility"][0]["params"]["minimum"] = "0"
    with pytest.raises(ValueError, match="dtype|type"):
        score_local_pipeline(pd.DataFrame({"x": ["-1"]}), artifact)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("model", ["logistic_regression", "voting_classifier", "voting_regressor"])
def test_scoring_policy_classification_and_ensembles(tmp_path, engine, model):
    """Probabilities and ensemble estimates retain exclusions without changing output meaning."""
    from skyulf.inference.local_scoring import score_local_pipeline
    from skyulf.inference.project_code import load_project_module

    source = load_project_module(SOURCE)
    x = np.arange(40, dtype=float)
    classification = model != "voting_regressor"
    rows = pd.DataFrame({"x": x, "target": (x % 2).astype(int) if classification else 2 * x})
    models = {
        "voting_classifier": {
            "base_estimators": ["logistic_regression", "decision_tree"],
            "voting": "soft",
        },
        "voting_regressor": {"base_estimators": ["linear_regression", "ridge"]},
    }
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {"type": model, "params": models.get(model, {})},
            "project_python_source": SOURCE,
            "project_scoring": source.build_scoring(),
        }
    )
    if engine == "polars":
        rows = pl.from_pandas(rows)
    pipeline.fit(SplitDataset(train=rows[:32], test=rows[32:]), target_column="target")
    save_local_pipeline(pipeline, tmp_path / "model")
    artifact = load_local_pipeline(tmp_path / "model")
    result = score_local_pipeline(pd.DataFrame({"x": [-1.0, 2.0, 8.0]}), artifact)
    assert result.scoring_status.tolist() == ["excluded", "predicted", "predicted"]
    assert pd.isna(result.prediction.iloc[0])
    if classification:
        assert result.probability_0.isna().tolist() == [True, False, False]
        np.testing.assert_allclose(
            (result.probability_0 + result.probability_1).iloc[1:].astype(float), 1.0
        )
    else:
        expected = predict_local_pipeline(pd.DataFrame({"x": [2.0, 8.0]}), artifact)
        np.testing.assert_allclose(result.prediction.iloc[1:].astype(float), expected.prediction)
    if model.startswith("voting_"):
        estimator = artifact.pipeline.model_estimator
        assert estimator is not None and estimator.model is not None
        assert len(estimator.model.estimators_) == 2


def test_template_scoring_rules_are_usable_from_saved_package():
    """The generated project supplies working reusable callbacks and an opt-in recipe."""
    from pathlib import Path

    from skyulf.inference.project_code import load_project_module
    from skyulf.inference.project_scoring import run_project_scoring
    from skyulf.integrations.databricks._project_files import project_source

    root = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/features"
    )
    source = project_source(root)
    module = load_project_module(source)
    assert module.build_scoring() == {"reuse_pre_split": True, "skip_target_steps": False}
    config = {
        "eligibility": [
            {
                "name": "observed",
                "version": "1",
                "function": "custom.scoring_custom.require_observed_values",
                "params": {"columns": ["x"]},
            }
        ],
        "outputs": [
            {
                "name": "band",
                "version": "1",
                "function": "custom.scoring_custom.prediction_band",
                "params": {"thresholds": [10.0, 50.0], "labels": ["low", "medium", "high"]},
                "columns": [{"name": "band", "dtype": "string"}],
            }
        ],
    }
    rows = pd.DataFrame({"x": [None, 10.0, 70.0]})
    result = run_project_scoring(
        rows,
        lambda frame: pd.DataFrame({"prediction": frame.x}),
        source=source,
        config=config,
        row_keys=[],
        prediction_dtypes={"prediction": "float64"},
    )
    assert result.exclusion_reason.iloc[0] == "missing:x"
    assert result.band.iloc[1:].tolist() == ["medium", "high"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_template_scoring_sections_reject_invalid_inputs_and_add_bands(engine, tmp_path):
    """Explicitly enabled examples must run with real exclusion and output values."""
    import importlib
    import shutil
    from pathlib import Path

    from skyulf.inference.project_code import load_project_module
    from skyulf.inference.project_scoring import run_project_scoring
    from skyulf.integrations.databricks._project_files import project_source

    root = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/features"
    )
    root = Path(shutil.copytree(root, tmp_path / "features"))
    scoring_path = root / "scoring.py"
    scoring_path.write_text(
        re.sub(r'(?m)^(        )# (?=[{}" ])', r"\1", scoring_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    source = project_source(root)
    module = load_project_module(source)
    scoring = importlib.import_module(f"{module.__name__}.scoring")
    config = {
        "eligibility": scoring.build_eligibility_rules(),
        "outputs": scoring.build_output_rules(),
    }
    rows = pd.DataFrame({"feature_value": [None, -1.0, 10.0, 70.0, 121.0, float("inf")]})
    native = pl.from_pandas(rows) if engine == "polars" else rows

    def predict(frame):
        """Only the two eligible rows may reach model execution."""
        values = frame["feature_value"].to_list()
        assert values == [10.0, 70.0]
        return pd.DataFrame({"prediction": values})

    result = run_project_scoring(
        native,
        predict,
        source=source,
        config=config,
        row_keys=[],
        prediction_dtypes={"prediction": "float64"},
    )
    assert result.scoring_status.tolist() == [
        "excluded",
        "excluded",
        "predicted",
        "predicted",
        "excluded",
        "excluded",
    ]
    assert result.exclusion_reason.iloc[0] == "missing:feature_value"
    assert (
        result.exclusion_reason.iloc[np.array([1, 4, 5])].tolist()
        == ["outside_range:feature_value"] * 3
    )
    assert result.band.iloc[2:4].tolist() == ["medium", "high"]
    assert result.prediction.iloc[np.array([0, 1, 4, 5])].isna().all()


@pytest.mark.parametrize(
    "limits",
    [
        {"minimum": 10, "maximum": 1},
        {"minimum": float("nan")},
        {"maximum": float("inf")},
        {},
    ],
)
def test_template_scoring_range_rejects_invalid_limits(limits):
    """An invalid business range must fail instead of silently accepting or excluding rows."""
    import importlib
    from pathlib import Path

    from skyulf.inference.project_code import load_project_module
    from skyulf.integrations.databricks._project_files import project_source

    root = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/features"
    )
    module = load_project_module(project_source(root))
    custom = importlib.import_module(f"{module.__name__}.custom.scoring_custom")
    with pytest.raises(ValueError, match="finite ordered"):
        custom.require_value_range(pd.DataFrame({"x": [3.0]}), {"column": "x", **limits})
