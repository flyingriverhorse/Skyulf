"""Hard voting artifacts expose labels without invented probabilities."""

import json

import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.pipeline_evaluation import evaluate_holdout
from skyulf.integrations.databricks.scoring.workflow import (
    InputSource,
    ModelSelection,
    OutputSink,
    WorkflowConfig,
    preflight,
)
from skyulf.pipeline import SkyulfPipeline


def _fit(tmp_path, *, voting: str = "hard"):
    """Fit a real Core voting model with both classes in each training subset."""
    rows = pd.DataFrame({"x": [float(i) for i in range(24)], "target": [i % 2 for i in range(24)]})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {
                "type": "voting_classifier",
                "params": {
                    "base_estimators": ["logistic_regression", "gaussian_nb"],
                    "voting": voting,
                },
            },
        }
    )
    pipeline.fit(SplitDataset(train=rows.iloc[:18], test=rows.head(0)), target_column="target")
    path = tmp_path / voting
    save_pipeline(pipeline, path)
    return path, load_pipeline(path), rows.iloc[18:]


def test_hard_voting_artifact_predicts_labels_and_evaluates_without_probabilities(tmp_path):
    """A probability-free classifier still yields saved labels and held-out label metrics."""
    path, artifact, holdout = _fit(tmp_path)
    result = predict_pipeline(holdout[["x"]], artifact)
    metrics = evaluate_holdout(artifact, holdout, target_column="target")

    assert artifact.manifest.classification_probabilities is False
    assert artifact.manifest.classes == (0, 1)
    assert result.columns.tolist() == ["prediction"]
    assert result["prediction"].isin([0, 1]).all()
    assert "heldout_accuracy" in metrics
    assert not any("auc" in name or "log_loss" in name for name in metrics)

    config = WorkflowConfig(
        runtime="standalone",
        engine="pandas",
        source=InputSource(kind="caller_frame"),
        model=ModelSelection(kind="local_pipeline", path=str(path)),
        sink=OutputSink(kind="return_frame"),
    )
    assert preflight(config, artifact=artifact).output_columns == ("prediction",)


def test_probability_capability_preserves_soft_voting_and_legacy_manifests(tmp_path):
    """Old manifest files load with the existing classifier probability contract."""
    path, artifact, holdout = _fit(tmp_path, voting="soft")
    assert artifact.manifest.classification_probabilities is True
    assert predict_pipeline(holdout[["x"]], artifact).columns.tolist() == [
        "prediction",
        "probability_0",
        "probability_1",
    ]
    metadata = path / "manifest.json"
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document.pop("classification_probabilities")
    metadata.write_text(json.dumps(document), encoding="utf-8")
    assert load_pipeline(path).manifest.classification_probabilities is True


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_mlflow_voting_signature_matches_pyfunc_predictions(tmp_path, voting):
    """Optional MLflow exposes only the probabilities supported by each voting mode."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.models.pipeline_model import (
        SkyulfPipelinePythonModel,
        _signature,
    )

    _, artifact, holdout = _fit(tmp_path, voting=voting)
    expected = ["prediction"]
    if voting == "soft":
        expected += ["probability_0", "probability_1"]
    assert [column.name for column in _signature(artifact).outputs.inputs] == expected
    pyfunc = SkyulfPipelinePythonModel()
    pyfunc._artifact = artifact
    assert pyfunc.predict(None, holdout[["x"]]).columns.tolist() == expected


def test_probability_capability_is_checked_against_fitted_model(tmp_path):
    """Metadata cannot claim probabilities that hard voting does not expose."""
    path, _, _ = _fit(tmp_path)
    metadata = path / "manifest.json"
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document["classification_probabilities"] = True
    metadata.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest disagrees"):
        load_pipeline(path)


def test_old_regression_manifest_keeps_loading_without_capability_field(tmp_path):
    """Adding classifier metadata must not invalidate saved regression artifacts."""
    rows = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=rows, test=rows.head(0)), target_column="target")
    path = tmp_path / "regression"
    save_pipeline(pipeline, path)
    metadata = path / "manifest.json"
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document.pop("classification_probabilities")
    metadata.write_text(json.dumps(document), encoding="utf-8")

    artifact = load_pipeline(path)
    assert artifact.manifest.task == "regression"
    assert predict_pipeline(rows[["x"]], artifact).columns.tolist() == ["prediction"]


def test_hard_voting_cannot_enable_tuned_thresholds(tmp_path):
    """A threshold policy cannot be saved without class probabilities."""
    path, artifact, _ = _fit(tmp_path)
    assert path.exists()
    with pytest.raises(ValueError, match="probabilit"):
        save_pipeline(artifact.pipeline, tmp_path / "thresholds", use_tuned_thresholds=True)
