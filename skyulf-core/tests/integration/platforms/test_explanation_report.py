"""Readable SHAP reports preserve contribution arithmetic and optional execution."""

import json
from unittest.mock import Mock

import pytest


def _evidence():
    """Use signed contributions so a waterfall cannot become an importance chart."""
    return {
        "status": "completed",
        "sample_count": 3,
        "feature_count": 2,
        "feature_names": ["amount", "city_<script>"],
        "limits": {"max_samples": 3, "max_features": 2, "max_display_samples": 1},
        "shap": {
            "mean_abs_importance": {"amount": 2.0, "city_<script>": 1.0},
            "samples": [
                {
                    "base_value": 5.0,
                    "shap_values": {"amount": 2.0, "city_<script>": -1.0},
                    "feature_values": {"amount": 10.0, "city_<script>": 1.0},
                }
            ],
        },
    }


def test_report_contains_exportable_charts_and_signed_waterfall():
    """A report must show actual charts, escaped labels and base plus signed contributions."""
    pytest.importorskip("matplotlib")
    from skyulf.integrations.databricks.observability.reports.explanation_report import (
        render_explanation_report,
    )

    report = render_explanation_report(_evidence())
    assert report.count("data:image/png;base64,") == 2
    assert "Global feature importance" in report and "Sample 1" in report
    assert "Base: 5" in report and "Explained output: 6" in report
    assert "city_&lt;script&gt;" in report and "<script>" not in report
    assert "not necessarily a probability" in report


@pytest.mark.parametrize("status", ["disabled", "unavailable"])
def test_unavailable_reports_do_not_import_plotting(status, monkeypatch):
    """Disabled or failed SHAP still has a readable status without plotting dependencies."""
    from skyulf.integrations.databricks.observability.reports import explanation_report

    monkeypatch.setattr(explanation_report, "_figure", lambda *args: pytest.fail("no plotting"))
    report = explanation_report.render_explanation_report(
        {"status": status, "reason": "feature_limit_exceeded"}
    )
    assert status in report and "feature_limit_exceeded" in report
    assert "data:image" not in report


def test_notebook_displays_each_training_run_once(tmp_path):
    """Multi-model outputs must show each child's evidence without duplicate reports."""
    from skyulf.integrations.databricks.observability.reports.explanation_report import (
        display_explanation_reports,
    )

    report = tmp_path / "explanations.html"
    report.write_text("<h2>Model explanations</h2>")
    client = Mock()
    client.list_artifacts.return_value = [Mock(path="explanations.html")]
    client.download_artifacts.return_value = str(report)
    display = Mock()
    display_explanation_reports(
        {"run_id": "parent", "branches": [{"run_id": "child"}, {"run_id": "child"}]},
        client,
        display,
    )
    assert display.call_count == 2
    assert [call.args[0] for call in client.download_artifacts.call_args_list] == [
        "parent",
        "child",
    ]


def test_training_persists_report_and_json(monkeypatch):
    """The HTML and JSON must describe the same bounded explanation, linked to its run."""
    pytest.importorskip("matplotlib")
    from skyulf.integrations.databricks.observability.reports import (
        explanations as local_explanations,
    )

    monkeypatch.setattr(local_explanations, "explain_training_artifact", lambda *args: _evidence())
    run = Mock(run_id="run123")
    run.client.get_run.return_value.info.experiment_id = "42"
    run.client.get_run.return_value.data.tags = {}
    local_explanations.log_training_explanations(run, Mock(), Mock())
    saved = run.client.log_dict.call_args.args[1]
    assert json.loads(json.dumps(saved))["status"] == "completed"
    html = run.client.log_text.call_args.args[1]
    assert "/ml/experiments/42/runs/run123" in html
    assert "runs:/run123/model" in html
    assert html.count("data:image/png;base64,") == 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "model_type",
    [
        "linear_regression",
        "logistic_regression",
        "voting_regressor",
        "voting_classifier",
        "stacking_regressor",
        "stacking_classifier",
    ],
)
def test_real_models_and_ensembles_explain_fitted_features(tmp_path, engine, model_type):
    """Each supported family must produce additive explanations after saved preprocessing."""
    pytest.importorskip("shap")
    import numpy as np
    import pandas as pd
    import polars as pl

    from skyulf.data.dataset import SplitDataset
    from skyulf.integrations.databricks.observability.reports.explanations import (
        explain_training_artifact,
    )
    from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow

    classification = model_type.endswith("classifier") or model_type == "logistic_regression"
    x = np.arange(30, dtype=float)
    frame = pd.DataFrame({"x": x, "z": np.sin(x), "target": (x % 2) if classification else x * 2})
    params = {}
    if model_type.startswith(("voting", "stacking")):
        params = {
            "base_estimators": ["logistic_regression", "gaussian_nb"]
            if classification
            else ["linear_regression", "ridge_regression"],
            "n_jobs": 1,
        }
        if model_type == "voting_classifier":
            params["voting"] = "soft"
        if model_type.startswith("stacking"):
            params["cv"] = 2
    config = {
        "preprocessing": [
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x", "z"]}}
        ],
        "modeling": {"type": model_type, "params": params},
        "explainability": {"method": "shap", "max_samples": 3, "max_display_samples": 3},
    }
    native = pl.from_pandas(frame) if engine == "polars" else frame
    artifact = fit_workflow(
        config,
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=tmp_path / "model",
        max_rows=100,
        max_bytes=1024 * 1024,
    )
    result = explain_training_artifact(artifact, native)
    assert result is not None and result["status"] == "completed"
    assert result["feature_names"] == ["x", "z"]
    assert artifact.pipeline.model_estimator is not None
    model = artifact.pipeline.model_estimator._unwrap_tuned_model()
    samples = result["shap"]["samples"]
    positions = np.sort(np.random.default_rng(42).choice(len(frame), 3, replace=False))
    inputs = native.select(["x", "z"]) if isinstance(native, pl.DataFrame) else native[["x", "z"]]
    values = artifact.pipeline.feature_engineer.transform(inputs).to_numpy()[positions]
    explained = [row["base_value"] + sum(row["shap_values"].values()) for row in samples]
    if model_type == "logistic_regression":
        expected = model.decision_function(values)
    elif classification:
        expected = model.predict_proba(values)[:, 1]
    else:
        expected = model.predict(values)
    np.testing.assert_allclose(explained, expected, atol=1e-4, rtol=1e-4)
