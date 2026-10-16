"""Chart inputs stay optional, bounded and tied to the exact fitted model."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.charts.evaluation_chart_data import (
    chart_recorder,
    chart_settings,
    load_chart_sample,
)


class ArtifactClient:
    """Keep genuine artifact bytes so corruption and schema drift are observable."""

    def __init__(self):
        """Create an isolated in-memory artifact store."""
        self.files = {}

    def log_artifact(self, run_id, path, artifact_path):
        """Store uploaded Parquet bytes under the requested run."""
        self.files[run_id, f"{artifact_path}/{Path(path).name}"] = Path(path).read_bytes()

    def log_dict(self, run_id, value, path):
        """Serialize metadata to match the production artifact boundary."""
        self.files[run_id, path] = json.dumps(value).encode()

    def download_artifacts(self, run_id, path, destination):
        """Return the exact saved bytes through a disposable local file."""
        result = Path(destination) / Path(path).name
        result.write_bytes(self.files[run_id, path])
        return str(result)


def fitted_candidate():
    """Include an unrelated identifier to ensure chart inputs project model columns only."""
    return SimpleNamespace(
        artifact=SimpleNamespace(
            manifest=SimpleNamespace(
                input_columns=("amount",),
                pipeline_sha256="a" * 64,
            )
        ),
        spec=SimpleNamespace(
            target_column="target", dataset_id="source@3/split", holdout_key_sha256="b" * 64
        ),
        evidence_holdout=pd.DataFrame(
            {"amount": np.arange(30), "target": np.arange(30) * 2, "private_id": np.arange(30)}
        ),
    )


def test_disabled_charts_do_not_write_or_inspect_model():
    """Opting out must leave training free of chart data and plotting dependencies."""
    client = ArtifactClient()
    assert chart_recorder(SimpleNamespace(client=client, run_id="run"), None, None, None) is None
    assert client.files == {}


@pytest.mark.parametrize(
    "value",
    [
        True,
        "true",
        {"enabled": "true"},
        {"enabled": True, "max_rows": True},
        {"enabled": True, "max_rows": 10001},
        {"enabled": True, "unknown": 1},
    ],
)
def test_invalid_chart_settings_fail_before_training(value):
    """Mistyped flags and unbounded requests must not silently enable expensive work."""
    with pytest.raises(ValueError):
        chart_settings(value)


def test_chart_sample_is_bounded_projected_and_pinned():
    """A report must use the original heldout rows and fitted feature contract."""
    client = ArtifactClient()
    fitted = fitted_candidate()
    settings = {"enabled": True, "max_rows": 7}
    run = SimpleNamespace(client=client, run_id="run")
    record_sample(run, fitted, settings)
    frame, metadata = load_chart_sample(client, "run", "a" * 64)
    assert frame is not None
    assert len(frame) == 7
    assert list(frame.columns) == ["observed", "prediction"]
    assert metadata["holdout_rows"] == 30
    assert metadata["dataset_id"] == fitted.spec.dataset_id
    assert metadata["model_digest"] == "a" * 64
    assert metadata["settings"]["max_rows"] == 7
    assert set(frame["observed"]) <= set(fitted.evidence_holdout["target"])


def test_corrupt_or_wrong_model_sample_is_rejected():
    """Saved evidence from a different model or changed data cannot create plausible charts."""
    client = ArtifactClient()
    record_sample(
        SimpleNamespace(client=client, run_id="run"), fitted_candidate(), {"enabled": True}
    )
    with pytest.raises(ValueError, match="model"):
        load_chart_sample(client, "run", "c" * 64)
    client.files["run", "evaluation_charts/holdout.parquet"] += b"changed"
    with pytest.raises(ValueError, match="digest"):
        load_chart_sample(client, "run", "a" * 64)


def test_samples_are_reproducible_without_changing_source():
    """Repeated rendering preparation must preserve row pairing and the original frame."""
    client = ArtifactClient()
    fitted = fitted_candidate()
    original = fitted.evidence_holdout.copy(deep=True)
    for run_id in ("first", "second"):
        record_sample(
            SimpleNamespace(client=client, run_id=run_id), fitted, {"enabled": True, "max_rows": 6}
        )
    one, _ = load_chart_sample(client, "first", "a" * 64)
    two, _ = load_chart_sample(client, "second", "a" * 64)
    assert one is not None and two is not None
    pd.testing.assert_frame_equal(one, two)
    pd.testing.assert_frame_equal(original, fitted.evidence_holdout)
    assert (one["observed"] == one["prediction"]).all()


def record_sample(run, fitted, settings):
    """Provide paired outputs without persisting any raw feature or identifier."""
    callback = chart_recorder(run, fitted.artifact, fitted.spec, settings)
    assert callback is not None
    actual = fitted.evidence_holdout["target"].to_numpy()
    callback(actual, pd.DataFrame({"prediction": actual}))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_temporal_predictions_are_sampled_after_full_holdout_evaluation(tmp_path, engine):
    """Sampling must not change rolling features or lose the preceding heldout rows."""
    from test_temporal_history import fitted_temporal_pipeline

    from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
    from skyulf.integrations.databricks.training.fitting.candidate import evaluate_candidate

    fitted = fitted_candidate()
    path = tmp_path / "model"
    save_pipeline(fitted_temporal_pipeline(engine), path)
    artifact = load_pipeline(path)
    rows = pd.DataFrame(
        {"t": np.arange(20, 30, dtype=np.int64), "v": np.arange(20, 30, dtype=float)}
    )
    expected = predict_pipeline(rows, artifact)
    rows["target"] = np.arange(20, 30, dtype=float)
    client = ArtifactClient()
    evaluate_candidate(
        artifact,
        rows,
        spec=fitted.spec,
        metric="heldout_rmse",
        chart_run=SimpleNamespace(client=client, run_id="run"),
        evaluation_charts={"enabled": True, "max_rows": 4},
    )
    sample, _ = load_chart_sample(client, "run", artifact.manifest.pipeline_sha256)
    assert sample is not None
    indices = (sample["observed"] - 20).astype(int).to_numpy()
    np.testing.assert_allclose(sample["prediction"], expected["prediction"].iloc[indices])
    assert list(sample.columns) == ["observed", "prediction"]


def test_optional_serialization_failure_records_unavailable(monkeypatch):
    """Chart evidence preparation must not invalidate an otherwise valid training result."""

    def fail(*args, **kwargs):
        """Represent an unsupported serialization result for optional diagnostics."""
        raise RuntimeError("serialization unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    client = ArtifactClient()
    record_sample(
        SimpleNamespace(client=client, run_id="run"), fitted_candidate(), {"enabled": True}
    )
    sample, metadata = load_chart_sample(client, "run", "a" * 64)
    assert sample is None
    assert metadata["status"] == "unavailable"
    assert "RuntimeError" in metadata["reason"]


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_saved_exclusion_sample_plots_only_scored_pairs(tmp_path, monkeypatch, task):
    """Persisted nullable predictions must yield aligned diagnostics without changing the sample."""
    pytest.importorskip("matplotlib")
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
    from skyulf.inference.pipeline_evaluation import evaluate_holdout
    from skyulf.integrations.databricks.observability.charts import evaluation_chart_report
    from skyulf.pipeline import SkyulfPipeline

    values = np.arange(1, 41, dtype=float)
    frame = pd.DataFrame(
        {"x": values, "target": values * 2 if task == "regression" else values % 2}
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "bounds", "transformer": "IQR", "params": {"columns": ["x"]}}
            ],
            "modeling": {
                "type": "linear_regression" if task == "regression" else "logistic_regression"
            },
        }
    )
    pipeline.fit(SplitDataset(train=frame[:32], test=frame[32:]), target_column="target")
    save_pipeline(pipeline, tmp_path / "model")
    artifact = load_pipeline(tmp_path / "model")
    client = ArtifactClient()
    spec = fitted_candidate().spec
    observer = chart_recorder(
        SimpleNamespace(client=client, run_id="run"),
        artifact,
        spec,
        {"enabled": True, "max_rows": 3},
    )
    heldout = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 1000.0],
            "target": [2.0, 4.0, 6.0, 8.0, 2000.0]
            if task == "regression"
            else [0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
    evaluate_holdout(artifact, heldout, target_column="target", on_predictions=observer)
    sample, metadata = load_chart_sample(client, "run", artifact.manifest.pipeline_sha256)
    assert sample is not None and sample.scoring_status.tolist().count("excluded") == 1
    original_files = dict(client.files)
    monkeypatch.setattr(evaluation_chart_report, "load_run_pipeline", lambda *a, **k: artifact)
    monkeypatch.setattr(evaluation_chart_report, "_estimator_figures", lambda *a: ({}, []))
    monkeypatch.setattr(evaluation_chart_report, "evidence_charts", lambda *a: {})
    captured = []

    def publish(client, destination, figures, *, prefix, caption):
        """Inspect real chart content while other tests cover MLflow image uploads."""
        for name, figure in figures.items():
            if name == "actual_vs_predicted":
                plotted = figure.axes[0].collections[0].get_offsets()
                expected = sample.loc[
                    sample.scoring_status.eq("predicted"), ["observed", "prediction"]
                ]
                np.testing.assert_allclose(plotted, expected.to_numpy(dtype=float))
            figure.clear()
        captured.append(caption)
        return list(figures)

    monkeypatch.setattr(evaluation_chart_report, "publish_figures", publish)
    report = evaluation_chart_report.report_model(
        client,
        "unused",
        {
            "run_id": "run",
            "model_digest": artifact.manifest.pipeline_sha256,
            "dataset_id": spec.dataset_id,
            "holdout_key_sha256": spec.holdout_key_sha256,
            "holdout_rows": len(heldout),
        },
        destination="charts",
        prefix="single",
    )
    assert report["sample_rows"] == 3 and report["holdout_rows"] == 5
    assert report["sample_scored_rows"] == 2 and report["sample_excluded_rows"] == 1
    assert report["sample_excluded_by_reason"] == {"outlier:bounds": 1}
    assert report["image_keys"] and "1 excluded" in captured[0]
    assert client.files == original_files


def _status_artifact():
    """Describe a regression model with declared scoring statuses for focused contract checks."""
    return SimpleNamespace(
        manifest=SimpleNamespace(task="regression", classes=(), classification_probabilities=False),
        pipeline=SimpleNamespace(config={"project_scoring": {"eligibility": [], "outputs": []}}),
    )


@pytest.mark.parametrize("survivors", [0, 1])
def test_small_scored_chart_population_is_explicit(survivors):
    """An almost entirely excluded sample must skip diagnostics without inventing data."""
    from skyulf.integrations.databricks.observability.charts.evaluation_chart_report import (
        _scored_chart_sample,
    )
    from skyulf.integrations.databricks.observability.charts.evaluation_charts import (
        diagnostic_charts,
    )

    frame = pd.DataFrame(
        {
            "observed": [1.0] * (survivors + 1),
            "prediction": [1.0] * survivors + [None],
            "scoring_status": ["predicted"] * survivors + ["excluded"],
            "exclusion_reason": [None] * survivors + ["outlier:bounds"],
        }
    )
    report = {}
    plotted = _scored_chart_sample(frame, _status_artifact(), report)
    figures, reasons = diagnostic_charts("regression", plotted.observed, plotted, (), 20)
    assert figures == {} and reasons == ["Holdout sample has fewer than two rows."]
    assert report["sample_scored_rows"] == survivors and report["sample_excluded_rows"] == 1


@pytest.mark.parametrize("corruption", ["column", "status", "prediction"])
def test_chart_exclusion_support_does_not_hide_corrupt_predictions(corruption):
    """Only explicit exclusions may leave the diagnostic population; malformed values still fail."""
    from skyulf.integrations.databricks.observability.charts.evaluation_chart_report import (
        _scored_chart_sample,
    )
    from skyulf.integrations.databricks.observability.charts.evaluation_charts import (
        diagnostic_charts,
    )

    frame = pd.DataFrame(
        {
            "observed": [1.0, 2.0],
            "prediction": [1.0, 2.0],
            "scoring_status": ["predicted", "predicted"],
            "exclusion_reason": [None, None],
        }
    )
    if corruption == "column":
        frame["undeclared"] = 1
    elif corruption == "status":
        frame.loc[0, "scoring_status"] = None
    else:
        frame.loc[0, "prediction"] = np.nan
    with pytest.raises(ValueError, match="contract|status|finite"):
        plotted = _scored_chart_sample(frame, _status_artifact(), {})
        diagnostic_charts("regression", plotted.observed, plotted, (), 20)
