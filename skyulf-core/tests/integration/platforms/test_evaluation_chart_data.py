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
