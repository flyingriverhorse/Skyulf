"""Compare pinned local MLflow model versions without changing registry state."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import LocalPipelineArtifact
from skyulf.integrations.databricks.local_batch import fit_local_workflow
from skyulf.integrations.mlflow import validation
from skyulf.integrations.mlflow.registry import ResolvedModel


def _fitted(tmp_path, offset: float) -> LocalPipelineArtifact:
    """Fit one model whose known bias makes comparison direction observable."""
    x = np.arange(12, dtype="float64")
    frame = pd.DataFrame({"x": x, "target": 2.0 * x + offset})
    return fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        SplitDataset(train=frame, test=frame.head(0)),
        target_column="target",
        artifact_path=tmp_path / f"model-{offset}",
        max_rows=12,
        max_bytes=10_000,
    )


def _reference(artifact: LocalPipelineArtifact, version: str) -> ResolvedModel:
    """Bind the fitted artifact to one concrete registry identity."""
    return ResolvedModel(
        name="workspace.test.risk",
        version=version,
        model_uri=f"models:/workspace.test.risk/{version}",
        signature=None,
        digest=artifact.manifest.pipeline_sha256,
    )


def _holdout() -> pd.DataFrame:
    """Reserve the same unseen labels for both registered versions."""
    x = np.arange(20, 25, dtype="float64")
    return pd.DataFrame({"x": x, "target": 2.0 * x})


def test_comparison_uses_same_holdout_and_minimizes_error(tmp_path, monkeypatch) -> None:
    """A better candidate should pass only after both pinned versions are scored."""
    champion = _fitted(tmp_path, 10.0)
    candidate = _fitted(tmp_path, 0.0)
    references = {_reference(champion, "1"): champion, _reference(candidate, "2"): candidate}
    loaded: list[str] = []

    def load(resolved, **kwargs):
        """Record concrete loads; evaluation has no alias mutation path."""
        loaded.append(resolved.version)
        return references[resolved]

    monkeypatch.setattr(validation, "load_registered_local_pipeline", load)
    heldout = _holdout()
    original = heldout.copy(deep=True)
    report = validation.compare_registered_local_models(
        _reference(candidate, "2"),
        _reference(champion, "1"),
        heldout,
        target_column="target",
        dataset_id="workspace.test.labels@5/holdout-v1",
        metric="heldout_rmse",
        min_improvement=1.0,
        max_rows=20,
        max_bytes=10_000,
    )

    assert loaded == ["2", "1"]
    pd.testing.assert_frame_equal(heldout, original)
    assert report.eligible is True
    assert report.reason == "candidate_improved"
    assert report.candidate_metrics["heldout_rmse"] == pytest.approx(0.0, abs=1e-8)
    assert report.champion_metrics is not None
    assert report.champion_metrics["heldout_rmse"] == pytest.approx(10.0, abs=1e-8)
    assert report.improvement == pytest.approx(10.0)
    assert report.dataset_id == "workspace.test.labels@5/holdout-v1"
    assert report.row_count == 5
    assert report.code_version == version("skyulf-core")
    assert json.loads(json.dumps(asdict(report)))["candidate_version"] == "2"


def test_first_candidate_is_reported_without_automatic_champion(tmp_path, monkeypatch) -> None:
    """The first registered version must still require a separate bootstrap choice."""
    candidate = _fitted(tmp_path, 0.0)
    monkeypatch.setattr(
        validation, "load_registered_local_pipeline", lambda *args, **kwargs: candidate
    )
    report = validation.compare_registered_local_models(
        _reference(candidate, "1"),
        None,
        _holdout(),
        target_column="target",
        dataset_id="workspace.test.labels@5/holdout-v1",
        metric="heldout_r2",
        min_improvement=0.0,
        max_rows=20,
        max_bytes=10_000,
    )

    assert report.eligible is False
    assert report.reason == "no_champion"
    assert report.champion_metrics is None


def test_incompatible_class_contract_rejects_before_scoring(tmp_path, monkeypatch) -> None:
    """Different class labels cannot produce a valid champion comparison."""
    baseline = _fitted(tmp_path, 0.0)
    champion = replace(
        baseline,
        manifest=baseline.manifest.model_copy(
            update={"task": "classification", "classes": ("no", "yes")}
        ),
    )
    candidate = replace(
        baseline,
        manifest=baseline.manifest.model_copy(
            update={"task": "classification", "classes": ("negative", "positive")}
        ),
    )
    monkeypatch.setattr(
        validation,
        "load_registered_local_pipeline",
        lambda resolved, **kwargs: candidate if resolved.version == "2" else champion,
    )
    with pytest.raises(ValueError, match="class"):
        validation.compare_registered_local_models(
            _reference(candidate, "2"),
            _reference(champion, "1"),
            _holdout(),
            target_column="target",
            dataset_id="workspace.test.labels@5/holdout-v1",
            metric="heldout_accuracy",
            min_improvement=0.0,
            max_rows=20,
            max_bytes=10_000,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"metric": "unknown_metric"}, "metric"),
        ({"dataset_id": ""}, "dataset_id"),
        ({"max_rows": 2}, "max_rows"),
        ({"max_bytes": 1}, "max_bytes"),
        ({"min_improvement": -1.0}, "min_improvement"),
    ],
)
def test_invalid_comparison_fails_before_model_load(
    tmp_path, monkeypatch, changes, message
) -> None:
    """Malformed policy or oversized data must not download registered models."""
    candidate = _fitted(tmp_path, 0.0)
    monkeypatch.setattr(
        validation,
        "load_registered_local_pipeline",
        lambda *args, **kwargs: pytest.fail("model should not load"),
    )
    options: dict[str, Any] = {
        "target_column": "target",
        "dataset_id": "workspace.test.labels@5/holdout-v1",
        "metric": "heldout_rmse",
        "min_improvement": 0.0,
        "max_rows": 20,
        "max_bytes": 10_000,
    }
    options.update(changes)
    with pytest.raises(ValueError, match=message):
        validation.compare_registered_local_models(
            _reference(candidate, "2"), None, _holdout(), **options
        )


@pytest.mark.parametrize("metric", ["heldout_accuracy", "heldout_roc_auc", "heldout_log_loss"])
def test_polars_classification_comparison_selects_core_metrics(
    tmp_path, monkeypatch, metric
) -> None:
    """A candidate must beat a biased champion in the chosen metric direction."""
    import polars as pl

    x = np.r_[np.arange(-12, 0), np.arange(1, 13)].astype("float64")
    truth = np.where(x > 0, "yes", "no")
    train = pl.DataFrame({"x": x, "target": truth})
    reversed_train = pl.DataFrame({"x": x, "target": np.where(x > 0, "no", "yes")})
    config = {"preprocessing": [], "modeling": {"type": "logistic_regression"}}
    candidate = fit_local_workflow(
        config,
        SplitDataset(train=train, test=train.head(0)),
        target_column="target",
        artifact_path=tmp_path / "candidate-polars",
        max_rows=24,
        max_bytes=20_000,
    )
    champion = fit_local_workflow(
        config,
        SplitDataset(train=reversed_train, test=reversed_train.head(0)),
        target_column="target",
        artifact_path=tmp_path / "champion-polars",
        max_rows=24,
        max_bytes=20_000,
    )
    assert candidate.manifest.classes == champion.manifest.classes
    monkeypatch.setattr(
        validation,
        "load_registered_local_pipeline",
        lambda resolved, **kwargs: candidate if resolved.version == "2" else champion,
    )
    heldout = pl.DataFrame({"x": [-3.0, -2.0, 2.0, 3.0], "target": ["no", "no", "yes", "yes"]})

    report = validation.compare_registered_local_models(
        _reference(candidate, "2"),
        _reference(champion, "1"),
        heldout,
        target_column="target",
        dataset_id="workspace.test.labels@6/class-holdout",
        metric=metric,
        min_improvement=0.2,
        quality_threshold=0.9,
        max_rows=10,
        max_bytes=10_000,
    )

    assert report.champion_metrics is not None
    if metric == "heldout_log_loss":
        assert report.metric_direction == "minimize"
        assert report.candidate_metrics[metric] < report.champion_metrics[metric]
    else:
        assert report.metric_direction == "maximize"
        assert report.candidate_metrics[metric] == pytest.approx(1.0)
        assert report.champion_metrics[metric] == pytest.approx(0.0)
    assert report.eligible is True


def test_real_local_registry_comparison_keeps_alias_pinned(tmp_path) -> None:
    """Fetching both registered versions must leave the champion alias unchanged."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.local_model import log_local_model
    from skyulf.integrations.mlflow.registry import register_model, resolve_model
    from skyulf.integrations.mlflow.tracking import TrackingConfig, track_run

    uri = f"sqlite:///{(tmp_path / 'models.db').as_posix()}"
    tracking = TrackingConfig(enabled=True, tracking_uri=uri, experiment_name="sm22-local")
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment("sm22-local", artifact_location=(tmp_path / "artifacts").as_uri())
    name = "sm22-candidate-champion"
    for offset in (10.0, 0.0):
        _fitted(tmp_path, offset)
        with track_run(tracking, run_name=f"candidate-{offset}") as run:
            assert run.run_id is not None
            model_uri = log_local_model(
                tmp_path / f"model-{offset}",
                run_id=run.run_id,
                artifact_path="model",
                tracking_uri=uri,
            )
        register_model(model_uri, name, tracking_uri=uri, registry_uri=uri)
    client.set_registered_model_alias(name, "champion", "1")
    champion = resolve_model(name, alias="champion", tracking_uri=uri, registry_uri=uri)
    candidate = resolve_model(name, version="2", tracking_uri=uri, registry_uri=uri)

    report = validation.compare_registered_local_models(
        candidate,
        champion,
        _holdout(),
        target_column="target",
        dataset_id="workspace.test.labels@5/holdout-v1",
        metric="heldout_rmse",
        min_improvement=1.0,
        max_rows=20,
        max_bytes=10_000,
        tracking_uri=uri,
        registry_uri=uri,
    )

    assert report.eligible is True
    assert report.candidate_version == "2"
    assert report.champion_version == "1"
    assert str(client.get_model_version_by_alias(name, "champion").version) == "1"


def test_candidate_must_clear_absolute_quality_gate(tmp_path, monkeypatch) -> None:
    """Relative improvement alone must not qualify a still-poor challenger."""
    champion = _fitted(tmp_path, 10.0)
    candidate = _fitted(tmp_path, 5.0)
    monkeypatch.setattr(
        validation,
        "load_registered_local_pipeline",
        lambda resolved, **kwargs: candidate if resolved.version == "2" else champion,
    )

    report = validation.compare_registered_local_models(
        _reference(candidate, "2"),
        _reference(champion, "1"),
        _holdout(),
        target_column="target",
        dataset_id="workspace.test.labels@5/holdout-v1",
        metric="heldout_rmse",
        min_improvement=1.0,
        quality_threshold=1.0,
        max_rows=20,
        max_bytes=10_000,
    )

    assert report.improvement == pytest.approx(5.0)
    assert report.eligible is False
    assert report.reason == "quality_gate_failed"


def test_malformed_resolved_version_fails_before_registry_load(tmp_path, monkeypatch) -> None:
    """A caller-built reference with a non-string version must fail clearly."""
    candidate = _fitted(tmp_path, 0.0)
    reference = replace(_reference(candidate, "2"), version=2)
    monkeypatch.setattr(
        validation,
        "load_registered_local_pipeline",
        lambda *args, **kwargs: pytest.fail("registry must not be called"),
    )
    with pytest.raises(ValueError, match="concrete version"):
        validation.compare_registered_local_models(
            reference,
            None,
            _holdout(),
            target_column="target",
            dataset_id="workspace.test.labels@5/holdout-v1",
            metric="heldout_rmse",
            min_improvement=0.0,
            max_rows=20,
            max_bytes=10_000,
        )


def test_registry_permission_failure_does_not_yield_a_report(tmp_path, monkeypatch) -> None:
    """A denied candidate load must propagate instead of looking like a failed metric."""
    from skyulf.integrations.mlflow.registry import RegistryAccessError

    candidate = _fitted(tmp_path, 0.0)

    def denied(*args, **kwargs):
        """Represent an access failure at the optional registry boundary."""
        raise RegistryAccessError("denied")

    monkeypatch.setattr(validation, "load_registered_local_pipeline", denied)
    with pytest.raises(RegistryAccessError, match="denied"):
        validation.compare_registered_local_models(
            _reference(candidate, "2"),
            None,
            _holdout(),
            target_column="target",
            dataset_id="workspace.test.labels@5/holdout-v1",
            metric="heldout_rmse",
            min_improvement=0.0,
            max_rows=20,
            max_bytes=10_000,
        )


def test_too_small_holdout_fails_before_registry_load(tmp_path, monkeypatch) -> None:
    """One labeled row cannot produce meaningful held-out metrics or a remote load."""
    candidate = _fitted(tmp_path, 0.0)
    monkeypatch.setattr(
        validation,
        "load_registered_local_pipeline",
        lambda *args, **kwargs: pytest.fail("registry must not be called"),
    )
    with pytest.raises(ValueError, match="at least two"):
        validation.compare_registered_local_models(
            _reference(candidate, "2"),
            None,
            _holdout().head(1),
            target_column="target",
            dataset_id="workspace.test.labels@5/holdout-v1",
            metric="heldout_rmse",
            min_improvement=0.0,
            max_rows=20,
            max_bytes=10_000,
        )
