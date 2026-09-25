"""Label-aware local candidate training from pinned Databricks snapshots."""

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks import local_retraining as retraining


def _spec(**changes):
    """Keep training, holdout and label cutoffs explicit in tests."""
    values = {
        "table": "workspace.test.labels",
        "version": 4,
        "start": datetime(2026, 1, 1, tzinfo=UTC),
        "holdout_start": datetime(2026, 2, 1, tzinfo=UTC),
        "cutoff": datetime(2026, 3, 1, tzinfo=UTC),
        "event_column": "event_time",
        "result_available_at_column": "label_at",
        "record_key_columns": ("id",),
        "input_columns": ("x",),
        "target_column": "target",
        "max_rows": 10,
        "max_bytes": 10000,
    }
    values.update(changes)
    return retraining.LocalTrainingSpec(**values)


def _frame():
    """Include an unavailable label and rows straddling the temporal holdout."""
    return pd.DataFrame(
        {
            "id": [3, 1, 4, 2, 5],
            "event_time": pd.to_datetime(
                ["2026-02-10", "2026-01-10", "2026-02-11", "2026-01-11", "2026-02-12"], utc=True
            ),
            "label_at": pd.to_datetime(
                ["2026-02-15", "2026-01-15", "2026-03-02", "2026-01-16", "2026-02-16"], utc=True
            ),
            "x": [3.0, 1.0, 4.0, 2.0, 5.0],
            "target": [6.0, 2.0, 8.0, 4.0, 10.0],
        }
    )


def test_split_excludes_late_labels_and_preserves_temporal_boundary():
    """A label unavailable at cutoff must never enter fit or evaluation."""
    train, holdout, skipped = retraining.split_labeled_snapshot(_frame(), _spec())
    assert train["x"].tolist() == [1.0, 2.0]
    assert holdout["x"].tolist() == [3.0, 5.0]
    assert skipped == 1
    assert "event_time" not in train.columns


def test_split_is_reproducible_under_source_row_reordering():
    """Snapshot row order cannot change the train or holdout membership."""
    first = retraining.split_labeled_snapshot(_frame(), _spec())
    second = retraining.split_labeled_snapshot(_frame().sample(frac=1, random_state=4), _spec())
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert first[2] == second[2]


def test_split_rejects_event_after_label_and_unpinned_source():
    """Invalid temporal provenance must fail before training begins."""
    frame = _frame()
    frame.loc[0, "label_at"] = pd.Timestamp("2026-02-01", tz="UTC")
    with pytest.raises(ValueError, match="event time"):
        retraining.split_labeled_snapshot(frame, _spec())
    with pytest.raises(ValueError, match="version"):
        _spec(version=None)


def test_reader_pins_projects_and_limits_before_collecting():
    """Training must not materialize a whole unversioned Delta table."""
    source = MagicMock()
    source.read.format.return_value = source.read
    source.read.option.return_value = source.read
    source.read.table.return_value = source
    source.where.return_value = source
    source.select.return_value = source
    source.orderBy.return_value = source
    source.limit.return_value = source
    source.toLocalIterator.return_value = iter([_frame().iloc[0].to_dict()])
    frame = retraining.read_training_snapshot(source, _spec())
    source.read.option.assert_called_once_with("versionAsOf", 4)
    source.where.assert_called_once()
    source.select.assert_called_once_with("id", "event_time", "label_at", "x", "target")
    source.limit.assert_called_once_with(11)
    assert len(frame) == 1


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_temporal_split_fits_real_skyulf_pipeline_without_holdout_leakage(tmp_path, engine):
    """Both local engines must fit on past labels and score unseen later rows."""
    train, holdout, _ = retraining.split_labeled_snapshot(_frame(), _spec())
    native_train = pl.from_pandas(train) if engine == "polars" else train
    native_holdout = pl.from_pandas(holdout) if engine == "polars" else holdout
    artifact = retraining.fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        SplitDataset(train=native_train, test=native_train.head(0)),
        target_column="target",
        artifact_path=tmp_path / engine,
        max_rows=10,
        max_bytes=10000,
    )
    metrics = retraining.evaluate_local_holdout(artifact, native_holdout, target_column="target")
    assert metrics["heldout_rmse"] == pytest.approx(0.0, abs=1e-8)


def test_failed_candidate_never_mutates_champion(monkeypatch, tmp_path):
    """A failed publication leaves promotion to an explicit later operation."""
    spec = _spec()
    monkeypatch.setattr(retraining, "read_training_snapshot", lambda spark, request: _frame())
    monkeypatch.setattr(
        retraining,
        "fit_local_workflow",
        lambda *args, **kwargs: SimpleNamespace(manifest=SimpleNamespace(pipeline_sha256="a" * 64)),
    )
    monkeypatch.setattr(
        retraining, "evaluate_local_holdout", lambda *args, **kwargs: {"heldout_rmse": 0.1}
    )
    monkeypatch.setattr(retraining, "_log_local_model", lambda *args, **kwargs: "runs:/run/model")
    monkeypatch.setattr(
        retraining,
        "register_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("publish failed")),
    )
    monkeypatch.setattr(
        retraining,
        "resolve_model",
        lambda *args, **kwargs: pytest.fail("alias must not be read before publish"),
    )

    @contextmanager
    def fake_run(*args, **kwargs):
        """Let publication fail after recording without contacting MLflow."""
        yield SimpleNamespace(
            run_id="run",
            client=SimpleNamespace(log_dict=lambda *args, **kwargs: None),
            log_config=lambda *args, **kwargs: None,
            log_params=lambda *args, **kwargs: None,
            set_tags=lambda *args, **kwargs: None,
            log_metrics=lambda *args, **kwargs: None,
        )

    monkeypatch.setattr(retraining, "track_run", fake_run)
    with pytest.raises(RuntimeError, match="publish failed"):
        retraining.train_local_candidate(
            None,
            spec,
            {"preprocessing": [], "modeling": {"type": "linear_regression"}},
            model_name="workspace.test.model",
            tracking_uri="file:unused",
            registry_uri="file:unused",
            experiment_name="test",
            run_name="candidate",
            artifact_path=tmp_path / "artifact",
            metric="heldout_rmse",
            min_improvement=0,
        )


def test_invalid_comparison_request_fails_before_mlflow_publication(monkeypatch, tmp_path):
    """An unusable metric must not create an orphan candidate version."""
    monkeypatch.setattr(retraining, "read_training_snapshot", lambda spark, request: _frame())
    monkeypatch.setattr(
        retraining,
        "fit_local_workflow",
        lambda *args, **kwargs: SimpleNamespace(manifest=SimpleNamespace(pipeline_sha256="a" * 64)),
    )
    monkeypatch.setattr(
        retraining, "evaluate_local_holdout", lambda *args, **kwargs: {"heldout_rmse": 0.1}
    )
    monkeypatch.setattr(
        retraining,
        "register_model",
        lambda *args, **kwargs: pytest.fail("invalid comparison must not register"),
    )
    with pytest.raises(ValueError, match="Selected metric"):
        retraining.train_local_candidate(
            None,
            _spec(),
            {"preprocessing": [], "modeling": {"type": "linear_regression"}},
            model_name="workspace.test.model",
            tracking_uri="file:unused",
            registry_uri="file:unused",
            experiment_name="test",
            run_name="candidate",
            artifact_path=tmp_path / "artifact",
            metric="heldout_not_a_metric",
            min_improvement=0,
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_candidate_workflow_logs_and_registers_without_alias(monkeypatch, tmp_path, engine):
    """A real local MLflow run must retain model, metrics and comparison evidence."""
    import mlflow

    x = list(range(12))
    frame = pd.DataFrame(
        {
            "id": x,
            "event_time": pd.to_datetime(["2026-01-10"] * 8 + ["2026-02-10"] * 4, utc=True),
            "label_at": pd.to_datetime(["2026-01-11"] * 8 + ["2026-02-11"] * 4, utc=True),
            "x": pd.Series(x, dtype="float64"),
            "target": pd.Series([2 * value for value in x], dtype="float64"),
        }
    )
    monkeypatch.setattr(retraining, "read_training_snapshot", lambda spark, request: frame)
    store = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=store, registry_uri=store)
    client.create_experiment("candidate_test", artifact_location=(tmp_path / "mlruns").as_uri())
    result = retraining.train_local_candidate(
        None,
        _spec(max_rows=12, max_bytes=20000),
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        model_name=f"candidate_{engine}",
        tracking_uri=store,
        registry_uri=store,
        experiment_name="candidate_test",
        run_name=engine,
        artifact_path=tmp_path / "artifact",
        metric="heldout_rmse",
        min_improvement=0,
        engine=engine,
    )
    run = client.get_run(result.run_id)
    artifacts = {artifact.path for artifact in client.list_artifacts(result.run_id)}
    assert result.model_version == "1"
    assert result.training_rows == 8 and result.holdout_rows == 4
    assert result.comparison.reason == "no_champion"
    assert run.data.metrics["heldout_rmse"] == pytest.approx(0.0, abs=1e-8)
    assert "candidate_comparison.json" in artifacts
    assert "candidate_training_spec.json" in artifacts
    assert "model" in artifacts
    assert run.data.tags["task"] == "training"
    assert "dataset_id" not in run.data.tags and "phase" not in run.data.tags
    assert run.data.tags["train_data_destination"] == _spec().table
    assert run.data.tags["train_data_version"] == str(_spec().version)
    assert run.data.tags["test_data_version"] == str(_spec().version)
    assert "training_data.json" in artifacts
    model_tags = client.get_model_version(f"candidate_{engine}", "1").tags
    assert model_tags["engine"] == engine
    assert model_tags["model_type"] == "linear_regression"
    assert model_tags["train_data_destination"] == _spec().table
    registrations = []
    original_compare = retraining.compare_registered_local_models

    def compare_after_registration(candidate, *args, **kwargs):
        """Comparison must follow the orchestrator's explicit registration hook."""
        assert registrations == [candidate]
        return original_compare(candidate, *args, **kwargs)

    monkeypatch.setattr(retraining, "compare_registered_local_models", compare_after_registration)
    next_result = retraining.train_local_candidate(
        None,
        _spec(max_rows=12, max_bytes=20000),
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        model_name=f"candidate_{engine}",
        tracking_uri=store,
        registry_uri=store,
        experiment_name="candidate_test",
        run_name=f"{engine}-challenger",
        artifact_path=tmp_path / "artifact2",
        metric="heldout_rmse",
        min_improvement=0,
        engine=engine,
        champion_version=result.model_version,
        on_registered=registrations.append,
    )
    assert next_result.model_version == "2"
    assert next_result.comparison.champion_version == "1"
    assert next_result.comparison.eligible is False
    with pytest.raises(Exception, match="alias|Alias"):
        client.get_model_version_by_alias(result.model_name, "champion")
