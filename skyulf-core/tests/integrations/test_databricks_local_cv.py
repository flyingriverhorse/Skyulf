"""Bundle CV must reuse Core while keeping final evaluation and fitted state isolated."""

from datetime import UTC, datetime
from typing import cast
from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest

from skyulf.integrations.databricks.local_cv import LocalCVSpec, evaluate_training_cv
from skyulf.integrations.databricks.local_retraining import (
    LocalTrainingSpec,
    split_labeled_snapshot,
)
from skyulf.modeling.base import BaseModelCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


def _pipeline(classification=False):
    """Exercise real learned preprocessing and configurable model parameters."""
    return {
        "preprocessing": [
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}}
        ],
        "modeling": {
            "type": "logistic_regression" if classification else "linear_regression",
            "params": {"C": 0.5} if classification else {"fit_intercept": True},
        },
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["k_fold", "shuffle_split", "stratified_k_fold"])
def test_cv_refits_each_fold_on_raw_training_rows(monkeypatch, engine, method):
    """Fold preprocessing must never see final holdout rows or another fold's learned state."""
    classification = method == "stratified_k_fold"
    frame = pd.DataFrame({"id": range(60), "x": range(60)})
    frame["target"] = frame.x % 2 if classification else 3 * frame.x + 2
    source = LocalTrainingSpec(
        table="workspace.test.source",
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=100,
        max_bytes=100000,
        stratify=classification,
    )
    train, holdout, _ = split_labeled_snapshot(frame, source)
    native = pl.from_pandas(train) if engine == "polars" else train
    observed = []
    original = FeatureEngineerFoldAdapter.fit_transform

    def inspect_fit(self, X, y):
        """Observe real fold payloads without replacing learned preprocessing."""
        observed.append(set(X["x"].to_list()))
        assert isinstance(X, pl.DataFrame if engine == "polars" else pd.DataFrame)
        assert "target" not in X.columns
        return original(self, X, y)

    monkeypatch.setattr(FeatureEngineerFoldAdapter, "fit_transform", inspect_fit)
    settings = LocalCVSpec(enabled=True, folds=3, method=method)
    result = evaluate_training_cv(
        native, _pipeline(classification), settings, target_column="target"
    )
    assert result is not None
    assert len(observed) == 3
    assert all(rows < set(train.x) and rows.isdisjoint(holdout.x) for rows in observed)
    assert result["fold_refit"]["fit_calls"] == 3
    assert len(result["folds"]) == 3
    assert result["aggregated_metrics"]
    if not classification:
        assert result["aggregated_metrics"]["rmse"]["mean"] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_temporal_cv_keeps_time_metadata_out_of_features(monkeypatch, engine):
    """Chronological CV must use normalized source times while fitting features only."""
    frame = pd.DataFrame(
        {
            "id": range(30),
            "x": range(30),
            "target": [2 * x for x in range(30)],
            "event": pd.date_range("2026-01-01", periods=30, tz="UTC"),
        }
    ).iloc[::-1]
    source = LocalTrainingSpec(
        table="workspace.test.source",
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=100,
        max_bytes=100000,
        split_strategy="temporal",
        event_column="event",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        holdout_start=datetime(2026, 1, 25, tzinfo=UTC),
        cutoff=datetime(2026, 2, 1, tzinfo=UTC),
    )
    train, holdout, _ = split_labeled_snapshot(frame, source, keep_training_event=True)
    assert "event" in train and "event" not in holdout
    native = pl.from_pandas(train) if engine == "polars" else train
    seen = []
    original = FeatureEngineerFoldAdapter.fit_transform

    def inspect_fit(self, X, y):
        """The chronological key cannot accidentally become a numeric model feature."""
        assert list(X.columns) == ["x"]
        seen.append(X["x"].to_list())
        return original(self, X, y)

    monkeypatch.setattr(FeatureEngineerFoldAdapter, "fit_transform", inspect_fit)
    result = evaluate_training_cv(
        native,
        _pipeline(),
        LocalCVSpec(enabled=True, folds=3, method="time_series_split", shuffle=False),
        target_column="target",
        event_column="event",
    )
    assert result is not None
    assert seen == [list(range(6)), list(range(12)), list(range(18))]
    assert result["cv_config"]["time_column"] == "event"
    assert result["aggregated_metrics"]["rmse"]["mean"] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize(
    "changes",
    [
        {"folds": 1},
        {"folds": 21},
        {"folds": True},
        {"enabled": "yes"},
        {"random_state": -1},
        {"random_state": 2**32},
        {"shuffle": 1},
        {"method": "nested_cv"},
        {"method": "unknown"},
        {"method": "time_series_split", "shuffle": True},
        {"method": "shuffle_split", "shuffle": False},
    ],
)
def test_invalid_cv_settings_fail_explicitly(changes):
    """Requested validation semantics must not silently fall back to a different splitter."""
    with pytest.raises(ValueError):
        LocalCVSpec(**changes)


def test_disabled_cv_does_not_fit_or_validate_data():
    """Leaving CV disabled must add no fold fits to an ordinary training run."""
    assert evaluate_training_cv(pd.DataFrame(), {}, LocalCVSpec(), target_column="target") is None


def test_stratified_regression_and_rare_classes_fail_before_fit():
    """Classification guards must apply before Core's permissive splitter fallback."""
    settings = LocalCVSpec(enabled=True, folds=3, method="stratified_k_fold")
    frame = pd.DataFrame({"x": range(12), "target": [0] * 10 + [1] * 2})
    with pytest.raises(ValueError, match="classification"):
        evaluate_training_cv(frame, _pipeline(), settings, target_column="target")
    with pytest.raises(ValueError, match="rows per class"):
        evaluate_training_cv(frame, _pipeline(True), settings, target_column="target")


def test_temporal_cv_rejects_missing_or_tied_boundary_times():
    """Equal timestamps across a fold boundary cannot represent strictly future validation."""
    settings = LocalCVSpec(enabled=True, folds=2, method="time_series_split", shuffle=False)
    frame = pd.DataFrame({"x": range(12), "target": range(12)})
    with pytest.raises(ValueError, match="event_column"):
        evaluate_training_cv(frame, _pipeline(), settings, target_column="target")
    frame["event"] = pd.Timestamp("2026-01-01", tz="UTC")
    with pytest.raises(ValueError, match="timestamp.*boundary"):
        evaluate_training_cv(
            frame, _pipeline(), settings, target_column="target", event_column="event"
        )


def test_workflow_rejects_cv_before_monthly_source_or_registry_access(monkeypatch, tmp_path):
    """Invalid CV must fail before a direct workflow invocation resolves external state."""
    from skyulf.integrations.databricks import local_workflow

    history = Mock(side_effect=AssertionError("source history accessed"))
    monkeypatch.setattr(local_workflow, "_monthly_training_spec", history)
    with pytest.raises(ValueError, match="classification"):
        local_workflow.run_action(
            None,
            {
                "pipeline": _pipeline(),
                "target_column": "target",
                "cv_enabled": True,
                "cv_type": "stratified_k_fold",
                "score_model_selection": "champion",
                "promotion_policy": "manual_approval",
            },
            "train_monthly",
            experiment_name="unused",
            artifact_path=tmp_path,
        )
    history.assert_not_called()


def test_offline_cv_validation_uses_core_model_and_split_contract(workflow_config):
    """Editable Bundle settings must be accepted and invalid combinations rejected offline."""
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    config = {**workflow_config, "cv_enabled": True, "cv_folds": 2}
    assert validate_workflow_config(config, action="train") == config
    with pytest.raises(ValueError, match="classification"):
        validate_workflow_config({**config, "cv_type": "stratified_k_fold"}, action="train")
    with pytest.raises(ValueError, match="splitter"):
        validate_workflow_config(
            {
                **config,
                "pipeline": {
                    **_pipeline(),
                    "preprocessing": [
                        {"name": "split", "transformer": "TrainTestSplitter", "params": {}}
                    ],
                },
            },
            action="train",
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_cv_preserves_fixed_random_forest_parameters(monkeypatch, engine):
    """Basic CV cannot silently replace the user's hyperparameters with model defaults."""
    from skyulf.registry import NodeRegistry

    frame = pd.DataFrame({"x": range(40), "target": [0, 1] * 20})
    config = _pipeline(True)
    config["modeling"] = {
        "type": "random_forest_classifier",
        "params": {
            "n_estimators": 3,
            "max_depth": 2,
            "random_state": 19,
        },
    }
    calculator = cast(
        type[BaseModelCalculator], NodeRegistry.get_calculator("random_forest_classifier")
    )
    original = calculator.fit
    models = []

    def record_model(self, *args, **kwargs):
        """Inspect fitted estimators while exercising the real Core calculator."""
        model = original(self, *args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(calculator, "fit", record_model)
    native = pl.from_pandas(frame) if engine == "polars" else frame
    result = evaluate_training_cv(
        native,
        config,
        LocalCVSpec(enabled=True, folds=2, method="stratified_k_fold"),
        target_column="target",
    )
    assert result is not None
    assert len(models) == 2 and len(result["folds"]) == 2
    assert all(m.n_estimators == 3 and m.max_depth == 2 and m.random_state == 19 for m in models)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_temporal_cv_publishes_same_final_pipeline_and_separate_metrics(
    monkeypatch, tmp_path, engine
):
    """CV metadata and fold models cannot alter the saved pipeline or its final holdout scores."""
    import json
    from pathlib import Path

    import mlflow

    from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
    from skyulf.integrations.databricks import local_retraining

    frame = pd.DataFrame(
        {
            "id": range(30),
            "x": [float(i) for i in range(30)],
            "target": [float(i * 2 + 1) for i in range(30)],
            "event": pd.date_range("2026-01-01", periods=30, tz="UTC"),
        }
    )
    monkeypatch.setattr(local_retraining, "read_training_snapshot", lambda *args: frame)
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment("cv", artifact_location=(tmp_path / "runs").as_uri())
    spec = LocalTrainingSpec(
        table="workspace.test.source",
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=100,
        max_bytes=100000,
        split_strategy="temporal",
        event_column="event",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        holdout_start=datetime(2026, 1, 25, tzinfo=UTC),
        cutoff=datetime(2026, 2, 1, tzinfo=UTC),
    )
    predictions = []
    for enabled in (False, True):
        artifact_path = tmp_path / f"artifact_{enabled}"
        result = local_retraining.train_local_candidate(
            None,
            spec,
            _pipeline(),
            model_name="cv_model",
            tracking_uri=uri,
            registry_uri=uri,
            experiment_name="cv",
            run_name="cv",
            artifact_path=artifact_path,
            metric="heldout_rmse",
            min_improvement=0,
            quality_threshold=1,
            engine=engine,
            cv=LocalCVSpec(enabled=enabled, folds=3, method="time_series_split", shuffle=False),
        )
        artifact = load_local_pipeline(artifact_path)
        assert artifact.manifest.input_columns == ("x",)
        predictions.append(predict_local_pipeline(frame.loc[24:, ["x"]], artifact))
        metrics = client.get_run(result.run_id).data.metrics
        assert metrics["heldout_rmse"] == pytest.approx(0, abs=1e-8)
        assert ("cv_rmse_mean" in metrics) == enabled
        if enabled:
            path = client.download_artifacts(result.run_id, "cross_validation.json", str(tmp_path))
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            assert report["training_rows"] == 24
            assert report["fold_refit"]["max_fit_rows"] == 18
            assert report["dataset_id"] == result.dataset_id
    pd.testing.assert_frame_equal(predictions[0], predictions[1])
