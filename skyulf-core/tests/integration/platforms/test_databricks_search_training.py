"""Bundle search selects models using only the bounded training partition."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config
from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
from skyulf.integrations.databricks.training.fitting import local_retraining as training
from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec


def _search_model(task="regression", strategy="grid"):
    """Use two small candidates and an explicit fixed tree count for meaningful checks."""
    return {
        "type": "hyperparameter_tuner",
        "base_model": {
            "type": "random_forest_classifier"
            if task == "classification"
            else "random_forest_regressor",
            "params": {"n_estimators": 3, "n_jobs": 1},
        },
        "strategy": strategy,
        "search_space": {"max_depth": [2, 4]},
        "metric": "accuracy" if task == "classification" else "rmse",
        "n_trials": 2,
        "random_state": 17,
    }


def test_workflow_accepts_tuner_with_shared_cv_and_preserves_request(workflow_config):
    """Advanced setup must accept the Core wrapper without mutating its saved request."""
    workflow_config["pipeline"]["modeling"] = _search_model()
    workflow_config.update(cv_enabled=True, cv_folds=2, cv_type="k_fold")
    original = deepcopy(workflow_config)
    checked = validate_workflow_config(workflow_config, action="train")
    assert checked == workflow_config == original


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_temporal_tuner_artifact_does_not_require_ordering_metadata(tmp_path, engine):
    """A training-only time column must not become a required prediction feature."""
    frame = pd.DataFrame(
        {
            "x": np.arange(24, dtype=float),
            "event": pd.date_range("2026-01-01", periods=24, tz="UTC"),
            "target": np.arange(24, dtype=float) * 2,
        }
    )
    model = _search_model()
    model.update(
        cv_enabled=True,
        cv_type="time_series_split",
        cv_folds=2,
        cv_shuffle=False,
        cv_time_column="event",
    )
    config = {"preprocessing": [], "modeling": model}
    native = pl.from_pandas(frame) if engine == "polars" else frame
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=50,
        max_bytes=100000,
    )
    assert artifact.manifest.input_columns == ("x",)
    inputs = native.select("x") if isinstance(native, pl.DataFrame) else native[["x"]]
    loaded = load_local_pipeline(tmp_path / "artifact")
    pd.testing.assert_frame_equal(
        predict_local_pipeline(inputs, artifact), predict_local_pipeline(inputs, loaded)
    )


def test_pipeline_tuner_preserves_selected_ensemble_members(tmp_path):
    """Ensemble tuning must use the requested learners rather than calculator defaults."""
    frame = pd.DataFrame({"x": np.arange(24, dtype=float), "target": np.arange(24) * 2.0})
    model = _search_model()
    model.update(
        base_model={
            "type": "voting_regressor",
            "params": {"base_estimators": ["linear_regression", "ridge"]},
        },
        search_space={"ridge__alpha": [0.1, 1.0]},
        cv_enabled=True,
        cv_folds=2,
    )
    artifact = fit_local_workflow(
        {"preprocessing": [], "modeling": model},
        SplitDataset(train=frame, test=frame.head(0)),
        target_column="target",
        artifact_path=tmp_path / "ensemble",
        max_rows=50,
        max_bytes=100000,
    )
    assert artifact.pipeline.model_estimator is not None
    estimator = artifact.pipeline.model_estimator._unwrap_tuned_model()
    assert [name for name, _ in estimator.estimators] == ["linear_regression", "ridge"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("cv_enabled", [True, False])
def test_candidate_search_isolates_holdout_and_logs_selected_artifact(
    monkeypatch, tmp_path, engine, task, cv_enabled
):
    """Search folds, final refit and saved predictions must share one training-only recipe."""
    from skyulf.pipeline._pipeline import _PipelineTuningPreprocessor

    source = pd.DataFrame({"id": range(40), "x": np.arange(40, dtype=float)})
    source["target"] = (source.x % 2).astype(int) if task == "classification" else source.x * 2
    spec = training.LocalTrainingSpec(
        table="workspace.test.search",
        version=1,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=50,
        max_bytes=100000,
    )
    pipeline = {
        "preprocessing": [
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}}
        ],
        "modeling": _search_model(task, "random"),
    }
    cv = LocalCVSpec(enabled=cv_enabled, folds=2)
    effective = training.candidate_config(
        spec,
        pipeline,
        engine=engine,
        cv=cv,
        metric="heldout_accuracy" if task == "classification" else "heldout_rmse",
        min_improvement=0,
        champion_version=None,
        quality_threshold=None,
        risk_category=None,
    )
    monkeypatch.setattr(training, "read_training_snapshot", lambda *_: source.copy())
    snapshots = []
    original_fit = _PipelineTuningPreprocessor.fit_transform

    def capture_fit(self, features, labels):
        """Observe raw membership at every candidate fold and final refit."""
        snapshots.append(set(features["x"].to_numpy()))
        return original_fit(self, features, labels)

    monkeypatch.setattr(_PipelineTuningPreprocessor, "fit_transform", capture_fit)
    documents = {}

    def log_dict(run_id, value, filename):
        """Require portable evidence rather than accepting NaN or unpickled objects."""
        documents[filename] = json.loads(json.dumps(value, allow_nan=False))

    run = SimpleNamespace(
        run_id="test",
        client=SimpleNamespace(log_dict=log_dict),
        log_config=Mock(),
        log_params=Mock(),
        log_metrics=Mock(),
        set_tags=Mock(),
    )
    fitted = training.fit_candidate(
        object(),
        spec,
        pipeline,
        run=run,
        pipeline_config=effective,
        artifact_path=tmp_path / "trained",
        engine=engine,
        cv=cv,
        risk_category=None,
    )
    training.log_fitted_candidate(run, fitted, pipeline, engine=engine, risk_category=None)
    train, heldout, _ = training.split_labeled_snapshot(source, spec, engine=engine)
    train_values, heldout_values = set(train.x), set(heldout.x)
    assert snapshots[-1] == train_values
    assert all(values <= train_values and not values & heldout_values for values in snapshots)
    assert any(len(values) < len(train_values) for values in snapshots)
    assert fitted.cv_results is None
    evidence = documents["tuning.json"]
    assert evidence["n_trials"] == 2
    assert evidence["best_params"]["n_estimators"] == 3
    logged_params = {
        key: value for call in run.log_params.call_args_list for key, value in call.args[0].items()
    }
    assert logged_params["tuning_requested_trials"] == 2
    assert logged_params["tuning_random_state"] == 17
    assert logged_params["tuning_requested_metric"] == pipeline["modeling"]["metric"]
    assert logged_params["tuning_best_params.n_estimators"] == "3"
    assert int(logged_params["tuning_best_params.max_depth"]) in (2, 4)
    assert json.loads(logged_params["tuning_search_space"])["max_depth"] == [2, 4]
    assert logged_params["model_type"] == pipeline["modeling"]["base_model"]["type"]
    assert logged_params["model_params.n_estimators"] == "3"
    assert logged_params["model_params.max_depth"] == logged_params["tuning_best_params.max_depth"]
    assert logged_params["split_strategy"] == "random"
    assert logged_params["split_test_size"] == "0.2"
    assert logged_params["split_random_state"] == "42"
    assert json.loads(logged_params["preprocessing_steps"]) == ["scale"]
    assert json.loads(logged_params["pre_split_steps"]) == []
    assert documents["training_parameters.json"]["model_params"]["n_estimators"] == 3
    assert "cross_validation.json" not in documents
    model = fitted.artifact.pipeline.model_estimator._unwrap_tuned_model()
    assert model.n_estimators == 3
    native = pl.from_pandas(heldout[["x"]]) if engine == "polars" else heldout[["x"]]
    reloaded = load_local_pipeline(tmp_path / "trained")
    pd.testing.assert_frame_equal(
        predict_local_pipeline(native, fitted.artifact), predict_local_pipeline(native, reloaded)
    )
