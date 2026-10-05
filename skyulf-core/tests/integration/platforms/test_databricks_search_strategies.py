"""Generated local search strategies must fit actual Core artifacts."""

import math

import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec
from skyulf.integrations.databricks.training.tuning.local_search import prepare_search_pipeline
from skyulf.integrations.databricks.training.tuning.local_search_results import (
    post_selection_cv,
    tuning_evidence,
    validate_search_membership,
)


def _rows() -> pd.DataFrame:
    """Keep search and holdout partitions observable with a tiny regression task."""
    return pd.DataFrame(
        {
            "x": [float(i) for i in range(32)],
            "target": [2.0 * i + (i % 3) * 0.1 for i in range(32)],
        }
    )


def _recipe(strategy: str, **changes) -> dict:
    """Mirror the compact selected-base-model setup emitted by the template."""
    modeling = {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "ridge_regression", "params": {"fit_intercept": True}},
        "strategy": strategy,
        "metric": "rmse",
        "search_space": {"alpha": [0.1, 1.0]},
        "n_trials": 2,
        "max_candidates": 2,
        "random_state": 19,
    }
    if strategy.startswith("halving"):
        modeling["strategy_params"] = {"factor": 2, "min_resources": 8, "max_resources": 24}
    if strategy == "optuna":
        modeling["strategy_params"] = {"sampler": "random", "pruner": "none", "pruning": False}
    modeling.update(changes)
    return {"preprocessing": [], "modeling": modeling}


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
def test_each_strategy_fits_a_persisted_selected_model(tmp_path, strategy: str) -> None:
    """Every exposed strategy must produce a finite saved result and prediction-ready model."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
        pytest.importorskip("optuna_integration")
    rows = _rows()
    training, holdout = rows.iloc[:28], rows.iloc[28:]
    cv = LocalCVSpec(enabled=True, folds=2)
    effective = prepare_search_pipeline(
        _recipe(strategy), cv, target_column="target", event_column=None
    )
    validate_search_membership(training, effective, cv, target_column="target", event_column=None)

    artifact = fit_local_workflow(
        effective,
        SplitDataset(train=training, test=holdout),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=40,
        max_bytes=100_000,
    )
    evidence = tuning_evidence(artifact)
    assert evidence is not None
    assert evidence["status"] == "completed"
    assert math.isfinite(evidence["best_score"])
    assert evidence["modeling"]["strategy"] == strategy
    assert artifact.pipeline.model_estimator is not None
    assert artifact.pipeline.model_estimator._unwrap_tuned_model().predict([[1.0]]).shape == (1,)


def test_auto_core_space_fits_small_linear_model(tmp_path) -> None:
    """An empty candidate object must resolve a usable bounded Core search space."""
    rows = _rows()
    cv = LocalCVSpec(enabled=True, folds=2)
    requested = _recipe(
        "grid",
        base_model={"type": "linear_regression", "params": {"n_jobs": 1}},
        search_space={},
        max_candidates=2,
    )
    effective = prepare_search_pipeline(requested, cv, target_column="target", event_column=None)
    assert effective["modeling"]["search_space"]["fit_intercept"] == [True, False]
    artifact = fit_local_workflow(
        effective,
        SplitDataset(train=rows.iloc[:28], test=rows.iloc[28:]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=40,
        max_bytes=100_000,
    )
    evidence = tuning_evidence(artifact)
    assert evidence is not None
    assert evidence["best_params"]["fit_intercept"] in (True, False)


def test_selected_voting_ensemble_fits_with_structural_learners(tmp_path) -> None:
    """A selected ensemble's learner structure must survive candidate fitting."""
    rows = _rows()
    cv = LocalCVSpec(enabled=True, folds=2)
    requested = _recipe(
        "grid",
        base_model={
            "type": "voting_regressor",
            "params": {
                "base_estimators": ["ridge", "lasso"],
            },
        },
        search_space={"weights": [None]},
        max_candidates=1,
    )
    effective = prepare_search_pipeline(requested, cv, target_column="target", event_column=None)
    artifact = fit_local_workflow(
        effective,
        SplitDataset(train=rows.iloc[:28], test=rows.iloc[28:]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=40,
        max_bytes=100_000,
    )
    evidence = tuning_evidence(artifact)
    assert evidence is not None
    assert evidence["modeling"]["base_model"]["params"]["base_estimators"] == ["ridge", "lasso"]
    assert math.isfinite(evidence["best_score"])


def test_nested_search_returns_saved_independent_outer_evaluation(tmp_path) -> None:
    """Nested setup must expose independent outer searches from the saved training artifact."""
    rows = _rows()
    training, holdout = rows.iloc[:28], rows.iloc[28:]
    cv = LocalCVSpec(enabled=True, folds=3, method="nested_cv")
    effective = prepare_search_pipeline(
        _recipe("grid"), cv, target_column="target", event_column=None
    )
    validate_search_membership(training, effective, cv, target_column="target", event_column=None)
    artifact = fit_local_workflow(
        effective,
        SplitDataset(train=training, test=holdout),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=40,
        max_bytes=100_000,
    )
    report = post_selection_cv(training, artifact, cv, target_column="target")
    assert report is not None
    assert report["status"] == "nested_cv"
    assert report["cv_config"]["method"] == "nested_cv"
    assert report["outer_folds"] == 3 and report["inner_folds"] == 2
    assert len(report["folds"]) == 3
    evidence = tuning_evidence(artifact)
    assert evidence is not None and report == evidence["nested_cv"]
    assert report["aggregated_metrics"]


def test_cv_disabled_preserves_single_training_only_search(tmp_path) -> None:
    """The default CV-off mode still selects a model on one training split."""
    rows = _rows()
    training, holdout = rows.iloc[:28], rows.iloc[28:]
    cv = LocalCVSpec(enabled=False)
    effective = prepare_search_pipeline(
        _recipe("grid"), cv, target_column="target", event_column=None
    )
    validate_search_membership(training, effective, cv, target_column="target", event_column=None)
    artifact = fit_local_workflow(
        effective,
        SplitDataset(train=training, test=holdout),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=40,
        max_bytes=100_000,
    )
    evidence = tuning_evidence(artifact)
    assert evidence is not None
    assert evidence["modeling"]["cv_enabled"] is False
    assert post_selection_cv(training, artifact, cv, target_column="target") is None
