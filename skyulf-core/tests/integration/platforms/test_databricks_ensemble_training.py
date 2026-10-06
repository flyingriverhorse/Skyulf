"""Selected Core ensembles fit through the bounded Databricks search route."""

import itertools
import json
import math
from unittest.mock import Mock

import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow
from skyulf.integrations.databricks.training.fitting.local_retraining import LocalTrainingSpec
from skyulf.integrations.databricks.training.shared.training_parameters import (
    log_training_parameters,
)
from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec
from skyulf.integrations.databricks.training.tuning.local_search import prepare_search_pipeline
from skyulf.integrations.databricks.training.tuning.local_search_results import (
    post_selection_cv,
    tuning_evidence,
)

_FAMILIES = ("voting_classifier", "stacking_classifier", "voting_regressor", "stacking_regressor")
_STRATEGIES = ("grid", "random", "halving_grid", "halving_random", "optuna")


def test_nested_cv_ensemble_returns_independent_search_evidence(tmp_path, monkeypatch):
    """Nested ensemble reports reuse completed outer searches without another training pass."""
    frame = _rows(False)
    cv = LocalCVSpec(enabled=True, folds=2, method="nested_cv")
    config = prepare_search_pipeline(
        _search("stacking_regressor", "grid"), cv, target_column="target", event_column=None
    )
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=frame, test=frame.head(0)),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=50,
        max_bytes=100000,
    )
    import skyulf.integrations.databricks.training.tuning.local_search_results as results

    def unexpected_fit(*args, **kwargs):
        """Fail if reporting retrains the already evaluated artifact."""
        pytest.fail("Nested evidence reporting must not refit a model")

    monkeypatch.setattr(results, "evaluate_training_cv", unexpected_fit)
    report = post_selection_cv(frame.head(0), artifact, cv, target_column="target")
    assert report is not None
    assert report["status"] == "nested_cv"
    assert report["aggregated_metrics"]
    assert report["outer_folds"] == 2
    assert len(report["folds"]) == 2
    saved = tuning_evidence(load_local_pipeline(tmp_path / "artifact"))
    assert saved is not None
    assert report == saved["nested_cv"]
    assert artifact.pipeline.config["modeling"]["base_model"]["params"]["tune_base_models"] is True


def _rows(classification: bool) -> pd.DataFrame:
    """Keep every candidate fold and the final holdout small but viable."""
    x = list(range(40))
    target = [i % 2 for i in x] if classification else [2.0 * i + (i % 3) * 0.1 for i in x]
    return pd.DataFrame({"x": [float(i) for i in x], "target": target})


def _model(family: str) -> tuple[dict, dict]:
    """Choose two cheap base learners and one real nested tuning axis."""
    if family == "voting_classifier":
        return (
            {
                "base_estimators": ["logistic_regression", "gaussian_nb"],
                "base_estimator_params": {"logistic_regression": {"max_iter": 100}},
                "voting": "soft",
                "weights": {"gaussian_nb": 1, "logistic_regression": 3},
            },
            {"logistic_regression__C": [0.5, 1.0]},
        )
    if family == "stacking_classifier":
        return (
            {
                "base_estimators": ["logistic_regression", "gaussian_nb"],
                "final_estimator": "logistic_regression",
                "cv": 2,
                "passthrough": True,
                "base_estimator_params": {"logistic_regression": {"max_iter": 100}},
                "final_estimator_params": {"max_iter": 100},
            },
            {"final_estimator__C": [0.5, 1.0]},
        )
    if family == "voting_regressor":
        return (
            {
                "base_estimators": ["linear_regression", "ridge"],
                "weights": [2, 1],
                "base_estimator_params": {"ridge": {"fit_intercept": False}},
            },
            {"ridge__alpha": [0.5, 1.0]},
        )
    return (
        {
            "base_estimators": ["linear_regression", "ridge"],
            "final_estimator": "ridge",
            "cv": 2,
            "passthrough": True,
            "base_estimator_params": {"ridge": {"fit_intercept": False}},
            "final_estimator_params": {"fit_intercept": False},
        },
        {"final_estimator__alpha": [0.5, 1.0]},
    )


def _search(family: str, strategy: str) -> dict:
    """Build the same selected-base-model wrapper accepted by local preflight."""
    params, space = _model(family)
    modeling = {
        "type": "hyperparameter_tuner",
        "base_model": {"type": family, "params": params},
        "strategy": strategy,
        "metric": "accuracy" if family.endswith("classifier") else "rmse",
        "search_space": space,
        "n_trials": 2,
        "max_candidates": 2,
        "random_state": 17,
    }
    if strategy.startswith("halving"):
        modeling["strategy_params"] = {"factor": 2, "min_resources": 12, "max_resources": 32}
    if strategy == "optuna":
        modeling["strategy_params"] = {"sampler": "random", "pruner": "none", "pruning": False}
    return {"preprocessing": [], "modeling": modeling}


@pytest.mark.parametrize("family,strategy", tuple(itertools.product(_FAMILIES, _STRATEGIES)))
def test_all_ensemble_families_and_search_strategies_fit_and_replay(
    tmp_path, family: str, strategy: str
) -> None:
    """Every generated ensemble route must save a prediction-ready selected model."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
        pytest.importorskip("optuna_integration")
    classification = family.endswith("classifier")
    frame = _rows(classification)
    cv = LocalCVSpec(
        enabled=True, folds=2, method="stratified_k_fold" if classification else "k_fold"
    )
    config = prepare_search_pipeline(
        _search(family, strategy), cv, target_column="target", event_column=None
    )
    fit_local_workflow(
        config,
        SplitDataset(train=frame.iloc[:32], test=frame.iloc[32:]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=50,
        max_bytes=100_000,
    )
    restored = load_local_pipeline(tmp_path / "artifact")
    query = frame.loc[[32, 33], ["x"]]
    actual = predict_local_pipeline(query, restored)
    evidence = tuning_evidence(restored)

    assert len(actual) == 2
    assert actual["prediction"].notna().all()
    assert evidence is not None
    assert math.isfinite(evidence["best_score"])
    assert evidence["modeling"]["base_model"]["type"] == family
    estimator = restored.pipeline.model_estimator
    assert estimator is not None
    model = estimator._unwrap_tuned_model()
    if classification:
        assert model.named_estimators_["logistic_regression"].max_iter == 100
    else:
        assert model.named_estimators_["ridge"].fit_intercept is False
    if family.startswith("stacking"):
        assert model.cv == 2
        assert model.passthrough is True
        if classification:
            assert model.final_estimator_.max_iter == 100
        else:
            assert model.final_estimator_.fit_intercept is False
    else:
        assert list(model.weights) == ([3, 1] if classification else [2, 1])
    run = Mock()
    spec = LocalTrainingSpec(
        table="workspace.test.source",
        version=1,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=50,
        max_bytes=100_000,
    )
    log_training_parameters(run, restored, spec, config)
    params = run.log_params.call_args.args[0]
    assert params["model_type"] == family
    assert json.loads(params["ensemble.base_models"]) == [name for name, _ in model.estimators]
    for name, value in evidence["best_params"].items():
        assert json.loads(params[f"model_params.{name}"]) == value
    if family.startswith("stacking"):
        assert params["ensemble.cv"] == "2"
        assert params["ensemble.final_estimator"] == type(model.final_estimator_).__name__
    else:
        assert json.loads(params["ensemble.weights"]) == list(model.weights)


@pytest.mark.parametrize("family", _FAMILIES)
def test_polars_selected_ensemble_replays_pandas_prediction(tmp_path, family: str) -> None:
    """Polars fitted ensemble artifacts must retain their trained engine and feature schema."""
    classification = family.endswith("classifier")
    frame = _rows(classification)
    native = pl.from_pandas(frame)
    cv = LocalCVSpec(
        enabled=True, folds=2, method="stratified_k_fold" if classification else "k_fold"
    )
    config = prepare_search_pipeline(
        _search(family, "grid"), cv, target_column="target", event_column=None
    )
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=native.slice(0, 32), test=native.slice(32, 8)),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=50,
        max_bytes=100_000,
    )
    result = predict_local_pipeline(pd.DataFrame({"x": [40.0]}), artifact)
    assert artifact.manifest.fitted_engine == "polars"
    assert len(result) == 1


def test_weighted_hard_vote_saves_prediction_only_artifact(tmp_path) -> None:
    """Hard voting must survive save/load even though it has no probability output."""
    frame = _rows(True)
    requested = _search("voting_classifier", "grid")
    requested["modeling"]["base_model"]["params"]["voting"] = "hard"
    cv = LocalCVSpec(enabled=True, folds=2, method="stratified_k_fold")
    config = prepare_search_pipeline(requested, cv, target_column="target", event_column=None)
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=frame.iloc[:32], test=frame.iloc[32:]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=50,
        max_bytes=100_000,
    )
    restored = load_local_pipeline(tmp_path / "artifact")
    result = predict_local_pipeline(frame.loc[[32, 33], ["x"]], restored)
    assert list(result.columns) == ["prediction"]
    assert set(result["prediction"]).issubset({0, 1})
    assert artifact.manifest.classes == (0, 1)


def test_calibrated_soft_vote_preserves_base_settings(tmp_path) -> None:
    """Calibration wraps each fixed base learner while retaining its pinned parameters."""
    frame = _rows(True)
    requested = _search("voting_classifier", "grid")
    params = requested["modeling"]["base_model"]["params"]
    params.update(calibrate_base_models=True, calibration_method="sigmoid", calibration_cv=2)
    requested["modeling"]["search_space"] = {"logistic_regression__estimator__C": [0.5, 1.0]}
    cv = LocalCVSpec(enabled=True, folds=2, method="stratified_k_fold")
    config = prepare_search_pipeline(requested, cv, target_column="target", event_column=None)
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=frame.iloc[:32], test=frame.iloc[32:]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=50,
        max_bytes=100_000,
    )
    estimator = artifact.pipeline.model_estimator
    assert estimator is not None
    model = estimator._unwrap_tuned_model()
    assert model.named_estimators_["logistic_regression"].method == "sigmoid"
    assert model.named_estimators_["logistic_regression"].cv == 2
    assert model.named_estimators_["logistic_regression"].estimator.max_iter == 100
    assert "probability_1" in predict_local_pipeline(frame.loc[[32], ["x"]], artifact)


@pytest.mark.parametrize("family", ["stacking_classifier", "stacking_regressor"])
def test_stacking_internal_cv_is_distinct_from_shared_search_cv(tmp_path, family: str) -> None:
    """The stack's two meta folds and search's three selection folds stay separate."""
    frame = _rows(family.endswith("classifier"))
    cv = LocalCVSpec(
        enabled=True,
        folds=3,
        method="stratified_k_fold" if family.endswith("classifier") else "k_fold",
    )
    config = prepare_search_pipeline(
        _search(family, "grid"), cv, target_column="target", event_column=None
    )
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=frame.iloc[:32], test=frame.iloc[32:]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=50,
        max_bytes=100_000,
    )
    estimator = artifact.pipeline.model_estimator
    assert estimator is not None
    model = estimator._unwrap_tuned_model()
    assert config["modeling"]["cv_folds"] == 3
    assert model.cv == 2


def test_auto_base_tuning_keeps_fixed_nested_parameter(tmp_path) -> None:
    """Automatic nested axes cannot overwrite a selected learner's fixed alpha."""
    frame = _rows(False)
    requested = _search("voting_regressor", "random")
    params = requested["modeling"]["base_model"]["params"]
    params["tune_base_models"] = True
    params["base_estimator_params"]["ridge"]["alpha"] = 0.4
    requested["modeling"]["search_space"] = {}
    requested["modeling"]["n_trials"] = 1
    cv = LocalCVSpec(enabled=True, folds=2)
    config = prepare_search_pipeline(requested, cv, target_column="target", event_column=None)
    artifact = fit_local_workflow(
        config,
        SplitDataset(train=frame.iloc[:32], test=frame.iloc[32:]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=50,
        max_bytes=100_000,
    )
    estimator = artifact.pipeline.model_estimator
    assert estimator is not None
    model = estimator._unwrap_tuned_model()
    assert model.named_estimators_["ridge"].alpha == 0.4


@pytest.mark.parametrize("voting,calibrate", [("soft", False), ("hard", False), ("soft", True)])
def test_selected_sgd_classifier_is_fitted_in_voting_ensemble(
    tmp_path, voting: str, calibrate: bool
) -> None:
    """Canvas-selected SGD must be fitted rather than silently skipped by Core."""
    frame = _rows(True)
    requested = _search("voting_classifier", "grid")
    params = requested["modeling"]["base_model"]["params"]
    params["base_estimators"] = ["sgd_classifier", "gaussian_nb"]
    params["base_estimator_params"] = {"sgd_classifier": {"max_iter": 100}}
    params["weights"] = [2, 1]
    params["voting"] = voting
    if calibrate:
        params.update(calibrate_base_models=True, calibration_method="sigmoid", calibration_cv=2)
    prefix = "sgd_classifier__estimator__" if calibrate else "sgd_classifier__"
    requested["modeling"]["search_space"] = {f"{prefix}alpha": [0.0001, 0.001]}
    cv = LocalCVSpec(enabled=True, folds=2, method="stratified_k_fold")
    config = prepare_search_pipeline(requested, cv, target_column="target", event_column=None)
    fit_local_workflow(
        config,
        SplitDataset(train=frame.iloc[:32], test=frame.iloc[32:]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=50,
        max_bytes=100_000,
    )
    restored = load_local_pipeline(tmp_path / "artifact")
    estimator = restored.pipeline.model_estimator
    assert estimator is not None
    model = estimator._unwrap_tuned_model()
    selected = model.named_estimators_["sgd_classifier"]
    sgd = selected.estimator if calibrate else selected
    result = predict_local_pipeline(frame.loc[[32, 33], ["x"]], restored)
    assert [name for name, _ in model.estimators] == ["sgd_classifier", "gaussian_nb"]
    assert sgd.loss == "log_loss"
    assert sgd.max_iter == 100
    if voting == "soft":
        assert "probability_1" in result.columns
    else:
        assert list(result.columns) == ["prediction"]


def test_sgd_automatic_base_tuning_has_nested_axis() -> None:
    """Automatic base tuning must include the SGD registry space."""
    requested = _search("voting_classifier", "random")
    params = requested["modeling"]["base_model"]["params"]
    params["base_estimators"] = ["sgd_classifier", "gaussian_nb"]
    params["base_estimator_params"] = {"sgd_classifier": {"max_iter": 100}}
    params["weights"] = [2, 1]
    params["tune_base_models"] = True
    requested["modeling"]["search_space"] = {}
    cv = LocalCVSpec(enabled=True, folds=2, method="stratified_k_fold")
    config = prepare_search_pipeline(requested, cv, target_column="target", event_column=None)
    assert "sgd_classifier__alpha" in config["modeling"]["search_space"]
