"""Pruning capability must describe the same training work that can be skipped."""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.sklearn_wrapper import SklearnCalculator


@pytest.mark.parametrize("preprocessing", [False, True])
def test_forest_supports_fold_pruning_only_with_multiple_search_folds(preprocessing):
    """A normal fit may skip later CV folds, but cannot stop a single holdout fit."""
    tuner = TuningCalculator(SklearnCalculator(RandomForestClassifier, {}, "classification"))
    config = TuningConfig(strategy="optuna")
    support = tuner.pruning_support(config, n_splits=3, preprocessing=preprocessing)
    assert support["supported"] is True
    assert support["mode"] == "folds"
    single = tuner.pruning_support(config, n_splits=1, preprocessing=preprocessing)
    assert single["supported"] is False
    assert "holdout" in single["reason"]


def test_sgd_budget_search_and_preprocessing_fall_back_to_fold_pruning():
    """A searched epoch budget can use ordinary fit while retaining between-fold pruning."""
    tuner = TuningCalculator(SklearnCalculator(SGDClassifier, {"max_iter": 5}, "classification"))
    config = TuningConfig(strategy="optuna", search_space={"max_iter": [2, 5]})
    assert tuner.pruning_support(config, n_splits=2)["mode"] == "folds"
    direct = TuningConfig(strategy="optuna")
    assert tuner.pruning_support(direct, n_splits=2)["mode"] == "iterations"
    assert tuner.pruning_support(direct, n_splits=2, preprocessing=True)["mode"] == "folds"


@pytest.mark.parametrize("family", ["xgboost", "lightgbm"])
def test_boosting_with_fold_preprocessing_exposes_iteration_pruning(family):
    """Native callbacks remain available when each fold owns its preprocessing."""
    library = pytest.importorskip(family)
    model = library.XGBClassifier if family == "xgboost" else library.LGBMClassifier
    tuner = TuningCalculator(SklearnCalculator(model, {"n_estimators": 5}, "classification"))
    config = TuningConfig(strategy="optuna", search_space={"n_estimators": [3, 7]})
    support = tuner.pruning_support(config, n_splits=1, preprocessing=True)
    assert support["supported"] is True
    assert support["mode"] == "iterations"


def test_generic_pipeline_keeps_fold_pruning_without_unwrapping_its_transformers():
    """Only known fold adapters may be prepared specially for native callbacks."""
    from skyulf.modeling.pruning import resolve_pruning_plan

    model = Pipeline([("scale", StandardScaler()), ("model", SGDClassifier(max_iter=3))])
    plan = resolve_pruning_plan(model, {}, n_splits=3)
    assert plan.mode == "folds"
    assert plan.kind is None


def test_none_selection_does_not_disable_capability_but_explicit_opt_out_does():
    """The dropdown must remain selectable after None while respecting the legacy opt-out."""
    tuner = TuningCalculator(SklearnCalculator(RandomForestClassifier, {}, "classification"))
    config = TuningConfig(strategy="optuna", strategy_params={"pruner": "none"})
    assert tuner.pruning_support(config)["mode"] == "folds"
    config.strategy_params["pruning"] = False
    result = tuner.pruning_support(config)
    assert result["supported"] is False
    assert "pruning=False" in result["reason"]


@pytest.mark.parametrize("distribution", ["categorical", "integer"])
def test_native_distribution_budget_reserves_distinct_steps_for_each_fold(distribution):
    """Programmatic bounded distributions need the same common stride as Canvas lists."""
    optuna = pytest.importorskip("optuna")
    xgboost = pytest.importorskip("xgboost")
    from skyulf.modeling.pruning import resolve_pruning_plan

    values = (
        optuna.distributions.CategoricalDistribution([2, 7])
        if distribution == "categorical"
        else optuna.distributions.IntDistribution(2, 7)
    )
    plan = resolve_pruning_plan(xgboost.XGBClassifier(), {"n_estimators": values}, n_splits=3)
    assert plan.mode == "iterations"
    assert plan.iteration_budget == 7


@pytest.mark.parametrize("alias", ["num_iterations", "n_iter"])
@pytest.mark.parametrize("searched", [False, True])
def test_alternate_lightgbm_round_budget_falls_back_to_complete_folds(alias, searched):
    """Unknown native round precedence must not reuse intermediate report steps."""
    lightgbm = pytest.importorskip("lightgbm")
    from skyulf.modeling.pruning import resolve_pruning_plan

    model = lightgbm.LGBMClassifier(**({} if searched else {alias: 7}))
    space = {alias: [7]} if searched else {}
    assert resolve_pruning_plan(model, space, n_splits=3).mode == "folds"
    assert resolve_pruning_plan(model, space, n_splits=1).mode == "none"


def test_bare_xgboost_default_budget_can_prune_a_single_holdout():
    """Core callers using XGBoost's implicit round default retain native checkpoints."""
    xgboost = pytest.importorskip("xgboost")
    from skyulf.modeling.pruning import resolve_pruning_plan

    model = xgboost.XGBClassifier()
    plan = resolve_pruning_plan(model, {}, n_splits=1)
    assert plan.mode == "iterations"
    assert plan.iteration_budget == model.get_num_boosting_rounds()
