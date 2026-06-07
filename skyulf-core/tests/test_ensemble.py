"""Tests for the Voting / Stacking ensemble model nodes."""

import numpy as np
import pandas as pd

from skyulf.registry import NodeRegistry


def _clf_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(60, 3)), columns=["a", "b", "c"]  # ty: ignore[invalid-argument-type]
    )
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))
    return X, y


def _reg_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(60, 3)), columns=["a", "b", "c"]  # ty: ignore[invalid-argument-type]
    )
    y = pd.Series(X["a"] * 2.0 + X["c"])
    return X, y


def test_ensemble_nodes_registered():
    for nid in (
        "voting_classifier",
        "stacking_classifier",
        "voting_regressor",
        "stacking_regressor",
    ):
        assert nid in NodeRegistry.get_all_metadata()
        assert NodeRegistry.get_calculator(nid)
        assert NodeRegistry.get_applier(nid)


def test_voting_classifier_soft_proba_in_range():
    X, y = _clf_data()
    calc = NodeRegistry.get_calculator("voting_classifier")()
    applier = NodeRegistry.get_applier("voting_classifier")()
    model = calc.fit(X, y, {"base_estimators": ["random_forest", "gaussian_nb"], "voting": "soft"})

    assert [name for name, _ in model.estimators] == ["random_forest", "gaussian_nb"]
    assert model.voting == "soft"

    proba = applier.predict_proba(X, model)
    assert proba is not None
    arr = proba.to_numpy()
    assert ((arr >= 0) & (arr <= 1)).all()


def test_voting_classifier_hard_predicts_valid_labels():
    X, y = _clf_data(1)
    calc = NodeRegistry.get_calculator("voting_classifier")()
    applier = NodeRegistry.get_applier("voting_classifier")()
    model = calc.fit(
        X, y, {"base_estimators": ["random_forest", "decision_tree"], "voting": "hard"}
    )

    preds = applier.predict(X, model)
    assert set(np.unique(preds)).issubset({0, 1})
    assert len(preds) == len(X)


def test_stacking_classifier_fits_with_cv_and_predicts():
    X, y = _clf_data(2)
    calc = NodeRegistry.get_calculator("stacking_classifier")()
    applier = NodeRegistry.get_applier("stacking_classifier")()
    model = calc.fit(
        X,
        y,
        {
            "base_estimators": ["random_forest", "decision_tree"],
            "final_estimator": "logistic_regression",
            "cv": 3,
        },
    )

    from sklearn.linear_model import LogisticRegression

    assert isinstance(model.final_estimator, LogisticRegression)
    assert model.cv == 3
    preds = applier.predict(X, model)
    assert len(preds) == len(X)


def test_voting_regressor_predicts_finite():
    X, y = _reg_data()
    calc = NodeRegistry.get_calculator("voting_regressor")()
    applier = NodeRegistry.get_applier("voting_regressor")()
    model = calc.fit(X, y, {"base_estimators": ["linear_regression", "random_forest"]})

    preds = applier.predict(X, model)
    assert len(preds) == len(X)
    assert np.isfinite(np.asarray(preds)).all()


def test_stacking_regressor_default_final_estimator():
    from sklearn.linear_model import Ridge

    X, y = _reg_data(3)
    calc = NodeRegistry.get_calculator("stacking_regressor")()
    model = calc.fit(X, y, {"cv": 3})

    # No final_estimator provided → falls back to the default (ridge).
    assert isinstance(model.final_estimator, Ridge)


def test_unknown_base_estimator_is_skipped():
    X, y = _clf_data(4)
    calc = NodeRegistry.get_calculator("voting_classifier")()
    model = calc.fit(X, y, {"base_estimators": ["random_forest", "does_not_exist"]})

    # The unknown key is dropped; only the valid one remains.
    names = [name for name, _ in model.estimators]
    assert names == ["random_forest"]


def test_empty_selection_falls_back_to_defaults():
    X, y = _clf_data(5)
    calc = NodeRegistry.get_calculator("voting_classifier")()
    model = calc.fit(X, y, {})

    # Empty config → the node still trains using its default base learners.
    names = [name for name, _ in model.estimators]
    assert names == ["random_forest", "logistic_regression", "gradient_boosting"]


# --- per-base-model hyperparameters -----------------------------------------


def test_per_base_fixed_params_applied():
    """A ``base_estimator_params`` map is pushed onto the matching base model."""
    X, y = _clf_data(6)
    calc = NodeRegistry.get_calculator("voting_classifier")()
    model = calc.fit(
        X,
        y,
        {
            "base_estimators": ["decision_tree", "gaussian_nb"],
            "base_estimator_params": {"decision_tree": {"max_depth": 2}},
        },
    )

    assert dict(model.estimators)["decision_tree"].max_depth == 2


def test_stacking_final_estimator_params_applied():
    """``final_estimator_params`` configure the stacking meta-learner."""
    X, y = _clf_data(7)
    calc = NodeRegistry.get_calculator("stacking_classifier")()
    model = calc.fit(
        X,
        y,
        {
            "base_estimators": ["random_forest", "decision_tree"],
            "final_estimator": "logistic_regression",
            "final_estimator_params": {"C": 0.25},
            "cv": 3,
        },
    )

    assert model.final_estimator.C == 0.25


def test_nested_keys_folded_into_base_params():
    """Flat ``name__param`` keys (as produced by tuning) reach the base model."""
    X, y = _clf_data(8)
    calc = NodeRegistry.get_calculator("voting_classifier")()
    model = calc.fit(
        X,
        y,
        {
            "base_estimators": ["decision_tree", "gaussian_nb"],
            "decision_tree__max_depth": 3,
        },
    )

    assert dict(model.estimators)["decision_tree"].max_depth == 3


# --- advanced (hyperparameter) tuning ---------------------------------------


def test_prepare_tuning_params_injects_estimators():
    """``default_params`` exposes resolved estimators once tuning is prepared."""
    calc = NodeRegistry.get_calculator("voting_classifier")()
    calc.prepare_tuning_params(
        {"base_estimators": ["random_forest", "decision_tree"], "voting": "hard"}
    )

    params = calc.default_params
    assert [name for name, _ in params["estimators"]] == ["random_forest", "decision_tree"]
    assert params["voting"] == "hard"


def test_build_ensemble_search_space_nested_keys():
    """Base learners expand into ``<name>__<param>`` keys; meta-params remain."""
    from skyulf.modeling.hyperparameters import build_ensemble_search_space

    space = build_ensemble_search_space(
        "stacking_classifier",
        ["random_forest", "decision_tree"],
        final_estimator="logistic_regression",
        strategy="grid",
        problem_type="classification",
    )

    assert "cv" in space  # ensemble meta-param
    assert any(k.startswith("random_forest__") for k in space)
    assert any(k.startswith("decision_tree__") for k in space)
    assert any(k.startswith("final_estimator__") for k in space)


def test_advanced_tuning_runs_and_applies_nested_params():
    """End-to-end: tuning a voting ensemble over a nested base-model param."""
    from skyulf.modeling._tuning.engine import TuningCalculator
    from skyulf.modeling._tuning.schemas import TuningConfig

    X, y = _clf_data(9)
    base_calc = NodeRegistry.get_calculator("voting_classifier")()
    base_calc.prepare_tuning_params(
        {"base_estimators": ["decision_tree", "gaussian_nb"], "tune_base_models": True}
    )
    tuner = TuningCalculator(base_calc)

    config = TuningConfig(
        strategy="grid",
        metric="accuracy",
        search_space={"decision_tree__max_depth": [2, 3]},
        cv_folds=3,
    )
    model, result = tuner.fit(X, y, config=config.__dict__)

    # The refit model is a real VotingClassifier built with the chosen bases…
    assert [name for name, _ in model.estimators] == ["decision_tree", "gaussian_nb"]
    # …and the tuned nested param was routed onto the base model.
    assert model.get_params()["decision_tree__max_depth"] in (2, 3)
    assert "decision_tree__max_depth" in result.best_params
