"""Fixed nested CV must evaluate the same recipe as ordinary model fitting."""

import json
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from skyulf.modeling.cross_validation import perform_cross_validation
from skyulf.registry import NodeRegistry


def _regression_rows() -> tuple[pd.DataFrame, pd.Series]:
    """Supply nonlinear data that distinguishes the fixed ensemble and ridge settings."""
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(72, 3)), columns=list("abc"))
    y = pd.Series(4 * X.a**2 + 2 * X.b + rng.normal(size=72), name="target")
    return X, y


def _nested_report(calculator: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Run the public fixed-model CV entrypoint with small deterministic folds."""
    X, y = _regression_rows()
    return perform_cross_validation(
        calculator,
        NodeRegistry.get_applier("ridge_regression")(),
        X,
        y,
        config,
        cv_type="nested_cv",
        cv_nested_type="k_fold",
        n_folds=3,
        inner_folds=2,
        random_state=11,
    )


def _assert_fixed_scores(report: dict[str, Any], estimator: Any) -> None:
    """Compare inner and outer scores with fresh independent sklearn fits."""
    X, y = _regression_rows()
    outer_scores = []
    for fold, (train, test) in zip(
        report["folds"], KFold(3, shuffle=True, random_state=11).split(X), strict=True
    ):
        train_x, train_y = X.iloc[train], y.iloc[train]
        model = clone(estimator).fit(train_x, train_y)
        score = -mean_squared_error(y.iloc[test], model.predict(X.iloc[test]))
        outer_scores.append(score)
        inner_scores = []
        for inner_train, inner_test in KFold(2, shuffle=True, random_state=11).split(train_x):
            model = clone(estimator).fit(train_x.iloc[inner_train], train_y.iloc[inner_train])
            inner_scores.append(
                -mean_squared_error(
                    train_y.iloc[inner_test], model.predict(train_x.iloc[inner_test])
                )
            )
        assert fold["outer_score"] == pytest.approx(score)
        assert fold["inner_best_score"] == pytest.approx(np.mean(inner_scores))
        assert fold["n_trials"] == 1
    assert report["mean_score"] == pytest.approx(np.mean(outer_scores))
    assert report["fixed_parameters"] is True
    assert report["total_trials"] == 4
    assert json.loads(json.dumps(report))["mean_score"] == report["mean_score"]


@pytest.mark.parametrize(
    ("config", "alpha", "fit_intercept"),
    [
        ({"alpha": 999.0, "fit_intercept": False}, 999.0, False),
        ({"params": {"alpha": 999.0, "fit_intercept": False}}, 999.0, False),
        ({"type": "ridge_regression", "node_id": "ridge", "alpha": 999.0}, 999.0, True),
        ({"alpha": 999.0, "params": {"alpha": 7.0}}, 7.0, True),
        ({"alpha": 999.0, "params": {}}, 1.0, True),
    ],
)
def test_nested_cv_preserves_plain_model_config(
    config: dict[str, Any], alpha: float, fit_intercept: bool
) -> None:
    """Flat overrides and nested precedence must survive both levels of fixed CV."""
    original = deepcopy(config)
    calculator = NodeRegistry.get_calculator("ridge_regression")()
    report = _nested_report(calculator, config)
    _assert_fixed_scores(report, Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42))
    assert config == original


@pytest.mark.parametrize("family", ["voting_regressor", "stacking_regressor"])
@pytest.mark.parametrize("nested", [False, True])
def test_nested_cv_preserves_ensemble_structure(family: str, nested: bool) -> None:
    """Selected learners, weights, nested overrides and final learners must all affect scores."""
    params: dict[str, Any] = {
        "base_estimators": ["ridge", "decision_tree"],
        "base_estimator_params": {"ridge": {"alpha": 13.0}, "decision_tree": {"max_depth": 1}},
        "decision_tree__max_depth": 3,
        "n_jobs": 1,
    }
    estimators = [
        ("ridge", Ridge(alpha=13.0)),
        ("decision_tree", DecisionTreeRegressor(max_depth=3, random_state=42)),
    ]
    if family == "stacking_regressor":
        params.update(
            final_estimator="ridge",
            final_estimator_params={"alpha": 23.0},
            final_estimator__fit_intercept=False,
            cv=2,
            passthrough=True,
        )
        estimator: Any = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=23.0, fit_intercept=False),
            cv=2,
            passthrough=True,
            n_jobs=1,
        )
    else:
        params["weights"] = [1.0, 4.0]
        estimator = VotingRegressor(estimators=estimators, weights=[1.0, 4.0], n_jobs=1)
    config = {"params": params} if nested else params
    original = deepcopy(config)
    calculator = NodeRegistry.get_calculator(family)()
    report = _nested_report(calculator, config)
    _assert_fixed_scores(report, estimator)
    assert config == original
    assert calculator._tuning_base_config == {}


@pytest.mark.parametrize(
    ("config", "class_weight"),
    [
        ({"max_depth": 1, "class_weight": "balanced"}, "balanced"),
        (
            {"params": {"max_depth": 1, "class_weight": {"high": 9.0, "low": 1.0}}},
            {"high": 9.0, "low": 1.0},
        ),
        ({"max_depth": 1, "class_weight": {"high": 9.0, "low": 1.0}}, None),
    ],
)
def test_nested_cv_preserves_class_weight_config_contract(
    config: dict[str, Any], class_weight: Any
) -> None:
    """Nested class weights apply while legacy flat dictionary values remain excluded."""
    X, values = _regression_rows()
    y = pd.Series(np.where(values > values.quantile(0.75), "high", "low"), name="target")
    calculator = NodeRegistry.get_calculator("decision_tree_classifier")()
    report = perform_cross_validation(
        calculator,
        NodeRegistry.get_applier("decision_tree_classifier")(),
        X,
        y,
        config,
        cv_type="nested_cv",
        cv_nested_type="stratified_k_fold",
        n_folds=3,
        inner_folds=2,
        random_state=11,
    )
    scores = []
    for train, test in StratifiedKFold(3, shuffle=True, random_state=11).split(X, y):
        model = DecisionTreeClassifier(max_depth=1, class_weight=class_weight, random_state=42)
        model.fit(X.iloc[train], y.iloc[train])
        scores.append(accuracy_score(y.iloc[test], model.predict(X.iloc[test])))
    assert [fold["outer_score"] for fold in report["folds"]] == pytest.approx(scores)


def test_nested_cv_does_not_change_prepared_ensemble_recipe() -> None:
    """Evaluating a different recipe must leave a caller's previously selected learners intact."""
    X, y = _regression_rows()
    calculator = NodeRegistry.get_calculator("voting_regressor")()
    calculator.prepare_tuning_params(
        {
            "params": {
                "base_estimators": ["ridge"],
                "base_estimator_params": {"ridge": {"alpha": 31}},
            }
        }
    )
    previous = calculator.fit(X, y, {}).predict(X.to_numpy())
    config = {"params": {"base_estimators": ["linear_regression"]}}
    first = _nested_report(calculator, config)
    second = _nested_report(calculator, config)
    assert first == second
    assert calculator.fit(X, y, {}).predict(X.to_numpy()) == pytest.approx(previous)


@pytest.mark.parametrize(
    "family", ["voting_regressor", "stacking_regressor", "voting_classifier", "stacking_classifier"]
)
def test_nested_cv_accepts_empty_ensemble_config(family: str) -> None:
    """Every ensemble must resolve its default learners before nested CV constructs it."""
    X, y = _regression_rows()
    calculator = NodeRegistry.get_calculator(family)()
    if calculator.problem_type == "classification":
        y = pd.Series(np.where(y > y.median(), "high", "low"), name="target")
    report = perform_cross_validation(
        calculator,
        NodeRegistry.get_applier(family)(),
        X,
        y,
        {},
        cv_type="nested_cv",
        cv_nested_type=(
            "stratified_k_fold" if calculator.problem_type == "classification" else "k_fold"
        ),
        n_folds=2,
        inner_folds=2,
    )
    assert len(report["folds"]) == 2
    assert all(np.isfinite(fold["outer_score"]) for fold in report["folds"])
    assert report["fixed_parameters"] is True
