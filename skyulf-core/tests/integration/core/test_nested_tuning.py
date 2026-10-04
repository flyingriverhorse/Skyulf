"""Nested evaluation must score searches that never saw their outer test rows."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter
from skyulf.registry import NodeRegistry


def regression_rows() -> tuple[pd.DataFrame, pd.Series]:
    """Return noisy observations whose folds have different optimal regularization."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(72, 4)), columns=list("abcd"))
    y = pd.Series(2 * X.a - X.b + rng.normal(size=72), name="target")
    return X, y


@pytest.mark.parametrize("preprocess", [False, True])
def test_nested_grid_matches_independent_outer_searches(preprocess: bool) -> None:
    """Reported outer scores must match independent searches, including fold-local scaling."""
    X, y = regression_rows()
    config = TuningConfig(
        strategy="grid",
        search_space={"alpha": [0.01, 1.0, 100.0]},
        metric="mse",
        cv_type="nested_cv",
        cv_folds=3,
        cv_random_state=11,
    )
    adapter = (
        FeatureEngineerFoldAdapter(
            [{"name": "scale", "transformer": "StandardScaler", "params": {}}], "target"
        )
        if preprocess
        else None
    )
    model, result = TuningCalculator(NodeRegistry.get_calculator("ridge_regression")()).fit(
        X, y, config, preprocessing=adapter
    )
    report = getattr(result, "nested_cv", None)
    assert report is not None, "nested_cv must evaluate a fresh search in every outer fold"
    expected_scores = []
    for fold, (train, test) in zip(
        report["folds"], KFold(3, shuffle=True, random_state=11).split(X), strict=True
    ):
        estimator = make_pipeline(StandardScaler(), Ridge()) if preprocess else Ridge()
        key = "ridge__alpha" if preprocess else "alpha"
        search = GridSearchCV(
            estimator,
            {key: [0.01, 1.0, 100.0]},
            scoring="neg_mean_squared_error",
            cv=KFold(2, shuffle=True, random_state=11),
        ).fit(X.iloc[train], y.iloc[train])
        expected_scores.append(search.score(X.iloc[test], y.iloc[test]))
        assert fold["best_params"]["alpha"] == search.best_params_[key]
        assert fold["inner_best_score"] == pytest.approx(search.best_score_)
        assert fold["outer_score"] == pytest.approx(expected_scores[-1])
    assert report["mean_score"] == pytest.approx(np.mean(expected_scores))
    assert report["std_score"] == pytest.approx(np.std(expected_scores))
    assert report["total_trials"] == 12
    assert isinstance(model, Ridge)
    assert result.n_trials == 3


@pytest.mark.parametrize("strategy", ["grid", "random", "optuna", "halving_grid", "halving_random"])
@pytest.mark.parametrize("classification", [False, True])
def test_every_strategy_returns_complete_nested_evidence(
    strategy: Any, classification: bool
) -> None:
    """Each supported search must run once per outer fold plus a separate final search."""
    X, y = regression_rows()
    model_type = "decision_tree_classifier" if classification else "ridge_regression"
    if classification:
        y = pd.Series(np.where(y > y.median(), "yes", "no"), name="target")
    config = TuningConfig(
        strategy=strategy,
        metric="accuracy" if classification else "mse",
        search_space={"max_depth": [1, 3]} if classification else {"alpha": [0.1, 10.0]},
        cv_type="nested_cv",
        cv_folds=3,
        n_trials=2,
        strategy_params={"min_resources": 48, "factor": 2, "pruner": "none"},
    )
    _, result = TuningCalculator(NodeRegistry.get_calculator(model_type)()).fit(X, y, config)
    report = getattr(result, "nested_cv", None)
    assert report is not None
    assert len(report["folds"]) == 3
    assert all(f["train_rows"] == 48 and f["test_rows"] == 24 for f in report["folds"])
    assert all(np.isfinite(f["outer_score"]) for f in report["folds"])
    assert report["total_trials"] == sum(f["n_trials"] for f in report["folds"]) + result.n_trials


def test_external_validation_never_drives_nested_selection() -> None:
    """Changing reserved validation labels must not change any nested or final search result."""
    X, y = regression_rows()
    config = TuningConfig(
        strategy="grid",
        search_space={"alpha": [0.01, 1000.0]},
        metric="mse",
        cv_type="nested_cv",
        cv_folds=3,
    )
    tuner = TuningCalculator(NodeRegistry.get_calculator("ridge_regression")())
    _, ordinary = tuner.fit(X, y, config)
    _, reserved = tuner.fit(X, y, config, validation_data=(X.head(8), y.head(8) + 10000))
    assert getattr(reserved, "nested_cv", None) is not None
    assert reserved.best_params == ordinary.best_params
    assert reserved.best_score == ordinary.best_score
    assert reserved.nested_cv == ordinary.nested_cv


@pytest.mark.parametrize("strategy", ["halving_grid", "halving_random"])
@pytest.mark.parametrize("classification", [False, True])
def test_nested_halving_explicit_sample_ceiling_fits_each_outer_population(
    strategy: Any, classification: bool
) -> None:
    """A full-data sample ceiling must never cause replacement sampling in smaller searches."""
    X, y = regression_rows()
    family = "decision_tree_classifier" if classification else "ridge_regression"
    if classification:
        y = pd.Series(np.where(y > y.median(), "yes", "no"), name="target")
    params = {"resource": "n_samples", "min_resources": 36, "max_resources": 72, "factor": 2}
    config = TuningConfig(
        strategy=strategy,
        cv_type="nested_cv",
        cv_folds=3,
        n_trials=2,
        metric="accuracy" if classification else "mse",
        search_space={"max_depth": [1, 3]} if classification else {"alpha": [0.1, 10.0]},
        strategy_params=params.copy(),
    )
    _, result = TuningCalculator(NodeRegistry.get_calculator(family)()).fit(X, y, config)
    assert result.nested_cv is not None
    assert len(result.nested_cv["folds"]) == 3
    assert all(np.isfinite(fold["outer_score"]) for fold in result.nested_cv["folds"])
    assert config.strategy_params == params


@pytest.mark.parametrize("strategy", ["grid", "random", "optuna", "halving_grid", "halving_random"])
def test_preprocessing_never_fits_outer_test_rows(strategy: Any) -> None:
    """Every strategy must fit preprocessing only within its current outer training partition."""
    X, y = regression_rows()
    observations: list[tuple[int | None, set[int]]] = []
    phase: list[int | None] = [None]

    class AuditScaling:
        """Record actual fit membership across copied fold workers."""

        def fit_transform(self, X, y):
            """Track raw row identities before learning a fold's scaling."""
            observations.append((phase[0], set(X.index)))
            self.scaler = StandardScaler().fit(X)
            return pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns), y

        def transform(self, X, y):
            """Apply only the latest training partition's learned scaling."""
            return pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns), y

    def log(message: str) -> None:
        """Associate real preprocessing fits with their outer evaluation phase."""
        if message.startswith("Nested CV outer fold") and "starting inner" in message:
            phase[0] = int(message.split()[4].split("/")[0]) - 1
        if message.startswith("Nested CV complete"):
            phase[0] = None

    config = TuningConfig(
        strategy=strategy,
        search_space={"alpha": [0.1, 10.0]},
        metric="mse",
        cv_type="nested_cv",
        cv_folds=3,
        cv_inner_folds=3,
        n_trials=2,
        strategy_params={"min_resources": 48, "factor": 2, "pruner": "none"},
    )
    _, result = TuningCalculator(NodeRegistry.get_calculator("ridge_regression")()).fit(
        X, y, config, preprocessing=AuditScaling(), log_callback=log
    )
    assert result.nested_cv["inner_folds"] == 3
    for index, (train, test) in enumerate(KFold(3, shuffle=True, random_state=42).split(X)):
        fits = [rows for fold, rows in observations if fold == index]
        assert fits and any(rows == set(train) for rows in fits)
        assert any(len(rows) < len(train) for rows in fits)
        assert all(rows <= set(train) and rows.isdisjoint(test) for rows in fits)
    assert observations[-1] == (None, set(range(len(X))))


@pytest.mark.parametrize("inner", [True, 1, 2.5, 100])
def test_invalid_inner_folds_fail_before_any_fit(inner: Any) -> None:
    """Invalid nested partitions cannot degrade into incomplete outer evidence."""
    X, y = regression_rows()
    with pytest.raises(ValueError, match="fold"):
        TuningCalculator(NodeRegistry.get_calculator("ridge_regression")()).fit(
            X, y, TuningConfig(cv_type="nested_cv", cv_folds=3, cv_inner_folds=inner)
        )


def test_nested_threshold_selection_rejects_regression() -> None:
    """Binary threshold selection must reject regression instead of silently excluding it."""
    X, y = regression_rows()
    with pytest.raises(ValueError, match="binary classification"):
        TuningCalculator(NodeRegistry.get_calculator("ridge_regression")()).fit(
            X, y, TuningConfig(cv_type="nested_cv", tune_threshold=True, metric="r2")
        )


def test_nonfinite_outer_fold_rejects_the_whole_evaluation(monkeypatch) -> None:
    """A mean over surviving outer folds would overstate a failed selection procedure."""
    X, y = regression_rows()
    monkeypatch.setattr(
        "skyulf.modeling._tuning.nested.fit_and_score_candidate_fold", lambda **kwargs: float("nan")
    )
    with pytest.raises(ValueError, match="outer fold 1 failed"):
        TuningCalculator(NodeRegistry.get_calculator("ridge_regression")()).fit(
            X,
            y,
            TuningConfig(
                strategy="grid",
                metric="r2",
                search_space={"alpha": [1.0]},
                cv_type="nested_cv",
                cv_folds=6,
                cv_inner_folds=2,
            ),
        )
