"""Successive halving honors bounded estimator resources through fold wrappers."""

from typing import Literal

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.fold_pipeline import FoldAwareModelStep
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling._tuning.strategies.halving import (
    bound_sample_resources,
    build_halving_searcher,
)
from skyulf.modeling.regression import RandomForestRegressorCalculator


@pytest.mark.parametrize("maximum", ["auto", 100, "100"])
def test_sample_minimum_cannot_be_silently_lowered(maximum):
    """Insufficient fold populations must fail rather than quietly weaken minimum resources."""
    config = TuningConfig(
        strategy="halving_grid", strategy_params={"min_resources": 48, "max_resources": maximum}
    )
    with pytest.raises(ValueError, match="min_resources=48.*sample budget \\(32\\)"):
        bound_sample_resources(config, 32)


def test_estimator_resource_ceiling_is_independent_of_sample_count():
    """Few training rows must not reduce an explicitly requested forest size."""
    config = TuningConfig(
        strategy="halving_grid",
        strategy_params={
            "resource": "n_estimators",
            "min_resources": 50,
            "max_resources": 100,
        },
    )
    assert bound_sample_resources(config, 10) is config


@pytest.mark.parametrize("strategy", ["halving_grid", "halving_random"])
@pytest.mark.parametrize(
    "resource, bounds",
    [("n_samples", {}), ("n_estimators", {"min_resources": 2, "max_resources": 4})],
)
def test_real_halving_search_uses_bounded_resource_through_fold_wrapper(
    strategy: Literal["halving_grid", "halving_random"], resource: str, bounds: dict
) -> None:
    """A wrapped estimator must receive each scheduled resource rung."""
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = 2 * X[:, 0] + X[:, 1]
    estimator = Pipeline(
        [("model", FoldAwareModelStep(estimator=RandomForestRegressor(random_state=7)))]
    )
    config = TuningConfig(
        strategy=strategy,
        metric="mse",
        n_trials=2,
        search_space={"model__estimator__max_depth": [2, 3]},
        cv_folds=2,
        n_jobs=1,
        strategy_params={"factor": 2, "resource": resource, **bounds},
    )
    searcher = build_halving_searcher(config, estimator, KFold(2), "neg_mean_squared_error", None)
    searcher.fit(X, y)

    assert searcher.resource == (
        "model__estimator__n_estimators" if resource == "n_estimators" else "n_samples"
    )
    assert searcher.max_resources == (4 if resource == "n_estimators" else "auto")
    assert np.isfinite(searcher.best_score_)
    if resource == "n_estimators":
        assert set(searcher.cv_results_["n_resources"]).issubset({2, 4})
        assert (
            searcher.best_params_[searcher.resource]
            == searcher.cv_results_["n_resources"][searcher.best_index_]
        )


@pytest.mark.parametrize("strategy", ["halving_grid", "halving_random"])
@pytest.mark.parametrize(
    "settings, message",
    [
        ({"resource": "n_estimators"}, "max_resources"),
        ({"resource": "missing", "max_resources": 4}, "resource"),
        ({"resource": "n_estimators", "max_resources": 0}, "max_resources"),
        ({"resource": "n_estimators", "max_resources": 4, "min_resources": 0}, "min_resources"),
    ],
)
def test_invalid_resource_controls_fail_before_search(
    strategy: Literal["halving_grid", "halving_random"], settings: dict, message: str
) -> None:
    """Invalid resource settings must fail before sklearn starts fitting."""
    estimator = Pipeline([("model", FoldAwareModelStep(estimator=RandomForestRegressor()))])
    config = TuningConfig(
        strategy=strategy,
        search_space={"model__estimator__max_depth": [2, 3]},
        strategy_params=settings,
    )
    with pytest.raises(ValueError, match=message):
        build_halving_searcher(config, estimator, KFold(2), "neg_mean_squared_error", None)


@pytest.mark.parametrize("axis", ["n_estimators", "model__estimator__n_estimators"])
def test_resource_cannot_be_a_search_axis(axis: str) -> None:
    """The halving scheduler alone must control the selected resource parameter."""
    estimator = Pipeline([("model", FoldAwareModelStep(estimator=RandomForestRegressor()))])
    config = TuningConfig(
        strategy="halving_grid",
        search_space={axis: [2, 4]},
        strategy_params={"resource": "n_estimators", "min_resources": 2, "max_resources": 4},
    )
    with pytest.raises(ValueError, match="search_space"):
        build_halving_searcher(config, estimator, KFold(2), "neg_mean_squared_error", None)


@pytest.mark.parametrize("strategy", ["halving_grid", "halving_random"])
@pytest.mark.parametrize("fold_preprocessing", [False, True])
def test_final_refit_inherits_selected_resource_not_base_default(
    strategy: Literal["halving_grid", "halving_random"], fold_preprocessing: bool
) -> None:
    """Final model and reported best params must use the winning halving rung."""
    X = pd.DataFrame(np.arange(48, dtype=float).reshape(24, 2), columns=["a", "b"])
    y = 2 * X["a"] + X["b"]
    tuner = TuningCalculator(RandomForestRegressorCalculator())
    config = TuningConfig(
        strategy=strategy,
        metric="mse",
        search_space={"max_depth": [2, 3]},
        n_trials=2,
        cv_folds=2,
        n_jobs=1,
        strategy_params={
            "factor": 2,
            "resource": "n_estimators",
            "min_resources": 2,
            "max_resources": 4,
        },
    )

    class _IdentityPreprocessor:
        """Exercise the real fold wrapper without changing feature values."""

        def fit_transform(self, X, y):
            """Return the fold's training payload."""
            return X, y

        def transform(self, X, y):
            """Return the fold's validation payload."""
            return X, y

    model, result = tuner.fit(
        X,
        y,
        config=config.__dict__,
        preprocessing=_IdentityPreprocessor() if fold_preprocessing else None,
    )

    assert result.best_params["n_estimators"] in (2, 4)
    assert model.n_estimators == result.best_params["n_estimators"]
