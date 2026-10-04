"""Exercise real model training after weighted sampling inside every tuning route."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("imblearn")

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skyulf.pipeline import SkyulfPipeline

_OVER = [
    "random_over",
    "smote",
    "adasyn",
    "borderline_smote",
    "svm_smote",
    "kmeans_smote",
    "smote_tomek",
]
_UNDER = ["random_under_sampling", "nearmiss", "tomek_links", "edited_nearest_neighbours"]
_CASES = [(method, "direct") for method in _OVER + _UNDER] + [
    (method, route)
    for method in ["smote", "random_over", "random_under_sampling"]
    for route in ["grid", "random", "halving_grid", "halving_random", "optuna", "nested"]
]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method,route", _CASES)
def test_sampling_weights_reach_every_actual_fit(monkeypatch, method, route, engine):
    """Fold resampling must transform training weights while inference keeps all input rows."""
    if route == "optuna":
        pytest.importorskip("optuna")
    X, y = make_classification(
        n_samples=240,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        weights=[0.7, 0.3],
        class_sep=1.2,
        random_state=42,
    )
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
    frame["target"] = y
    weights = np.where(y == 1, 7.0, 3.0)
    step = {
        "name": "sampling",
        "transformer": "Oversampling" if method in _OVER else "Undersampling",
        "params": {
            "method": method,
            "synthetic_weight": "class_mean",
            "random_state": 19,
            "k_neighbors": 2,
            "kmeans_estimator": 2,
            "cluster_balance_threshold": 0.01,
            "n_jobs": 1,
        },
    }
    modeling = {
        "type": "logistic_regression",
        "params": {"class_weight": "balanced", "max_iter": 500},
    }
    if route != "direct":
        modeling = {
            "type": "hyperparameter_tuner",
            "base_model": modeling,
            "strategy": "grid" if route == "nested" else route,
            "search_space": {"C": [1.0]},
            "metric": "accuracy",
            "cv_folds": 2,
            "cv_inner_folds": 2,
            "cv_type": "nested_cv" if route == "nested" else "k_fold",
            "n_trials": 1,
            "n_jobs": 1,
            "strategy_params": {"min_resources": 100},
        }
    fits = []
    original = LogisticRegression.fit

    def fit(model, X, y, sample_weight=None):
        """Observe real estimator calls rather than successful wrapper return values."""
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, np.where(np.asarray(y) == 1, 7.0, 3.0))
        assert np.asarray(X).shape[1] == 4
        fits.append(len(y))
        return original(model, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(LogisticRegression, "fit", fit)
    pipeline = SkyulfPipeline({"preprocessing": [step], "modeling": modeling})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline.fit(native, "target", sample_weight=weights)
    features = (
        native.drop("target")
        if isinstance(native, pl.DataFrame)
        else native.drop(columns=["target"])
    )
    predicted = pipeline.predict(features)
    assert len(fits) >= (1 if route == "direct" else 3)
    assert len(predicted) == len(frame)
