"""Separate unweighted imputation from aligned weighted model training."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 - activate estimator
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import KNeighborsRegressor

from skyulf.modeling._sample_weights import SampleWeightError
from skyulf.pipeline import SkyulfPipeline


def _data() -> pd.DataFrame:
    """Keep row identity observable despite missing values and duplicate index labels."""
    rows = np.random.default_rng(19).permutation(72).astype(float)
    values = np.sin(rows / 3)
    values[rows.astype(int) % 5 == 0] = np.nan
    return pd.DataFrame(
        {"row": rows, "value": values, "context": np.cos(rows / 5), "target": rows / 10},
        index=np.zeros(len(rows)),
    )


def _imputation(kind: str) -> tuple[dict[str, Any], Any]:
    """Pair Core settings with an independent sklearn preprocessing reference."""
    params: dict[str, Any] = {"columns": ["value", "context"]}
    node = kind
    if kind == "SimpleImputer":
        params["strategy"] = "mean"
        reference = SimpleImputer(strategy="mean")
    elif kind == "KNNImputer":
        params["n_neighbors"] = 3
        reference = KNNImputer(n_neighbors=3)
    else:
        node = "IterativeImputer"
        params.update(max_iter=3, random_state=0)
        estimator = BayesianRidge()
        if kind == "IterativeKNN":
            params["estimator"] = "KNN"
            estimator = KNeighborsRegressor(n_neighbors=5)
        reference = IterativeImputer(estimator=estimator, max_iter=3, random_state=0)
    return {"name": "impute", "transformer": node, "params": params}, reference


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "kind", ["SimpleImputer", "KNNImputer", "IterativeImputer", "IterativeKNN"]
)
@pytest.mark.parametrize(
    "route", ["direct", "grid", "random", "halving_grid", "halving_random", "optuna", "nested"]
)
def test_imputation_preserves_weights_in_actual_fits(monkeypatch, engine, kind, route):
    """Imputation must preserve row weights across every fold and the final refit."""
    if route == "optuna":
        pytest.importorskip("optuna")
    frame = _data()
    step, reference_imputer = _imputation(kind)
    modeling: dict[str, Any] = {"type": "ridge_regression", "params": {"alpha": 1.0}}
    if route != "direct":
        modeling = {
            "type": "hyperparameter_tuner",
            "base_model": modeling,
            "strategy": "grid" if route == "nested" else route,
            "search_space": {"alpha": [1.0]},
            "metric": "mse",
            "cv_folds": 2,
            "cv_inner_folds": 2,
            "cv_type": "nested_cv" if route == "nested" else "k_fold",
            "n_trials": 1,
            "n_jobs": 1,
            "strategy_params": {"min_resources": 24},
        }
    pipeline = SkyulfPipeline({"preprocessing": [step], "modeling": modeling})
    observed = []
    original = Ridge.fit

    def record(model, X, y, sample_weight=None):
        """Check the payload while still executing the real estimator's fit method."""
        values = np.asarray(X)
        assert np.isfinite(values).all()
        assert sample_weight is not None
        np.testing.assert_array_equal(sample_weight, values[:, 0] + 1)
        observed.append(len(values))
        return original(model, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(Ridge, "fit", record)
    native = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline.fit(native, "target", sample_weight=frame.row.to_numpy() + 1)
    assert observed[-1] == len(frame)
    assert len(observed) >= (1 if route == "direct" else 3)
    filled = reference_imputer.fit_transform(frame[["value", "context"]])
    reference_X = np.column_stack([frame.row.to_numpy(), filled])
    reference = Ridge(alpha=1.0)
    original(
        reference, reference_X, frame.target.to_numpy(), sample_weight=frame.row.to_numpy() + 1
    )
    features = (
        native.drop(columns=["target"])
        if isinstance(native, pd.DataFrame)
        else native.drop("target")
    )
    predicted = pipeline.predict(features)
    np.testing.assert_allclose(np.asarray(predicted), reference.predict(reference_X), atol=1e-8)


@pytest.mark.parametrize("model", ["k_neighbors_classifier", "k_neighbors_regressor"])
def test_knn_final_model_is_distinct_from_knn_imputation(model):
    """An allowed KNN imputer must not imply that a final KNN model accepts row weights."""
    frame = _data()
    frame["target"] = np.arange(len(frame)) % 2
    step, _ = _imputation("KNNImputer")
    pipeline = SkyulfPipeline({"preprocessing": [step], "modeling": {"type": model}})
    with pytest.raises(SampleWeightError, match="does not support sample_weight"):
        pipeline.fit(frame, "target", sample_weight=frame.row.to_numpy() + 1)
