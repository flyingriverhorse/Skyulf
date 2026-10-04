"""Transport selected weights and assign explicit weights to synthetic training rows."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..engines import SkyulfPolarsWrapper
from ..modeling._sample_weights import SampleWeightError, validate_sample_weight
from ..utils import pack_pipeline_output, unpack_pipeline_input


def _synthetic_weights(y: Any, weights: np.ndarray, labels: Any, policy: str) -> np.ndarray:
    """Assign new rows a documented constant or their training class's mean weight."""
    labels = np.asarray(labels)
    if policy == "uniform":
        return np.ones(len(labels), dtype=float)
    means = {label: weights[np.asarray(y) == label].mean() for label in np.unique(y)}
    return np.asarray([means[label] for label in labels], dtype=float)


def _append_weights(
    X: pd.DataFrame, y: Any, weights: np.ndarray, X_res: Any, y_res: Any, policy: str
) -> np.ndarray:
    """Verify the original prefix before assigning weights to appended synthetic rows."""
    prefix = pd.DataFrame(np.asarray(X_res)[: len(X)], dtype=object)
    original = pd.DataFrame(np.asarray(X), dtype=object)
    if not prefix.equals(original):
        raise SampleWeightError("Synthetic sampler did not preserve the original feature rows.")
    if not np.array_equal(np.asarray(y_res)[: len(y)], np.asarray(y)):
        raise SampleWeightError("Synthetic sampler did not preserve the original labels.")
    added = _synthetic_weights(y, weights, np.asarray(y_res)[len(y) :], policy)
    return np.concatenate([weights, added])


def _sample_synthetic(
    sampler: Any, X: pd.DataFrame, y: Any, weights: np.ndarray, params: dict[str, Any]
) -> tuple[Any, Any, np.ndarray]:
    """Run synthetic sampling without adding weight values to the neighbor geometry."""
    policy = params.get("synthetic_weight")
    if policy not in ("class_mean", "uniform"):
        raise SampleWeightError(
            "Synthetic resampling with sample_weight requires explicit "
            "synthetic_weight='class_mean' or 'uniform'."
        )
    if params["method"] == "smote_tomek":
        from imblearn.under_sampling import TomekLinks  # noqa: PLC0415 - optional extra
        from sklearn.base import clone  # noqa: PLC0415

        X_res, y_res = clone(sampler.smote).fit_resample(X, y)
        combined = _append_weights(X, y, weights, X_res, y_res, policy)
        cleaner = TomekLinks(sampling_strategy="all", n_jobs=sampler.n_jobs)
        X_res, y_res = cleaner.fit_resample(X_res, y_res)
        return X_res, y_res, combined[cleaner.sample_indices_]
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res, _append_weights(X, y, weights, X_res, y_res, policy)


def _resample(
    X: pd.DataFrame, y: Any, weights: np.ndarray, params: dict[str, Any]
) -> tuple[Any, Any, np.ndarray]:
    """Reuse the regular sampler builders, including their parameter validation."""
    from .resampling import (  # noqa: PLC0415 - avoid initialization cycle
        _build_oversampler,
        _build_undersampler,
        _validate_numeric,
    )

    _validate_numeric(X)
    over = params["type"] == "oversampling"
    builder = _build_oversampler if over else _build_undersampler
    sampler = builder(params["method"], params)
    if over and params["method"] != "random_over":
        return _sample_synthetic(sampler, X, y, weights, params)
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res, weights[sampler.sample_indices_]


def fit_resample_weighted(
    calculator: Any, applier: Any, data: Any, config: dict[str, Any], weights: Any
) -> tuple[dict[str, Any], Any, np.ndarray]:
    """Fit a registered sampler and return its artifact, aligned data and weights.

    Random oversampling repeats source weights. Undersampling selects source
    weights. Synthetic methods require ``synthetic_weight`` explicitly:
    ``class_mean`` uses the input training class's arithmetic mean; ``uniform``
    assigns one. Neither policy affects sampling probabilities or distances.
    """
    X, y, wrapped, return_tuple = _prepare_inputs(data, config)
    validated = validate_sample_weight(weights, len(X))
    if validated is None:
        raise SampleWeightError("Weighted resampling requires sample_weight.")
    params = dict(calculator.fit((X, y), config))
    polars = isinstance(X, pl.DataFrame)
    X_pd = X.to_pandas() if polars else X
    y_pd = y.to_pandas() if isinstance(y, pl.Series) else y
    y_pd = pd.Series(y_pd)
    X_res, y_res, result_weights = _resample(X_pd, y_pd, validated, params)
    result_weights = validate_sample_weight(result_weights, len(X_res))
    assert result_weights is not None
    if polars:
        X_res, y_res = pl.from_pandas(X_res), pl.from_pandas(y_res)
    if wrapped:
        X_res = SkyulfPolarsWrapper(X_res)
    return params, pack_pipeline_output(X_res, y_res, return_tuple), result_weights


def _prepare_inputs(data: Any, config: dict[str, Any]) -> tuple[Any, Any, bool, bool]:
    """Retain the ordinary sampler's output shape before extracting embedded targets."""
    X, y, was_tuple = unpack_pipeline_input(data)
    return_tuple = was_tuple and y is not None
    wrapped = isinstance(X, SkyulfPolarsWrapper)
    X = X.to_native() if hasattr(X, "to_native") else X
    if y is None:
        target = config.get("target_column")
        if not target or target not in X.columns:
            raise SampleWeightError("Weighted resampling requires a target.")
        y = X[target]
        X = X.drop(target) if isinstance(X, pl.DataFrame) else X.drop(columns=[target])
    return X, y, wrapped, return_tuple
