"""Preserve positional row weights across tuning boundaries."""

from importlib import import_module
from typing import Any

import numpy as np

from ._sample_weights import validate_sample_weight


def prepare_weights(values: Any, rows: int, preprocessing: Any = None) -> Any:
    """Validate row weights and reject preprocessing without a trusted row contract."""
    weights = validate_sample_weight(values, rows)
    if weights is not None:
        policy = import_module("skyulf.preprocessing._weight_policy")
        policy.validate_weighted_preprocessing(preprocessing)
    return weights


def weight_kwargs(weights: Any) -> dict[str, Any]:
    """Omit the new keyword entirely for legacy unweighted calculators."""
    return {} if weights is None else {"sample_weight": weights}


def take_weights(weights: Any, positions: Any) -> Any:
    """Slice by actual training positions and validate the resulting fit subset."""
    return None if weights is None else validate_sample_weight(weights[positions], len(positions))


def preflight_weights(weights: Any, cv: Any, X: Any, y: Any) -> None:
    """Reject invalid deterministic training subsets before any candidate learns."""
    if weights is not None:
        for train, _ in cv.split(X, y):
            take_weights(weights, train)


def search_weights(weights: Any, rows: int) -> Any:
    """Append neutral placeholders for holdout rows that never enter training."""
    if weights is None:
        return None
    return np.concatenate((weights, np.ones(rows - len(weights))))


def fit_preprocessor(preprocessing: Any, X: Any, y: Any, weights: Any) -> tuple[Any, Any, Any]:
    """Return the fitted training representation and its explicitly propagated weights."""
    if preprocessing is None:
        return X, y, weights
    if weights is None:
        X, y = preprocessing.fit_transform(X, y)
        return X, y, None
    prepare_weights(weights, len(X), preprocessing)
    X, y = preprocessing.fit_transform(X, y, sample_weight=weights)
    return X, y, validate_sample_weight(preprocessing.train_sample_weight_, len(X))
