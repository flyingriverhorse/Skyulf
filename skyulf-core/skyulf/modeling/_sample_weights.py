"""Validate row weights and reject estimators without proven weight routing."""

import inspect
from decimal import Decimal
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)


class SampleWeightError(ValueError):
    """Identify weight contract failures that tuning must not silently suppress."""


def validate_sample_weight(values: Any, expected_rows: int) -> NDArray[np.float64] | None:
    """Return a copied, unnormalized float vector with a positive finite total.

    Boolean, missing, nonnumeric, negative, and nonfinite values are rejected.
    Validation applies to each actual fit subset, including combined class and
    user weights, rather than only to the original training vector.
    """
    if values is None:
        return None
    weights = _numeric_weight_vector(values, expected_rows)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise SampleWeightError("sample_weight must contain finite, nonnegative values.")
    with np.errstate(over="ignore", invalid="ignore"):
        total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise SampleWeightError("sample_weight must have a positive finite total for each fit.")
    return weights


def _numeric_weight_vector(values: Any, expected_rows: int) -> NDArray[np.float64]:
    """Preserve scalar types until booleans and nonnumeric values are rejected."""
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise SampleWeightError("sample_weight must be a one-dimensional numeric vector.") from exc
    if raw.ndim != 1 or len(raw) != expected_rows:
        raise SampleWeightError(
            f"sample_weight must be one-dimensional with {expected_rows} values."
        )
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, Decimal))
        for value in raw
    ):
        raise SampleWeightError("sample_weight must be numeric without boolean or null values.")
    try:
        return np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SampleWeightError("sample_weight must contain finite numeric values.") from exc


def ensure_sample_weight_support(model: Any, *, check_routing: bool = True) -> None:
    """Require named fit support and recursively validate composite children."""
    if check_routing:
        for name, child in _weighted_children(model):
            try:
                ensure_sample_weight_support(child)
            except SampleWeightError as exc:
                raise SampleWeightError(f"{type(model).__name__}.{name}: {exc}") from exc
    # These sklearn composites explicitly route fit kwargs to the children above.
    # This allowlist is backed by real-fit routing tests; arbitrary kwargs are not.
    if type(model) in (
        CalibratedClassifierCV,
        VotingClassifier,
        VotingRegressor,
        StackingClassifier,
        StackingRegressor,
    ):
        return
    parameter = inspect.signature(model.fit).parameters.get("sample_weight")
    if parameter is None or parameter.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise SampleWeightError(
            f"{type(model).__name__} does not support sample_weight in fit(); "
            "user sample_weight or nonnative class_weight cannot be applied; "
            "class weighting cannot be applied to this model."
        )


def _weighted_children(model: Any) -> list[tuple[str, Any]]:
    """Return every estimator that receives row weights in supported composites."""
    if isinstance(model, CalibratedClassifierCV):
        return [("estimator", model._get_estimator())]
    if not isinstance(
        model, (VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor)
    ):
        return []
    children = [(name, child) for name, child in model.estimators if child != "drop"]
    if isinstance(model, (StackingClassifier, StackingRegressor)):
        children.append(("final_estimator", _stacking_final(model)))
    return children


def _stacking_final(model: Any) -> Any:
    """Resolve sklearn's documented stacking default without fitting it."""
    from sklearn.linear_model import LogisticRegression, RidgeCV  # noqa: PLC0415

    if model.final_estimator is not None:
        return model.final_estimator
    return LogisticRegression() if isinstance(model, StackingClassifier) else RidgeCV()
