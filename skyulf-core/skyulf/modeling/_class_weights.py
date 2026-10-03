"""Shared class-weight policy for direct fits and tuning folds."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import type_of_target

from ._sample_weights import (
    _weighted_children,
    ensure_sample_weight_support,
    validate_sample_weight,
)


def constructor_accepts_class_weight(model_class: Any) -> bool:
    """Require a named parameter; arbitrary kwargs can silently ignore weighting."""
    return "class_weight" in inspect.signature(model_class).parameters


def split_class_weight_params(
    model_class: Any, params: dict[str, Any]
) -> tuple[dict[str, Any], Any]:
    """Keep native class weights in constructor params and extract fit-time weights.

    UI no-weight strings normalize to ``None``. Estimators such as XGBoost
    accept arbitrary kwargs but ignore class weights, so only an explicitly
    named constructor parameter counts as native support.
    """
    constructor_params = params.copy()
    if constructor_params.get("class_weight") in ("None", "none", ""):
        constructor_params["class_weight"] = None
    class_weight = None
    if "class_weight" in constructor_params and not constructor_accepts_class_weight(model_class):
        class_weight = constructor_params.pop("class_weight")
    return constructor_params, class_weight


def sample_weight_for_fit(
    model: Any, class_weight: Any, y: Any, sample_weight: Any = None
) -> NDArray[np.float64] | None:
    """Combine nonnative class weights once with this fit's user weights.

    Native class weights stay in constructor parameters; callers pass ``None``
    here for those models so their native weighted-frequency policy is retained.
    """
    has_class_weight = class_weight not in (None, "None", "none", "")
    if sample_weight is None and not has_class_weight:
        return None
    weights = validate_sample_weight(sample_weight, len(y))
    ensure_sample_weight_support(model, check_routing=sample_weight is not None)
    if has_class_weight:
        class_weights = compute_sample_weight(class_weight, y)
        with np.errstate(over="ignore", invalid="ignore"):
            weights = class_weights if weights is None else weights * class_weights
        weights = validate_sample_weight(weights, len(y))
    _validate_native_class_weight_product(model, y, weights)
    return weights


def _validate_native_class_weight_product(model: Any, y: Any, weights: Any) -> None:
    """Check fixed native factors without changing the weights passed to the model.

    Balanced policies depend on the estimator's weighted-frequency or bootstrap
    semantics, so their calculation remains entirely inside the native fit.
    """
    if weights is None:
        return
    native = getattr(model, "class_weight", None)
    if isinstance(native, (dict, list)):
        with np.errstate(over="ignore", invalid="ignore"):
            effective = weights * compute_sample_weight(native, y)
        validate_sample_weight(effective, len(y))
    if isinstance(model, (VotingClassifier, StackingClassifier)):
        if isinstance(model, StackingClassifier) and type_of_target(y) == "multilabel-indicator":
            y = np.column_stack(
                [LabelEncoder().fit_transform(column) for column in np.asarray(y).T]
            )
        else:
            y = LabelEncoder().fit_transform(y)
    for _, child in _weighted_children(model):
        _validate_native_class_weight_product(child, y, weights)
