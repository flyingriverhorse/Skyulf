"""Shared class-weight policy for direct fits and tuning folds."""

import inspect
from typing import Any

from sklearn.utils.class_weight import compute_sample_weight


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


def sample_weight_for_fit(model: Any, class_weight: Any, y: Any) -> Any:
    """Compute weights from this fit's labels, rejecting unsupported estimators."""
    if class_weight in (None, "None", "none", ""):
        return None
    if "sample_weight" not in inspect.signature(model.fit).parameters:
        raise ValueError(
            f"{type(model).__name__} does not support 'class_weight' natively "
            "and its fit() method does not accept 'sample_weight' either, so "
            "class weighting cannot be applied to this model."
        )
    return compute_sample_weight(class_weight, y)
