"""Data-free capability rules shared by Optuna execution and configuration inspection."""

from numbers import Integral
from typing import Any


def _search_can_select(estimator: Any, search_space: dict[str, Any], key: str, value: Any) -> bool:
    """Check fixed and categorical options, conservatively handling opaque distributions."""
    if key not in search_space:
        return getattr(estimator, key, None) == value
    candidates = search_space[key]
    if isinstance(candidates, list):
        return value in candidates
    choices = getattr(candidates, "choices", None)
    return choices is None or value in choices


def unsupported_pruning_reason(estimator: Any, search_space: dict[str, Any]) -> str | None:
    """Explain why the outer estimator cannot run Optuna's incremental trial loop.

    Inspect the estimator that the searcher receives, including any pipeline or
    fold wrapper. A supported estimator needs ``partial_fit`` and a fixed positive
    integer ``max_iter`` epoch budget. Search choices must preserve incremental
    compatibility. This checks capabilities only; it neither fits data nor depends
    on the currently selected pruner.
    """
    if not callable(getattr(estimator, "partial_fit", None)):
        return f"{type(estimator).__name__} does not support incremental training"
    epochs = getattr(estimator, "max_iter", None)
    if isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs <= 0:
        return f"{type(estimator).__name__} has no fixed positive integer max_iter epoch budget"
    if _search_can_select(estimator, search_space, "early_stopping", True):
        return "early_stopping=True is incompatible with incremental training"
    if _search_can_select(estimator, search_space, "class_weight", "balanced"):
        return "class_weight='balanced' is incompatible with incremental training"
    if _search_can_select(estimator, search_space, "solver", "lbfgs"):
        return "solver='lbfgs' does not support incremental training"
    if "max_iter" in search_space:
        return "a searched max_iter requires each candidate's ordinary fit budget"
    return None
