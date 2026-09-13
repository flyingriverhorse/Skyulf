"""Data-free capability rules shared by Optuna execution and configuration inspection."""

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Any, cast

from sklearn.pipeline import Pipeline

from ._tuning.fold_pipeline import FoldAwareModelStep


@dataclass(frozen=True)
class PruningPlan:
    """Describe an available pruning loop and its shared iteration coordinate."""

    mode: str
    reason: str
    kind: str | None = None
    iteration_budget: int = 1


def _pruning_target(estimator: Any) -> tuple[Any, str]:
    """Inspect known fold adapters without discarding arbitrary pipeline transforms."""
    if isinstance(estimator, FoldAwareModelStep):
        return estimator.estimator, "estimator__"
    if isinstance(estimator, Pipeline) and len(estimator.steps) == 1:
        name, step = estimator.steps[0]
        if name == "model" and isinstance(step, FoldAwareModelStep):
            return step.estimator, f"{name}__estimator__"
    return estimator, ""


def _boosting_kind(model: Any) -> str | None:
    """Recognize native library inheritance without importing optional boosters."""
    modules = {cls.__module__ for cls in type(model).__mro__}
    return next((name for name in ("xgboost", "lightgbm") if f"{name}.sklearn" in modules), None)


def _positive_integer(value: Any) -> bool:
    """Exclude booleans and invalid budgets before allocating iteration coordinates."""
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def _default_boosting_budget(model: Any, params: dict[str, Any]) -> Any:
    """Resolve XGBoost's implicit round count through its public estimator method."""
    budget = params.get("n_estimators")
    native_default = getattr(model, "get_num_boosting_rounds", None)
    return native_default() if budget is None and callable(native_default) else budget


def _boosting_budget(model: Any, space: dict[str, Any], prefix: str) -> int | None:
    """Use one upper round limit across trials so folds never reuse a report step."""
    params = model.get_params(deep=False)
    # Alternate LightGBM round parameters can override n_estimators. Keep these
    # searches on fold pruning until their resource precedence is modeled here.
    aliases = (
        "num_iterations",
        "num_iteration",
        "num_tree",
        "num_trees",
        "num_round",
        "num_rounds",
        "nrounds",
        "n_iter",
        "num_boost_round",
        "max_iter",
    )
    if any(name in params or f"{prefix}{name}" in space for name in aliases):
        return None
    values = space.get(f"{prefix}n_estimators", [_default_boosting_budget(model, params)])
    bounds: Any = values if isinstance(values, list) else getattr(values, "choices", None)
    if bounds is None:
        bounds = [getattr(values, "high", None)]
    if not bounds or not all(_positive_integer(value) for value in bounds):
        return None
    return int(max(cast(Iterable[int], bounds)))


def resolve_pruning_plan(
    estimator: Any,
    search_space: dict[str, Any],
    *,
    n_splits: int,
    preprocessing: bool = False,
) -> PruningPlan:
    """Choose native iterations, direct partial_fit, or complete-fold pruning.

    Known Skyulf fold adapters retain their preprocessing in the native fit
    helper. Other pipelines use their ordinary fit interface between CV folds.
    A single holdout has no later fold to skip, so it needs an iteration hook.
    """
    model, prefix = _pruning_target(estimator)
    kind = _boosting_kind(model)
    if kind is not None:
        budget = _boosting_budget(model, search_space, prefix)
        if budget is not None:
            return PruningPlan(
                "iterations", "Pruning can stop poor trials during boosting.", kind, budget
            )
    if not preprocessing and unsupported_pruning_reason(estimator, search_space) is None:
        return PruningPlan(
            "iterations",
            "Pruning can stop poor trials during incremental training.",
            "incremental",
            int(estimator.max_iter),
        )
    if n_splits >= 2:
        return PruningPlan(
            "folds",
            "Pruning can skip remaining CV folds after a completed fold; each fold is fully trained.",
        )
    return PruningPlan(
        "none",
        "Pruning between CV folds needs at least two folds; this search uses a single holdout without a supported iteration hook.",
    )


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
