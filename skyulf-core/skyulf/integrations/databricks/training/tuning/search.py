"""Validate bounded local Core searches before a training source is opened."""

import importlib
import inspect
import json
import math
from contextlib import suppress
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from .....modeling._tuning.metrics import (
    CUSTOM_SCORER_BUILDERS,
    METRIC_ALIAS_MAP,
    validate_metric_for_problem_type,
)
from .....modeling._tuning.params import instantiate_model
from .....modeling._tuning.schemas import TuningConfig
from .....modeling.base import BaseModelCalculator
from .....modeling.hyperparameters import get_default_search_space
from .....preprocessing.fold_adapter import SPLITTER_STEP_TYPES, FeatureEngineerFoldAdapter
from .....registry import NodeRegistry
from ..fitting.ensemble import (
    ENSEMBLE_MODELS,
    ensemble_structural_keys,
    merge_ensemble_fixed_space,
    prepare_ensemble_model,
)

if TYPE_CHECKING:
    from .cv import CVSpec

_MAX_AXES = 128
_MAX_CANDIDATES_PER_AXIS = 100
_MAX_SEARCH_BYTES = 65_536


def base_model_config(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Return and validate the registered selected model behind an optional tuner."""
    modeling = pipeline.get("modeling")
    if not isinstance(modeling, dict):
        raise ValueError("pipeline.modeling must be an object.")
    selected = (
        modeling.get("base_model") if modeling.get("type") == "hyperparameter_tuner" else modeling
    )
    if not isinstance(selected, dict) or not isinstance(selected.get("type"), str):
        raise ValueError("pipeline.modeling.base_model.type must name a registered model.")
    calculator_type = NodeRegistry.get_calculator(selected["type"])
    if not issubclass(calculator_type, BaseModelCalculator):
        raise ValueError("Selected model must be a registered Core model calculator.")
    calculator = calculator_type()
    if calculator.problem_type not in {"classification", "regression"}:
        raise ValueError("Selected model must support classification or regression.")
    if modeling.get("type") == "hyperparameter_tuner" and not getattr(
        calculator, "model_class", None
    ):
        raise ValueError("Selected model does not expose a tunable model_class.")
    return selected


def _validate_axis_values(values: list[Any]) -> None:
    """Reject non-scalar or nonfinite candidates before JSON encoding."""
    for value in values:
        if value is not None and type(value) not in {bool, str, int, float}:
            raise ValueError("search_space candidates must be finite JSON scalars.")
        if type(value) in {int, float} and not math.isfinite(value):
            raise ValueError("search_space candidates must be finite JSON scalars.")


def _validate_axis(key: Any, values: Any) -> None:
    """Check an axis name and candidate bounds without implicit coercion."""
    if not isinstance(key, str) or not key or not isinstance(values, list) or not values:
        raise ValueError("Each search_space axis needs a name and nonempty candidate list.")
    if len(values) > _MAX_CANDIDATES_PER_AXIS:
        raise ValueError("search_space has too many candidates in one axis.")
    _validate_axis_values(values)


def bounded_space(space: Any) -> dict[str, list[Any]]:
    """Accept small finite JSON scalar candidate lists without implicit coercion."""
    if not isinstance(space, dict) or len(space) > _MAX_AXES:
        raise ValueError(f"search_space must be an object with at most {_MAX_AXES} axes.")
    checked: dict[str, list[Any]] = {}
    for key, values in space.items():
        _validate_axis(key, values)
        checked[key] = values
    if len(json.dumps(checked, allow_nan=False).encode("utf-8")) > _MAX_SEARCH_BYTES:
        raise ValueError("search_space exceeds the JSON size limit.")
    return checked


def _validate_parameter_names(calculator: BaseModelCalculator, space: dict[str, list[Any]]) -> None:
    """Reject names Core would silently drop from estimator construction."""
    model_class = cast(Any, calculator).model_class
    signature = inspect.signature(model_class)
    names = {
        name
        for name, param in signature.parameters.items()
        if param.kind != inspect.Parameter.VAR_KEYWORD
    }
    with suppress(TypeError, ValueError):
        names.update(
            instantiate_model(model_class, calculator.default_params).get_params(deep=True)
        )
    for name in space:
        if name not in names:
            raise ValueError(f"Unknown model parameter in search_space: {name}.")


def validate_metric(metric: Any, problem_type: str) -> None:
    """Apply Core's native alias and task rules without accepting heldout metrics."""
    if not isinstance(metric, str) or metric.startswith("heldout_"):
        raise ValueError("metric must be a native Core tuning metric, not a heldout metric.")
    if metric not in METRIC_ALIAS_MAP and metric not in CUSTOM_SCORER_BUILDERS:
        raise ValueError(f"Unsupported tuning metric: {metric}.")
    validate_metric_for_problem_type(problem_type, metric)
    if problem_type == "classification" and metric in {
        "mse",
        "mae",
        "rmse",
        "r2",
        "explained_variance",
    }:
        raise ValueError(f"metric {metric} is not compatible with Classification.")


def _optuna_available() -> bool:
    """Resolve the optional sklearn integration only for an Optuna search."""
    try:
        importlib.import_module("optuna")
    except ImportError:
        return False
    for module_name in ("optuna.integration.sklearn", "optuna_integration.sklearn"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if getattr(module, "OptunaSearchCV", None) is not None:
            return True
    return False


def _validate_single_worker(value: Any) -> None:
    """Prevent nested estimator workers inside search folds and trials."""
    if isinstance(value, dict):
        for name, item in value.items():
            if name == "n_jobs" and (type(item) is not int or item != 1):
                raise ValueError("Estimator n_jobs must be 1 to avoid nested parallel fits.")
            _validate_single_worker(item)
    elif isinstance(value, list):
        for item in value:
            _validate_single_worker(item)


def _prepare_selected_model(
    pipeline: dict[str, Any],
    modeling: dict[str, Any],
    cv: "CVSpec",
    target_column: str,
    event_column: str | None,
) -> tuple[dict[str, Any], BaseModelCalculator]:
    """Resolve Core's selected model and bind the workflow's fold contract."""
    allowed = set(TuningConfig.__dataclass_fields__) | {
        "type",
        "base_model",
        "max_candidates",
        "node_id",
    }
    unknown = set(modeling) - allowed
    if unknown:
        raise ValueError(f"Unknown search setting: {', '.join(sorted(unknown))}.")
    selected = base_model_config(pipeline)
    calculator = NodeRegistry.get_calculator(selected["type"])()
    selected_params = selected.get("params", {})
    if not isinstance(selected_params, dict):
        raise ValueError("base_model.params must be an object.")
    if selected["type"] in ENSEMBLE_MODELS:
        selected.setdefault("params", {}).setdefault("tune_base_models", True)
    prepare_ensemble_model(selected, calculator)
    calculator.prepare_tuning_params(selected)
    _bind_search_time(modeling, cv, calculator.problem_type, event_column)
    steps = pipeline.get("preprocessing", [])
    if any(step.get("transformer") in SPLITTER_STEP_TYPES for step in steps):
        raise ValueError("Search owns the split; remove preprocessing splitter nodes.")
    FeatureEngineerFoldAdapter(steps, target_column)
    _bind_shared_cv(modeling, cv)
    return selected, calculator


def _bind_search_time(
    modeling: dict[str, Any], cv: "CVSpec", problem_type: str, event_column: str | None
) -> None:
    """Validate splitter compatibility and bind authoritative temporal metadata."""
    _validate_search_split_policy(cv, problem_type, event_column)
    requested_time = modeling.get("cv_time_column")
    if cv.temporal:
        if requested_time is not None and requested_time != event_column:
            raise ValueError("cv_time_column must match the shared event_column.")
        modeling["cv_time_column"] = event_column
    elif requested_time is not None:
        raise ValueError("cv_time_column is only used by time_series_split.")


def _validate_search_split_policy(
    cv: "CVSpec", problem_type: str, event_column: str | None
) -> None:
    """Reject task and enabled-state conflicts before binding time metadata."""
    method = cv.nested_type if cv.method == "nested_cv" else cv.method
    if (
        method in {"stratified_k_fold", "stratified_group_k_fold"}
        and problem_type != "classification"
    ):
        raise ValueError("Stratified CV requires a classification model.")
    if cv.group_column and not cv.enabled:
        raise ValueError("Group search requires enabled CV.")
    if cv.temporal and (not cv.enabled or not event_column):
        raise ValueError("Time-series search requires enabled CV and an event_column.")


def _bind_shared_cv(modeling: dict[str, Any], cv: "CVSpec") -> None:
    """Reject wrapper overrides before copying the workflow's shared fold settings."""
    shared = {
        "cv_enabled": cv.enabled,
        "cv_folds": cv.folds,
        "cv_type": cv.method,
        "cv_shuffle": cv.shuffle,
        "cv_random_state": cv.random_state,
        "cv_nested_type": cv.nested_type,
        "cv_group_column": cv.group_column,
        "cv_gap": cv.gap,
        "cv_test_size": cv.test_size,
        "cv_max_train_size": cv.max_train_size,
    }
    if cv.inner_folds is not None or "cv_inner_folds" in modeling:
        shared["cv_inner_folds"] = cv.inner_folds
    for name, value in shared.items():
        if name in modeling and (
            type(modeling[name]) is not type(value) or modeling[name] != value
        ):
            raise ValueError(f"Conflicting wrapper {name}; use the shared workflow CV setting.")
    modeling.update(shared)


def _validate_strategy(
    modeling: dict[str, Any], calculator: BaseModelCalculator
) -> tuple[str, int]:
    """Bound trials and accepted strategy controls before generating candidates."""
    strategy = modeling.get("strategy", "random")
    if strategy not in {"grid", "random", "optuna", "halving_grid", "halving_random"}:
        raise ValueError("strategy must be grid, random, optuna, halving_grid or halving_random.")
    n_trials = modeling.get("n_trials", 10)
    if type(n_trials) is not int or not 1 <= n_trials <= 1000:
        raise ValueError("n_trials must be an integer from 1 to 1000.")
    max_candidates = modeling.get("max_candidates", 1000)
    if type(max_candidates) is not int or not 1 <= max_candidates <= 10_000:
        raise ValueError("max_candidates must be an integer from 1 to 10000.")
    _validate_strategy_options(modeling, strategy, calculator)
    if modeling.get("tune_threshold") and calculator.problem_type != "classification":
        raise ValueError("tune_threshold requires binary classification.")
    seed = _validate_search_execution(modeling)
    validate_metric(modeling.get("metric"), calculator.problem_type)
    modeling.update(
        strategy=strategy, n_trials=n_trials, max_candidates=max_candidates, random_state=seed
    )
    return strategy, max_candidates


def _validate_search_execution(modeling: dict[str, Any]) -> int:
    """Validate threshold, worker and seed controls before metric validation."""
    threshold = modeling.get("tune_threshold", False)
    if type(threshold) is not bool:
        raise ValueError("tune_threshold must be boolean.")
    if threshold and (modeling.get("cv_type") != "nested_cv" or not modeling.get("cv_enabled")):
        raise ValueError("tune_threshold is supported only by nested_cv search.")
    if modeling.get("n_jobs", 1) != 1 or type(modeling.get("n_jobs", 1)) is not int:
        raise ValueError("n_jobs must be 1 to avoid nested parallel fits.")
    seed = modeling.get("random_state", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("random_state must be an integer from 0 to 2**32 - 1.")
    return seed


def _validate_strategy_options(
    modeling: dict[str, Any], strategy: str, calculator: BaseModelCalculator
) -> None:
    """Check strategy-specific time and scheduler options in request order."""
    timeout = modeling.get("timeout")
    if timeout is not None and (
        strategy != "optuna"
        or type(timeout) not in {int, float}
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be positive seconds and is supported only by optuna.")
    strategy_params = modeling.get("strategy_params", {})
    if not isinstance(strategy_params, dict):
        raise ValueError("strategy_params must be an object.")
    if strategy in {"halving_grid", "halving_random"}:
        _validate_halving_params(strategy_params, calculator)
    elif strategy == "optuna":
        _validate_optuna_params(strategy_params)
    elif strategy_params:
        raise ValueError("strategy_params are unsupported for grid and random search.")


def _validate_optuna_params(strategy_params: dict[str, Any]) -> None:
    """Validate Optuna choices and require the optional integration."""
    if set(strategy_params) - {"sampler", "pruner", "pruning"}:
        raise ValueError("Unsupported optuna strategy_params.")
    if strategy_params.get("sampler", "tpe") not in {"tpe", "random", "cmaes"}:
        raise ValueError("optuna sampler must be tpe, random or cmaes.")
    if strategy_params.get("pruner", "median") not in {"median", "hyperband", "none"}:
        raise ValueError("optuna pruner must be median, hyperband or none.")
    if "pruning" in strategy_params and type(strategy_params["pruning"]) is not bool:
        raise ValueError("optuna pruning must be boolean.")
    if not _optuna_available():
        raise ValueError("optuna strategy requires Optuna and its sklearn integration.")


def _validate_halving_params(params: dict[str, Any], calculator: BaseModelCalculator) -> None:
    """Bound Core's actual halving resource and scheduler controls."""
    if set(params) - {"factor", "resource", "min_resources", "max_resources"}:
        raise ValueError("Unsupported halving strategy_params.")
    factor = params.get("factor", 3)
    if type(factor) is not int or not 2 <= factor <= 10:
        raise ValueError("halving factor must be an integer from 2 to 10.")
    resource = params.get("resource", "n_samples")
    _normalize_halving_counts(params)
    maximum = params.get("max_resources", "auto")
    if maximum != "auto" and (type(maximum) is not int or maximum < 2 or maximum > 1_000_000):
        raise ValueError("halving max_resources must be auto or a bounded positive integer.")
    _validate_estimator_resource(resource, maximum, calculator)
    _validate_minimum_resources(params, resource, maximum)


def _normalize_halving_counts(params: dict[str, Any]) -> None:
    """Normalize the accepted decimal-string resource counts in place."""
    for field in ("min_resources", "max_resources"):
        value = params.get(field)
        if isinstance(value, str) and value.isascii() and value.isdecimal():
            params[field] = int(value)


def _validate_estimator_resource(
    resource: Any, maximum: Any, calculator: BaseModelCalculator
) -> None:
    """Require an explicit bound and a real estimator parameter for custom resources."""
    if resource != "n_samples":
        if not isinstance(resource, str) or not resource or maximum == "auto":
            raise ValueError("Estimator resource requires an explicit max_resources integer.")
        estimator = instantiate_model(cast(Any, calculator).model_class, calculator.default_params)
        if resource not in estimator.get_params(deep=True):
            raise ValueError(f"Unknown halving estimator resource: {resource}.")


def _validate_minimum_resources(params: dict[str, Any], resource: Any, maximum: Any) -> None:
    """Check the minimum resource policy against the previously validated maximum."""
    minimum = params.get("min_resources", "exhaust")
    if minimum not in {"exhaust", "smallest"} and (
        type(minimum) is not int or not 2 <= minimum <= 1_000_000
    ):
        raise ValueError(
            "halving min_resources must be exhaust, smallest or a positive bounded integer."
        )
    if resource != "n_samples" and type(minimum) is not int:
        raise ValueError("Estimator resource requires an explicit min_resources integer.")
    if type(minimum) is int and type(maximum) is int and minimum > maximum:
        raise ValueError("halving min_resources cannot exceed max_resources.")


def _prepare_space(
    modeling: dict[str, Any],
    selected: dict[str, Any],
    calculator: BaseModelCalculator,
    strategy: str,
    max_candidates: int,
) -> dict[str, list[Any]]:
    """Merge automatic and fixed axes, then enforce candidate and worker limits."""
    raw_space = modeling.get("search_space", {})
    automatic_space = raw_space == {}
    if automatic_space:
        raw_space = calculator.build_tuning_search_space(selected, strategy)
        if raw_space == {}:
            raw_space = get_default_search_space(selected["type"], strategy)
    space = bounded_space(raw_space)
    if automatic_space and strategy in {"halving_grid", "halving_random"}:
        space.pop(modeling.get("strategy_params", {}).get("resource", "n_samples"), None)
    _merge_fixed_axes(space, selected, calculator, automatic_space)
    merge_ensemble_fixed_space(space, selected, automatic=automatic_space)
    space = bounded_space(space)
    _validate_grid_size(space, strategy, max_candidates)
    _bind_estimator_workers(space, modeling, calculator, strategy)
    _validate_parameter_names(calculator, space)
    return space


def _merge_fixed_axes(
    space: dict[str, list[Any]],
    selected: dict[str, Any],
    calculator: BaseModelCalculator,
    automatic_space: bool,
) -> None:
    """Preserve explicit fixed parameters and reject contradictory manual axes."""
    fixed = selected.get("params", {})
    _validate_single_worker(fixed)
    structural = ensemble_structural_keys(calculator)
    for name, value in fixed.items():
        if automatic_space:
            space.pop(name, None)
        elif name in space and (
            len(space[name]) != 1
            or type(space[name][0]) is not type(value)
            or space[name][0] != value
        ):
            raise ValueError(f"Fixed base parameter conflicts with search_space: {name}.")
        if name not in structural:
            space[name] = [value]


def _validate_grid_size(space: dict[str, list[Any]], strategy: str, max_candidates: int) -> None:
    """Bound exhaustive candidate combinations before estimator construction."""
    if strategy in {"grid", "halving_grid"}:
        count = math.prod(len(values) for values in space.values())
        if count > max_candidates:
            raise ValueError(
                f"Grid search_space has {count} candidates, above max_candidates={max_candidates}."
            )


def _bind_estimator_workers(
    space: dict[str, list[Any]],
    modeling: dict[str, Any],
    calculator: BaseModelCalculator,
    strategy: str,
) -> None:
    """Constrain estimator workers and keep halving resources out of search axes."""
    estimator = instantiate_model(cast(Any, calculator).model_class, calculator.default_params)
    for name in estimator.get_params(deep=True):
        if name == "n_jobs" or name.endswith("__n_jobs"):
            space.setdefault(name, [1])
    if strategy in {"halving_grid", "halving_random"}:
        resource = modeling.get("strategy_params", {}).get("resource", "n_samples")
        if resource != "n_samples" and resource in space:
            raise ValueError(
                f"Halving resource {resource} cannot also be a fixed or searched parameter."
            )
    _validate_worker_axes(space)


def _validate_worker_axes(space: dict[str, list[Any]]) -> None:
    """Reject non-single-worker values on every nested parallelism axis."""
    for name, values in space.items():
        if (name == "n_jobs" or name.endswith("__n_jobs")) and any(
            type(value) is not int or value != 1 for value in values
        ):
            raise ValueError("Estimator n_jobs search candidates must all be 1.")


def prepare_search_pipeline(
    pipeline: dict[str, Any],
    cv: "CVSpec",
    *,
    target_column: str,
    event_column: str | None,
) -> dict[str, Any]:
    """Return a safe effective Core recipe with authoritative shared search CV."""
    prepared = deepcopy(pipeline)
    modeling = prepared.get("modeling")
    if not isinstance(modeling, dict):
        return prepared
    if modeling.get("type") != "hyperparameter_tuner":
        if modeling.get("type") in ENSEMBLE_MODELS:
            prepare_ensemble_model(modeling, NodeRegistry.get_calculator(modeling["type"])())
            if modeling["params"].get("tune_base_models"):
                raise ValueError("tune_base_models requires a hyperparameter_tuner wrapper.")
        return prepared
    selected, calculator = _prepare_selected_model(
        prepared, modeling, cv, target_column, event_column
    )
    strategy, max_candidates = _validate_strategy(modeling, calculator)
    modeling["search_space"] = _prepare_space(
        modeling, selected, calculator, strategy, max_candidates
    )
    modeling["n_jobs"] = 1
    modeling.setdefault("tune_threshold", False)
    return prepared
