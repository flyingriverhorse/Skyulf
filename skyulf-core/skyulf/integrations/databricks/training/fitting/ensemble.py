"""Admit Core ensemble recipes without silently changing project selections."""

import json
import math
from typing import Any

from .....modeling._tuning.params import instantiate_model

ENSEMBLE_MODELS = frozenset(
    {
        "voting_classifier",
        "voting_regressor",
        "stacking_classifier",
        "stacking_regressor",
    }
)


def _members(params: dict[str, Any], calculator: Any) -> list[str]:
    """Resolve defaults while rejecting unknown, duplicated or empty selections."""
    members = params.setdefault("base_estimators", list(calculator.DEFAULT_KEYS))
    if (
        not isinstance(members, list)
        or not members
        or any(
            not isinstance(name, str) or name not in calculator.BASE_ESTIMATORS for name in members
        )
        or len(set(members)) != len(members)
    ):
        raise ValueError("base_estimators must contain distinct available models for this task.")
    return members


def _parameter_map(value: Any, estimator: Any, label: str) -> None:
    """Reject parameter typos that Core's permissive ensemble resolver would ignore."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    unknown = set(value) - set(estimator.get_params(deep=True))
    if unknown:
        raise ValueError(f"Unknown {label} parameter: {', '.join(sorted(unknown))}.")


def _base_parameters(params: dict[str, Any], members: list[str], calculator: Any) -> None:
    """Validate fixed member parameters against the selected Core factories."""
    overrides = params.get("base_estimator_params", {})
    if not isinstance(overrides, dict) or set(overrides) - set(members):
        raise ValueError("base_estimator_params must name only selected base_estimators.")
    for name, values in overrides.items():
        _parameter_map(values, calculator.BASE_ESTIMATORS[name](), f"base_estimator_params.{name}")


def _fold_count(params: dict[str, Any], name: str, default: int) -> None:
    """Keep internal ensemble folds bounded and prohibit unfitted prefit mode."""
    value = params.get(name, default)
    if type(value) is not int or not 2 <= value <= 20:
        raise ValueError(f"Ensemble {name} must be an integer from 2 to 20.")


def _calibration(params: dict[str, Any], calculator: Any) -> None:
    """Admit explicit classification calibration without Core fallback/coercion."""
    enabled = params.get("calibrate_base_models", False)
    if type(enabled) is not bool:
        raise ValueError("calibrate_base_models must be boolean.")
    if enabled and calculator.problem_type != "classification":
        raise ValueError("Base model calibration requires classification.")
    if params.get("calibration_method", "sigmoid") not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration_method must be sigmoid or isotonic.")
    _fold_count(params, "calibration_cv", 3)


def _voting(params: dict[str, Any], members: list[str], calculator: Any) -> None:
    """Normalize named frontend weights to the exact selected learner order."""
    unused = {"final_estimator", "final_estimator_params", "cv", "passthrough"} & params.keys()
    if unused:
        raise ValueError(f"Voting does not use {', '.join(sorted(unused))}.")
    if "voting" in params and (
        calculator.problem_type != "classification" or params["voting"] not in {"soft", "hard"}
    ):
        raise ValueError("voting must be soft or hard for classification only.")
    weights = params.get("weights")
    if weights is None:
        return
    if isinstance(weights, dict):
        if set(weights) - set(members):
            raise ValueError("weights must name only selected base_estimators.")
        weights = [weights.get(name, 1) for name in members]
    _validate_weights(weights, members)
    params["weights"] = weights


def _validate_weights(weights: Any, members: list[str]) -> None:
    """Require one finite nonnegative weight per selected learner and a positive total."""
    if (
        not isinstance(weights, list)
        or len(weights) != len(members)
        or any(
            type(value) not in {int, float} or not math.isfinite(value) or value < 0
            for value in weights
        )
        or not any(weights)
    ):
        raise ValueError(
            "weights need one finite nonnegative value per model and a positive total."
        )


def _stacking(params: dict[str, Any], calculator: Any) -> None:
    """Keep meta-learner selection and its OOF folds separate from search CV."""
    if "weights" in params or "voting" in params:
        raise ValueError("Stacking does not use weights or voting.")
    final = params.get("final_estimator", calculator.DEFAULT_FINAL_KEY)
    if not isinstance(final, str) or final not in calculator.BASE_ESTIMATORS:
        raise ValueError("final_estimator must name an available model for this task.")
    _parameter_map(
        params.get("final_estimator_params", {}),
        calculator.BASE_ESTIMATORS[final](),
        "final_estimator_params",
    )
    _fold_count(params, "cv", 5)
    if type(params.get("passthrough", False)) is not bool:
        raise ValueError("passthrough must be boolean.")


def prepare_ensemble_model(selected: dict[str, Any], calculator: Any) -> None:
    """Validate and normalize the detached selected model before Core builds it."""
    if selected["type"] not in ENSEMBLE_MODELS:
        return
    params = selected.setdefault("params", {})
    if not isinstance(params, dict) or any(not isinstance(key, str) for key in params):
        raise ValueError("Ensemble params must be an object with string keys.")
    try:
        json.dumps(params, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ensemble parameters, including weights, must be finite JSON.") from exc
    members = _members(params, calculator)
    _base_parameters(params, members, calculator)
    _calibration(params, calculator)
    if type(params.get("tune_base_models", False)) is not bool:
        raise ValueError("tune_base_models must be boolean.")
    if calculator.IS_STACKING:
        _stacking(params, calculator)
    else:
        _voting(params, members, calculator)
    calculator.prepare_tuning_params(selected)
    estimator = instantiate_model(calculator.model_class, calculator.default_params)
    allowed = set(estimator.get_params(deep=True)) | ensemble_structural_keys(calculator)
    unknown = set(params) - (allowed - {"estimators"})
    if unknown:
        raise ValueError(f"Unknown ensemble parameter: {', '.join(sorted(unknown))}.")


def ensemble_structural_keys(calculator: Any) -> set[str]:
    """Keep ensemble builder controls out of sklearn candidate axes."""
    keys = set(calculator.STRUCTURAL_TUNING_KEYS)
    if "base_estimators" in keys:
        keys.add("tune_base_models")
    return keys


def merge_ensemble_fixed_space(
    space: dict[str, list[Any]],
    selected: dict[str, Any],
    *,
    automatic: bool,
) -> None:
    """Pin nested fixed parameters instead of searching over their catalog defaults."""
    if selected["type"] not in ENSEMBLE_MODELS:
        return
    params = selected.get("params", {})
    calibration = "estimator__" if params.get("calibrate_base_models") else ""
    fixed = {
        f"{name}__{calibration}{key}": value
        for name, values in params.get("base_estimator_params", {}).items()
        for key, value in values.items()
    }
    fixed |= {
        f"final_estimator__{key}": value
        for key, value in params.get("final_estimator_params", {}).items()
    }
    _merge_member_axes(space, params, fixed, automatic)


def _merge_member_axes(
    space: dict[str, list[Any]], params: dict[str, Any], fixed: dict[str, Any], automatic: bool
) -> None:
    """Check fixed nested overrides before adding scalar search axes."""
    for name, value in fixed.items():
        if name in params and params[name] != value:
            raise ValueError(f"Fixed base parameter conflicts with nested parameter: {name}.")
        if automatic:
            space.pop(name, None)
        elif name in space and space[name] != [value]:
            raise ValueError(f"Fixed base parameter conflicts with search_space: {name}.")
        # Structured fixed values remain on the prepared base estimator, never on an axis.
        if value is None or type(value) in {bool, int, float, str}:
            space[name] = [value]
