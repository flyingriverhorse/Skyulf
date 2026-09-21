"""Validate the numeric canvas controls consumed by executable nodes."""

import math
import re
from typing import Any


def _finite(value: Any) -> bool:
    """Reject nonnumeric and overflow-sized values without breaking validation."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _number(
    params: dict[str, Any],
    field: str,
    minimum: float,
    maximum: float = math.inf,
    *,
    integer: bool = False,
) -> None:
    """Keep omitted defaults while rejecting explicit invalid values."""
    if field not in params:
        return
    value = params[field]
    if not _finite(value) or not minimum <= value <= maximum or (integer and value != int(value)):
        kind = "integer" if integer else "number"
        raise ValueError(f"{field} must be a finite {kind} in [{minimum}, {maximum}].")


def _model_threshold(value: Any) -> bool:
    """Accept sklearn's documented mean/median expressions and finite numbers."""
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return _finite(value)
    if not isinstance(value, str):
        return False
    value = value.strip()
    if value in {"mean", "median"}:
        return True
    try:
        return math.isfinite(float(value))
    except ValueError:
        pass
    match = re.fullmatch(r"(.+)\*\s*(mean|median)", value)
    if not match:
        return False
    try:
        return math.isfinite(float(match[1]))
    except ValueError:
        return False


def _model_selection(params: dict[str, Any]) -> None:
    """Validate model-based selection thresholds and feature counts."""
    if "threshold" in params and not _model_threshold(params["threshold"]):
        raise ValueError(
            "threshold must be finite, mean, median or a finite multiple of mean/median."
        )
    if params.get("max_features") is not None:
        _number(params, "max_features", 0, integer=True)


def _rfe_selection(params: dict[str, Any]) -> None:
    """Validate recursive selection counts and fractional steps."""
    count_field = "n_features_to_select" if params.get("n_features_to_select") is not None else "k"
    for field in (count_field, "step"):
        if field != "step" and params.get(field) is None:
            continue
        _number(params, field, math.ulp(0.0))
        value = params.get(field, 1)
        if value >= 1 and value != int(value):
            raise ValueError(f"{field} must be a positive integer or a fraction below 1.")


def _univariate_selection(params: dict[str, Any]) -> None:
    """Validate the active generic univariate selector mode."""
    mode = params.get("mode", "k_best")
    field = (
        "param"
        if "param" in params
        else {"k_best": "k", "percentile": "percentile"}.get(mode, "alpha")
    )
    if mode == "k_best" and params.get(field) == "all":
        return
    _number(
        params,
        field,
        1 if mode == "k_best" else 0,
        math.inf if mode == "k_best" else 100 if mode == "percentile" else 1,
        integer=mode == "k_best",
    )


def _selection(params: dict[str, Any]) -> None:
    """Validate only the active selector's settings."""
    method = params.get("method", "variance")
    if method in {"variance", "variance_threshold", "correlation_threshold"}:
        _number(params, "threshold", 0, 1 if method == "correlation_threshold" else math.inf)
    elif method == "select_from_model":
        _model_selection(params)
    elif method == "select_k_best":
        if params.get("k") != "all":
            _number(params, "k", 1, integer=True)
    elif method == "rfe":
        _rfe_selection(params)
    elif method == "select_percentile":
        _number(params, "percentile", 0, 100)
    elif method in {"select_fpr", "select_fdr", "select_fwe"}:
        _number(params, "alpha", 0, 1)
    elif method == "generic_univariate_select":
        _univariate_selection(params)


def _finite_candidates(value: Any) -> None:
    """Allow null and categorical candidates while rejecting nonfinite numbers."""
    if isinstance(value, int | float) and not isinstance(value, bool) and not _finite(value):
        raise ValueError("search_space candidates must be finite.")
    if isinstance(value, dict):
        for candidate in value.values():
            _finite_candidates(candidate)
    elif isinstance(value, list):
        for candidate in value:
            _finite_candidates(candidate)


def _training(params: dict[str, Any], *, tuned: bool = False) -> None:
    """Validate consumed budgets and CV; tuned worker counts are server-owned."""
    if tuned:
        _number(params, "n_trials", 1, integer=True)
        _finite_candidates(params.get("search_space"))
    if params.get("cv_enabled", tuned):
        _number(params, "cv_folds", 2, integer=True)
    for seed in ("random_state", "cv_random_state"):
        if params.get(seed) is not None:
            _number(params, seed, 0, 4294967295, integer=True)
    _training_structure(params, tuned=tuned)


def _training_structure(params: dict[str, Any], *, tuned: bool) -> None:
    """Validate model CV and explicit basic-training worker counts."""
    model = params.get("model_type", params.get("algorithm", ""))
    structure = params if tuned else params.get("hyperparameters", {})
    if not isinstance(structure, dict):
        return
    if str(model).startswith("stacking"):
        _number(structure, "cv", 2, integer=True)
    if str(model).endswith("_classifier") and structure.get("calibrate_base_models"):
        _number(structure, "calibration_cv", 2, integer=True)
    if not tuned and structure.get("n_jobs") is not None:
        _number(structure, "n_jobs", -math.inf, integer=True)
        if structure["n_jobs"] == 0:
            raise ValueError("n_jobs must be a nonzero integer (-1 uses all cores).")


def validate_node_numbers(step_type: str, params: dict[str, Any]) -> None:
    """Reject invalid explicit numeric controls at the HTTP configuration boundary."""
    if step_type in {"TrainTestSplitter", "Split"}:
        _number(params, "test_size", 0, 1)
        _number(params, "validation_size", 0, 1)
        test = params.get("test_size", 0.2)
        validation = params.get("validation_size", 0)
        if not 0 < test < 1:
            raise ValueError("test_size must be between 0 and 1, exclusively.")
        if test + validation >= 1:
            raise ValueError("validation_size + test_size must be less than 1.")
    elif step_type in {"IQR", "ZScore"}:
        _number(params, "multiplier" if step_type == "IQR" else "threshold", math.ulp(0.0))
    elif step_type == "feature_selection":
        _selection(params)
    elif step_type in {"training", "tuning"}:
        _training_node(step_type, params)


def _training_node(step_type: str, params: dict[str, Any]) -> None:
    """Resolve nested tuning controls before validating consumed values."""
    tuned = step_type == "tuning" or params.get("run_mode") in {"tuned", "advanced"}
    config = params.get("tuning_config", {}) if tuned else params
    if isinstance(config, dict):
        _training(
            {
                **config,
                "algorithm": params.get("algorithm")
                or params.get("model_type")
                or config.get("algorithm", ""),
            },
            tuned=tuned,
        )
