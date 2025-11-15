"""Search configuration helpers shared across training and tuning workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .common import CrossValidationConfig, _parse_cross_validation_config
from core.feature_engineering.modeling.hyperparameter_tuning.registry import (
    get_default_strategy_value,
    resolve_strategy_selection,
)


@dataclass(frozen=True)
class SearchConfiguration:
    strategy: str
    selected_strategy: str
    search_space: Dict[str, List[Any]]
    n_iterations: Optional[int]
    scoring: Optional[str]
    random_state: Optional[int]
    cross_validation: CrossValidationConfig


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise ValueError("JSON payload is invalid") from exc
    return value


def _ensure_iterable_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _normalize_search_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"none", "null"}:
            return None
        return text
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_search_space(raw_space: Any) -> Dict[str, List[Any]]:
    parsed = _safe_json_loads(raw_space)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("Search space must be a JSON object mapping parameters to candidate values")

    search_space: Dict[str, List[Any]] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Search space keys must be non-empty strings")
        candidates = [_normalize_search_value(item) for item in _ensure_iterable_list(value)]

        filtered: List[Any] = []
        has_none = False
        for candidate in candidates:
            if candidate is None:
                has_none = True
                continue
            filtered.append(candidate)

        if has_none:
            filtered.append(None)

        if not filtered:
            continue
        search_space[key.strip()] = filtered

    return search_space


def _sanitize_parameters(raw_params: Any) -> Dict[str, Any]:
    parsed = _safe_json_loads(raw_params)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("Baseline hyperparameters must be a JSON object")
    sanitized: Dict[str, Any] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not key.strip():
            continue
        sanitized[key.strip()] = _normalize_search_value(value)
    return sanitized


def _coerce_none_strings(params: Dict[str, Any]) -> Dict[str, Any]:
    if not params:
        return {}
    coerced: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"none", "null"}:
                coerced[key] = None
                continue
            coerced[key] = text
        else:
            coerced[key] = value
    return coerced


def _filter_supported_parameters(params: Dict[str, Any], allowed_keys: Optional[set[str]]) -> Dict[str, Any]:
    if not params:
        return {}
    if not allowed_keys:
        return dict(params)
    filtered: Dict[str, Any] = {}
    for key, value in params.items():
        if key in allowed_keys:
            filtered[key] = value
    return filtered


def _coerce_cross_validation_config(raw: Any) -> CrossValidationConfig:
    if isinstance(raw, dict):
        return _parse_cross_validation_config(raw)
    return _parse_cross_validation_config({})


def _sanitize_logistic_regression_hyperparameters(
    base_params: Dict[str, Any], search_space: Dict[str, List[Any]]
) -> None:
    safe_solvers = ("lbfgs", "saga")
    safe_penalties = ("l2", "none")

    def _normalize_value(value: Any, safe_values: tuple[str, ...], fallback: str) -> str:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in safe_values:
                return lowered
        return fallback

    def _sanitize_candidates(candidates: List[Any], safe_values: tuple[str, ...], fallback: str) -> List[str]:
        sanitized: List[str] = []
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            lowered = candidate.strip().lower()
            if lowered in safe_values and lowered not in sanitized:
                sanitized.append(lowered)
        if not sanitized:
            sanitized = [fallback]
        return sanitized

    fallback_solver = safe_solvers[0]
    fallback_penalty = safe_penalties[0]

    normalized_solver = _normalize_value(base_params.get("solver"), safe_solvers, fallback_solver)
    normalized_penalty = _normalize_value(base_params.get("penalty"), safe_penalties, fallback_penalty)

    base_params["solver"] = normalized_solver
    base_params["penalty"] = normalized_penalty
    base_params.pop("l1_ratio", None)

    multi_class_value = base_params.get("multi_class")
    allowed_multi_classes = {"ovr", "multinomial"}
    if isinstance(multi_class_value, str):
        lowered_multi = multi_class_value.strip().lower()
        if lowered_multi in allowed_multi_classes:
            base_params["multi_class"] = lowered_multi
        else:
            base_params.pop("multi_class", None)
    else:
        base_params.pop("multi_class", None)

    if "solver" in search_space:
        search_space["solver"] = _sanitize_candidates(search_space["solver"], safe_solvers, normalized_solver)
    if "penalty" in search_space:
        search_space["penalty"] = _sanitize_candidates(search_space["penalty"], safe_penalties, normalized_penalty)

    if "multi_class" in search_space:
        multi_candidates = [
            candidate.strip().lower()
            for candidate in search_space["multi_class"]
            if isinstance(candidate, str) and candidate.strip().lower() in allowed_multi_classes
        ]
        deduped_multi = []
        for candidate in multi_candidates:
            if candidate not in deduped_multi:
                deduped_multi.append(candidate)
        if deduped_multi:
            search_space["multi_class"] = deduped_multi
        else:
            search_space.pop("multi_class", None)

    search_space.pop("l1_ratio", None)


def _build_search_configuration(job, node_config: Dict[str, Any]) -> SearchConfiguration:
    raw_strategy = job.search_strategy or node_config.get("search_strategy") or get_default_strategy_value()
    selected_strategy, strategy_impl = resolve_strategy_selection(raw_strategy)

    search_space_source = job.search_space or node_config.get("search_space") or {}
    search_space = _coerce_search_space(search_space_source)
    if not search_space:
        raise ValueError("Search space is empty. Provide at least one parameter with candidate values.")

    random_state = job.random_state
    if random_state is None:
        raw_random_state = node_config.get("search_random_state")
        if isinstance(raw_random_state, (int, float)):
            random_state = int(raw_random_state)
        elif isinstance(raw_random_state, str) and raw_random_state.strip().isdigit():
            random_state = int(raw_random_state.strip())

    scoring = job.scoring or node_config.get("scoring_metric")
    if isinstance(scoring, str):
        scoring = scoring.strip() or None
    else:
        scoring = None

    n_iterations = job.n_iterations
    if n_iterations is None:
        raw_iterations = node_config.get("search_iterations")
        if isinstance(raw_iterations, (int, float)):
            n_iterations = int(raw_iterations)
        elif isinstance(raw_iterations, str) and raw_iterations.strip().isdigit():
            n_iterations = int(raw_iterations.strip())

    if strategy_impl == "halving":
        n_iterations = None

    cross_validation = job.cross_validation or node_config
    cv_config = _coerce_cross_validation_config(cross_validation)
    if not cv_config.enabled:
        cv_config = CrossValidationConfig(
            True,
            "auto",
            max(cv_config.folds, 3),
            cv_config.shuffle,
            cv_config.random_state,
            cv_config.refit_strategy,
        )

    return SearchConfiguration(
        strategy=strategy_impl,
        selected_strategy=selected_strategy,
        search_space=search_space,
        n_iterations=n_iterations,
        scoring=scoring,
        random_state=random_state,
        cross_validation=cv_config,
    )


def _resolve_cv_config(search_config: SearchConfiguration) -> CrossValidationConfig:
    cv_config = search_config.cross_validation
    if (
        search_config.strategy in ("halving", "halving_random")
        and cv_config.shuffle
        and cv_config.random_state is None
    ):
        forced_random_state = (
            search_config.random_state
            if search_config.random_state is not None
            else 42
        )
        cv_config = CrossValidationConfig(
            enabled=cv_config.enabled,
            strategy=cv_config.strategy,
            folds=cv_config.folds,
            shuffle=cv_config.shuffle,
            random_state=forced_random_state,
            refit_strategy=cv_config.refit_strategy,
        )
    return cv_config


__all__ = [
    "SearchConfiguration",
    "_build_search_configuration",
    "_coerce_cross_validation_config",
    "_coerce_none_strings",
    "_coerce_search_space",
    "_filter_supported_parameters",
    "_resolve_cv_config",
    "_sanitize_logistic_regression_hyperparameters",
    "_sanitize_parameters",
]
