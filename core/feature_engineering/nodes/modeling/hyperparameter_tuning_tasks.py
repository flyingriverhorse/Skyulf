"""Celery tasks that execute hyperparameter tuning workflows."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV

from config import get_settings
from core.database.models import get_database_session

from .hyperparameter_tuning_jobs import get_tuning_job, update_tuning_job_status
from .hyperparameter_tuning_registry import get_default_strategy_value, resolve_strategy_selection
from .model_training_registry import get_model_spec
from .model_training_tasks import (
    CrossValidationConfig,
    _build_cv_splitter,
    _classification_metrics,
    _ensure_database_ready,
    _parse_cross_validation_config,
    _prepare_training_data,
    _regression_metrics,
    _resolve_training_inputs,
    celery_app,
)
from core.feature_engineering.schemas import HyperparameterTuningJobStatus
from .model_hyperparameters import get_hyperparameters_for_model

CategoricalDistribution: Any
TPESampler: Any

try:  # optuna support is optional for early-stopping strategies
    from optuna.distributions import CategoricalDistribution as _CategoricalDistribution
    from optuna.samplers import TPESampler as _TPESampler
except ImportError:  # pragma: no cover - optional dependency safeguard
    CategoricalDistribution = None
    TPESampler = None
    _HAS_OPTUNA = False
else:
    CategoricalDistribution = _CategoricalDistribution
    TPESampler = _TPESampler
    _HAS_OPTUNA = True

OptunaSearchCV: Any = None

if _HAS_OPTUNA:
    try:
        from optuna.integration import OptunaSearchCV as _OptunaSearchCV  # type: ignore[attr-defined]

        OptunaSearchCV = _OptunaSearchCV
    except ImportError:  # pragma: no cover - optuna>=3.4 fallback
        try:
            from optuna.integration.sklearn import OptunaSearchCV as _OptunaSearchCV  # type: ignore[attr-defined]

            OptunaSearchCV = _OptunaSearchCV
        except ImportError:  # pragma: no cover - optuna>=4 fallback
            try:
                from optuna_integration.sklearn import OptunaSearchCV as _OptunaSearchCV  # type: ignore[attr-defined]

                OptunaSearchCV = _OptunaSearchCV
            except ImportError:  # pragma: no cover - integration package missing
                OptunaSearchCV = cast(Any, None)

    if OptunaSearchCV is None:
        _HAS_OPTUNA = False

logger = logging.getLogger(__name__)

_settings = get_settings()

_OPTUNA_INTEGRATION_GUIDANCE = (
    "Optuna strategy requested but Optuna's sklearn integration is unavailable. "
    "Install 'optuna' together with 'optuna-integration' and restart the worker."
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


@dataclass(frozen=True)
class TuningInputs:
    frame: pd.DataFrame
    node_config: Dict[str, Any]
    dataset_meta: Optional[Dict[str, Any]]
    upstream_order: List[Any]


@dataclass(frozen=True)
class TrainingDataBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: Optional[pd.DataFrame]
    y_validation: Optional[pd.Series]
    feature_columns: List[str]
    target_meta: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class SearchExecutionResult:
    searcher: Any
    summary: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    best_estimator: Any


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise ValueError("Search space JSON is invalid") from exc
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


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (str, bool)):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, dict)):
        return value
    return repr(value)


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
            if not text:
                coerced[key] = None
                continue
            if text.lower() in {"none", "null"}:
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

    # Elastic net combinations require l1_ratio. Remove unsupported parameters to avoid estimator failures.
    search_space.pop("l1_ratio", None)


def _build_optuna_distributions(search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    if not _HAS_OPTUNA or CategoricalDistribution is None:
        raise RuntimeError(
            "Optuna is not installed. Install the optional dependency to use the optuna search strategy."
        )

    distributions: Dict[str, Any] = {}
    for key, candidates in search_space.items():
        if not candidates:
            continue

        unique_values: List[Any] = []
        seen: set[str] = set()
        for candidate in candidates:
            marker = repr(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            unique_values.append(candidate)

        if not unique_values:
            continue

        distributions[key] = CategoricalDistribution(unique_values)

    if not distributions:
        raise ValueError("Search space is empty after filtering unsupported hyperparameters for the optuna strategy.")

    return distributions


def _create_optuna_searcher(
    optuna_kwargs: Dict[str, Any],
    search_config: SearchConfiguration,
    # Injection points for testing — if provided these will be used
    # instead of the module-level OptunaSearchCV/TPESampler bindings.
    _search_cls: Any = None,
    _sampler_cls: Any = None,
    _accepts_sampler_fn: Optional[Any] = None,
):
    """
    Create an Optuna-backed searcher. The three underscore-prefixed
    parameters are intended for tests to inject dummy classes or
    callbacks to avoid brittle monkeypatching of module-level symbols.
    """
    # Resolve runtime bindings (prefer explicit injections)
    search_cls = _search_cls if _search_cls is not None else OptunaSearchCV
    sampler_cls = _sampler_cls if _sampler_cls is not None else TPESampler
    accepts_sampler = _accepts_sampler_fn if _accepts_sampler_fn is not None else _optuna_accepts_sampler

    if not _HAS_OPTUNA or search_cls is None:  # pragma: no cover - guarded by caller
        raise RuntimeError(_OPTUNA_INTEGRATION_GUIDANCE)

    kwargs: Dict[str, Any] = dict(optuna_kwargs)
    sampler_added = False

    if sampler_cls is not None and accepts_sampler():
        kwargs["sampler"] = sampler_cls(
            seed=search_config.random_state if search_config.random_state is not None else None
        )
        sampler_added = True

    try:
        return search_cls(**kwargs)
    except TypeError as exc:
        if sampler_added and "sampler" in str(exc):
            logger.debug("OptunaSearchCV rejected 'sampler'; retrying without sampler. Error: %s", exc)
            kwargs.pop("sampler", None)
            return search_cls(**kwargs)
        raise


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


_OPTUNA_ACCEPTS_SAMPLER: Optional[bool] = None


def _optuna_accepts_sampler() -> bool:
    global _OPTUNA_ACCEPTS_SAMPLER

    if _OPTUNA_ACCEPTS_SAMPLER is not None:
        return _OPTUNA_ACCEPTS_SAMPLER

    if not _HAS_OPTUNA or OptunaSearchCV is None:
        _OPTUNA_ACCEPTS_SAMPLER = False
        return False

    try:
        signature = inspect.signature(OptunaSearchCV.__init__)  # type: ignore[attr-defined]
    except (TypeError, ValueError):  # pragma: no cover - dynamic loading edge cases
        _OPTUNA_ACCEPTS_SAMPLER = False
        return False

    _OPTUNA_ACCEPTS_SAMPLER = "sampler" in signature.parameters
    return _OPTUNA_ACCEPTS_SAMPLER


def _summarize_results(cv_results_: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    if not cv_results_:
        return []

    params_list = cv_results_.get("params", [])
    mean_test = cv_results_.get("mean_test_score", [])
    std_test = cv_results_.get("std_test_score", [])
    mean_train = cv_results_.get("mean_train_score", [])
    mean_fit_time = cv_results_.get("mean_fit_time", [])
    rank_test = cv_results_.get("rank_test_score", [])

    records: List[Dict[str, Any]] = []
    for idx, params in enumerate(params_list):
        record = {
            "rank": int(rank_test[idx]) if idx < len(rank_test) else idx + 1,
            "mean_test_score": float(mean_test[idx]) if idx < len(mean_test) else None,
            "std_test_score": float(std_test[idx]) if idx < len(std_test) else None,
            "mean_train_score": float(mean_train[idx]) if idx < len(mean_train) else None,
            "mean_fit_time": float(mean_fit_time[idx]) if idx < len(mean_fit_time) else None,
            "params": {key: _serialize_value(value) for key, value in (params or {}).items()},
        }
        records.append(record)

    records.sort(key=lambda item: (item.get("rank") or 0, -(item.get("mean_test_score") or 0)), reverse=False)
    return records[:limit]


async def _load_tuning_inputs(session, job) -> TuningInputs:
    frame, node_config, dataset_meta, upstream_order = await _resolve_training_inputs(session, job)
    config_dict = node_config if isinstance(node_config, dict) else {}
    if isinstance(upstream_order, list):
        upstream_list = upstream_order
    elif upstream_order is None:
        upstream_list = []
    else:
        try:
            upstream_list = list(upstream_order)
        except TypeError:  # pragma: no cover - safeguard for unexpected input types
            upstream_list = [upstream_order]

    return TuningInputs(
        frame=frame,
        node_config=config_dict,
        dataset_meta=dataset_meta,
        upstream_order=upstream_list,
    )


def _extract_target_column(node_config: Dict[str, Any], job) -> str:
    job_metadata = job.job_metadata or {}
    target_column = (
        node_config.get("target_column")
        or node_config.get("targetColumn")
        or job_metadata.get("target_column")
    )
    if not target_column:
        raise ValueError("Tuning configuration missing target column")
    return str(target_column)


def _extract_problem_type_hint(node_config: Dict[str, Any]) -> str:
    raw = None
    if isinstance(node_config, dict):
        raw = node_config.get("problem_type") or node_config.get("problemType")
    if isinstance(raw, str):
        text = raw.strip()
        if text:
            return text
    return "auto"


def _resolve_model_spec_from_job(job) -> Any:
    model_type = getattr(job, "model_type", None)
    if not model_type:
        raise ValueError("Tuning job missing model_type")
    return get_model_spec(model_type)


def _build_training_data_bundle(frame: pd.DataFrame, target_column: str) -> TrainingDataBundle:
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        _,
        _,
        feature_columns,
        target_meta,
    ) = _prepare_training_data(frame, target_column)

    return TrainingDataBundle(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        feature_columns=feature_columns,
        target_meta=target_meta,
    )


def _determine_problem_type(problem_type_hint: str, spec: Any) -> str:
    normalized = (problem_type_hint or "").strip().lower()
    if normalized in {"classification", "regression"}:
        return normalized
    return spec.problem_type


def _prepare_search_parameters(
    job,
    node_config: Dict[str, Any],
    spec: Any,
    search_config: SearchConfiguration,
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    base_params = dict(spec.default_params)
    raw_baseline: Dict[str, Any] = {}

    if job.baseline_hyperparameters:
        raw_baseline.update(job.baseline_hyperparameters)
    else:
        baseline_config = node_config.get("baseline_hyperparameters") if isinstance(node_config, dict) else None
        if isinstance(baseline_config, str):
            try:
                raw_baseline.update(_sanitize_parameters(baseline_config))
            except ValueError as exc:
                raise ValueError("Baseline hyperparameters JSON is invalid") from exc

    raw_baseline = _coerce_none_strings(raw_baseline)

    metadata_fields = get_hyperparameters_for_model(spec.key)
    allowed_key_candidates = {field.get("name") for field in metadata_fields if isinstance(field, dict)}
    allowed_keys: set[str] = {key for key in allowed_key_candidates if isinstance(key, str)}
    allowed_keys.update(spec.default_params.keys())
    allowed_keys.update(search_config.search_space.keys())

    filtered_baseline = _filter_supported_parameters(raw_baseline, allowed_keys)
    base_params.update(filtered_baseline)

    filtered_search_space = _filter_supported_parameters(search_config.search_space, allowed_keys)
    if not filtered_search_space:
        raise ValueError(
            "Search space is empty after filtering unsupported hyperparameters for this estimator."
        )

    if spec.key == "logistic_regression":
        _sanitize_logistic_regression_hyperparameters(base_params, filtered_search_space)
        if not filtered_search_space:
            raise ValueError(
                "Search space is empty after normalizing logistic regression parameters. Use 'ovr' or 'multinomial' for multi_class."
            )

    return base_params, filtered_search_space


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


def _build_searcher(
    spec: Any,
    search_config: SearchConfiguration,
    base_params: Dict[str, Any],
    filtered_search_space: Dict[str, List[Any]],
    splitter,
):
    estimator = spec.factory(**base_params)
    search_kwargs = {
        "estimator": estimator,
        "scoring": search_config.scoring,
        "cv": splitter,
        "n_jobs": 1,
        "return_train_score": True,
        "refit": True,
    }

    if search_config.strategy == "grid":
        return GridSearchCV(param_grid=filtered_search_space, **search_kwargs)
    if search_config.strategy == "halving":
        return HalvingGridSearchCV(
            param_grid=filtered_search_space,
            resource="n_samples",
            **search_kwargs,
        )
    if search_config.strategy == "halving_random":
        halving_random_kwargs: Dict[str, Any] = {
            **search_kwargs,
            "param_distributions": filtered_search_space,
            "random_state": search_config.random_state,
        }
        if search_config.n_iterations is not None and search_config.n_iterations > 0:
            halving_random_kwargs["n_candidates"] = int(max(1, search_config.n_iterations))
        return HalvingRandomSearchCV(**halving_random_kwargs)
    if search_config.strategy == "optuna":
        if not _HAS_OPTUNA or OptunaSearchCV is None:
            raise RuntimeError(_OPTUNA_INTEGRATION_GUIDANCE)

        optuna_distributions = _build_optuna_distributions(filtered_search_space)
        iterations = search_config.n_iterations
        if iterations is None or iterations <= 0:
            iterations = min(30, max(len(optuna_distributions), 1) * 10)

        optuna_kwargs: Dict[str, Any] = {
            "estimator": estimator,
            "param_distributions": optuna_distributions,
            "n_trials": int(iterations),
            "cv": splitter,
            "scoring": search_config.scoring,
            "refit": True,
            "return_train_score": True,
            "n_jobs": 1,
        }
        if search_config.random_state is not None:
            optuna_kwargs["random_state"] = search_config.random_state

        return _create_optuna_searcher(optuna_kwargs, search_config)

    iterations = search_config.n_iterations
    if iterations is None or iterations <= 0:
        iterations = min(20, max(len(filtered_search_space), 1) * 5)
    return RandomizedSearchCV(
        param_distributions=filtered_search_space,
        n_iter=int(iterations),
        random_state=search_config.random_state,
        **search_kwargs,
    )


def _execute_search(
    searcher: Any,
    training_data: TrainingDataBundle,
    spec_key: str,
    resolved_problem_type: str,
    target_column: str,
    search_config: SearchConfiguration,
    cv_config: CrossValidationConfig,
) -> SearchExecutionResult:
    searcher.fit(training_data.X_train, training_data.y_train)
    summary = _summarize_results(searcher.cv_results_)
    metrics = _build_search_metrics(
        searcher,
        training_data,
        spec_key,
        resolved_problem_type,
        target_column,
        search_config,
        cv_config,
    )
    return SearchExecutionResult(
        searcher=searcher,
        summary=summary,
        metrics=metrics,
        best_estimator=searcher.best_estimator_,
    )


def _build_search_metrics(
    searcher: Any,
    training_data: TrainingDataBundle,
    spec_key: str,
    resolved_problem_type: str,
    target_column: str,
    search_config: SearchConfiguration,
    cv_config: CrossValidationConfig,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "search": {
            "strategy": search_config.strategy,
            "selected_strategy": search_config.selected_strategy,
            "scoring": search_config.scoring or "default",
            "n_candidates": len(searcher.cv_results_.get("params", [])),
            "best_index": int(searcher.best_index_),
            "best_score": float(searcher.best_score_),
        },
        "row_counts": {"train": int(training_data.y_train.shape[0])},
        "feature_columns": training_data.feature_columns,
        "target_column": target_column,
        "model_type": spec_key,
        "cross_validation": {
            "strategy": cv_config.strategy,
            "folds": cv_config.folds,
            "shuffle": cv_config.shuffle,
            "random_state": cv_config.random_state,
            "refit_strategy": cv_config.refit_strategy,
        },
    }

    if search_config.strategy in {"halving", "halving_random"}:
        if hasattr(searcher, "n_resources_"):
            metrics["search"]["n_resources_per_step"] = [
                int(value)
                for value in getattr(searcher, "n_resources_", [])
            ]
        if hasattr(searcher, "n_candidates_"):
            metrics["search"]["n_candidates_per_step"] = [
                int(value)
                for value in getattr(searcher, "n_candidates_", [])
            ]

    if search_config.strategy == "optuna" and hasattr(searcher, "n_trials_"):
        metrics["search"]["n_trials"] = int(getattr(searcher, "n_trials_", 0))

    X_train_features = training_data.X_train
    if resolved_problem_type == "classification":
        y_train_array = training_data.y_train.astype(int).to_numpy()
        metrics["train"] = _classification_metrics(searcher.best_estimator_, X_train_features, y_train_array)
    else:
        y_train_array = training_data.y_train.astype(float).to_numpy()
        metrics["train"] = _regression_metrics(searcher.best_estimator_, X_train_features, y_train_array)

    if (
        training_data.X_validation is not None
        and training_data.y_validation is not None
        and not training_data.X_validation.empty
        and training_data.y_validation.shape[0] > 0
    ):
        X_val_features = training_data.X_validation
        if resolved_problem_type == "classification":
            y_val_array = training_data.y_validation.astype(int).to_numpy()
            metrics["row_counts"]["validation"] = int(y_val_array.shape[0])
            metrics["validation"] = _classification_metrics(searcher.best_estimator_, X_val_features, y_val_array)
        else:
            y_val_array = training_data.y_validation.astype(float).to_numpy()
            metrics["row_counts"]["validation"] = int(y_val_array.shape[0])
            metrics["validation"] = _regression_metrics(searcher.best_estimator_, X_val_features, y_val_array)
    else:
        metrics["row_counts"]["validation"] = 0

    return metrics


def _persist_best_estimator(job, estimator) -> Optional[str]:
    if estimator is None:
        return None
    try:
        artifact_dir = Path(_settings.TRAINING_ARTIFACT_DIR) / job.pipeline_id / "tuning"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{job.id}_run{job.run_number}.joblib"
        joblib.dump(estimator, artifact_path, compress=("gzip", 3))
        return str(artifact_path)
    except Exception as exc:  # pragma: no cover - artifact persistence failure shouldn't abort job
        logger.warning("Failed to persist best estimator for tuning job %s: %s", job.id, exc)
        return None


def _build_metadata_update(
    resolved_problem_type: str,
    target_column: str,
    feature_columns: List[str],
    cv_config: CrossValidationConfig,
    dataset_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metadata_update: Dict[str, Any] = {
        "resolved_problem_type": resolved_problem_type,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "cross_validation": {
            "enabled": True,
            "strategy": cv_config.strategy,
            "folds": cv_config.folds,
            "shuffle": cv_config.shuffle,
            "random_state": cv_config.random_state,
            "refit_strategy": cv_config.refit_strategy,
        },
    }
    if dataset_meta:
        metadata_update["dataset"] = dataset_meta
    return metadata_update


async def _run_hyperparameter_tuning_workflow(job_id: str) -> None:
    await _ensure_database_ready()

    async with get_database_session(expire_on_commit=False) as session:
        job = await get_tuning_job(session, job_id)
        if job is None:
            logger.warning("Tuning job %s not found; skipping", job_id)
            return

        try:
            job = await update_tuning_job_status(
                session,
                job,
                status=HyperparameterTuningJobStatus.RUNNING,
            )

            inputs = await _load_tuning_inputs(session, job)
            target_column = _extract_target_column(inputs.node_config, job)
            problem_type_hint = _extract_problem_type_hint(inputs.node_config)
            spec = _resolve_model_spec_from_job(job)
            training_data = _build_training_data_bundle(inputs.frame, target_column)
            search_config = _build_search_configuration(job, inputs.node_config)
            resolved_problem_type = _determine_problem_type(problem_type_hint, spec)

            # Adjust scoring for multiclass classification when users select
            # metrics that default to binary averaging (e.g. 'f1', 'precision', 'recall').
            # For multiclass targets we prefer the weighted variant to avoid
            # scorer errors during cross-validation.
            scoring_override = search_config.scoring
            if (
                resolved_problem_type == "classification"
                and isinstance(scoring_override, str)
                and scoring_override.strip()
            ):
                normalized_scoring = scoring_override.strip()
                # determine number of classes from target metadata if available
                n_classes = None
                try:
                    if isinstance(training_data.target_meta, dict):
                        cats = training_data.target_meta.get("categories")
                        if isinstance(cats, (list, tuple)):
                            n_classes = len(cats)
                except Exception:
                    n_classes = None
                if n_classes is None:
                    try:
                        n_classes = int(len(training_data.y_train.astype(int).unique()))
                    except Exception:
                        n_classes = None

                if n_classes is not None and n_classes > 2:
                    mapping = {
                        "f1": "f1_weighted",
                        "precision": "precision_weighted",
                        "recall": "recall_weighted",
                    }
                    lower = normalized_scoring.lower()
                    if lower in mapping and normalized_scoring != mapping[lower]:
                        logger.info(
                            "Adjusting scoring metric '%s' -> '%s' for multiclass target (%s classes)",
                            normalized_scoring,
                            mapping[lower],
                            n_classes,
                        )
                        # create a new SearchConfiguration with adjusted scoring
                        search_config = SearchConfiguration(
                            strategy=search_config.strategy,
                            selected_strategy=search_config.selected_strategy,
                            search_space=search_config.search_space,
                            n_iterations=search_config.n_iterations,
                            scoring=mapping[lower],
                            random_state=search_config.random_state,
                            cross_validation=search_config.cross_validation,
                        )

            base_params, filtered_search_space = _prepare_search_parameters(
                job,
                inputs.node_config,
                spec,
                search_config,
            )

            cv_config = _resolve_cv_config(search_config)
            splitter = _build_cv_splitter(resolved_problem_type, cv_config, training_data.y_train)

            searcher = _build_searcher(
                spec,
                search_config,
                base_params,
                filtered_search_space,
                splitter,
            )

            search_result = _execute_search(
                searcher,
                training_data,
                spec.key,
                resolved_problem_type,
                target_column,
                search_config,
                cv_config,
            )

            artifact_uri = _persist_best_estimator(job, search_result.best_estimator)
            if artifact_uri:
                search_result.metrics["artifact_uri"] = artifact_uri

            metadata_update = _build_metadata_update(
                resolved_problem_type,
                target_column,
                training_data.feature_columns,
                cv_config,
                inputs.dataset_meta,
            )

            await update_tuning_job_status(
                session,
                job,
                status=HyperparameterTuningJobStatus.SUCCEEDED,
                metrics=search_result.metrics,
                results=search_result.summary,
                best_params={
                    key: _serialize_value(value)
                    for key, value in (search_result.searcher.best_params_ or {}).items()
                },
                best_score=float(search_result.searcher.best_score_),
                artifact_uri=artifact_uri,
                metadata=metadata_update,
            )
        except Exception as exc:  # pragma: no cover - defensive guard for worker runtime
            logger.exception("Tuning job %s failed", job_id)
            await update_tuning_job_status(
                session,
                job,
                status=HyperparameterTuningJobStatus.FAILED,
                error_message=str(exc),
            )


@celery_app.task(name="core.feature_engineering.nodes.modeling.hyperparameter_tuning.run")
def run_hyperparameter_tuning(job_id: str) -> None:
    """Celery entrypoint for hyperparameter tuning jobs."""

    asyncio.run(_run_hyperparameter_tuning_workflow(job_id))


def dispatch_hyperparameter_tuning_job(job_id: str) -> None:
    """Queue a hyperparameter tuning job via Celery."""

    run_hyperparameter_tuning.delay(job_id)
