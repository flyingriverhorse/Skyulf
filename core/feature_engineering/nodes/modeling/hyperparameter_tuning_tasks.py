"""Celery tasks that execute hyperparameter tuning workflows."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from config import get_settings
from core.database.models import get_database_session

from .hyperparameter_tuning_jobs import get_tuning_job, update_tuning_job_status
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

logger = logging.getLogger(__name__)

_settings = get_settings()


@dataclass(frozen=True)
class SearchConfiguration:
    strategy: str
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


def _build_search_configuration(job, node_config: Dict[str, Any]) -> SearchConfiguration:
    strategy = str(job.search_strategy or node_config.get("search_strategy", "random")).strip().lower()
    if strategy not in {"grid", "random"}:
        strategy = "random"

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

    cross_validation = job.cross_validation or node_config
    cv_config = _coerce_cross_validation_config(cross_validation)
    if not cv_config.enabled:
        cv_config = CrossValidationConfig(True, "auto", max(cv_config.folds, 3), cv_config.shuffle, cv_config.random_state, cv_config.refit_strategy)

    return SearchConfiguration(
        strategy=strategy,
        search_space=search_space,
        n_iterations=n_iterations,
        scoring=scoring,
        random_state=random_state,
        cross_validation=cv_config,
    )


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

            frame, node_config, dataset_meta, upstream_order = await _resolve_training_inputs(session, job)

            target_column = (
                node_config.get("target_column")
                or node_config.get("targetColumn")
                or (job.job_metadata or {}).get("target_column")
            )
            if not target_column:
                raise ValueError("Tuning configuration missing target column")

            problem_type_hint = (
                (node_config.get("problem_type") or node_config.get("problemType") or "auto")
                if isinstance(node_config, dict)
                else "auto"
            )

            model_type = job.model_type
            if not model_type:
                raise ValueError("Tuning job missing model_type")

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

            search_config = _build_search_configuration(job, node_config)

            spec = get_model_spec(model_type)
            resolved_problem_type = problem_type_hint or spec.problem_type
            if resolved_problem_type not in {"classification", "regression"}:
                resolved_problem_type = spec.problem_type

            base_params = dict(spec.default_params)
            raw_baseline: Dict[str, Any] = {}
            if job.baseline_hyperparameters:
                raw_baseline.update(job.baseline_hyperparameters)
            elif isinstance(node_config.get("baseline_hyperparameters"), str):
                try:
                    raw_baseline.update(_sanitize_parameters(node_config.get("baseline_hyperparameters")))
                except ValueError as exc:
                    raise ValueError("Baseline hyperparameters JSON is invalid") from exc

            raw_baseline = _coerce_none_strings(raw_baseline)

            metadata_fields = get_hyperparameters_for_model(spec.key)
            allowed_hyperparameter_keys = {field.get("name") for field in metadata_fields if isinstance(field, dict)}
            allowed_hyperparameter_keys = {key for key in allowed_hyperparameter_keys if isinstance(key, str)}
            allowed_hyperparameter_keys.update(spec.default_params.keys())
            allowed_hyperparameter_keys.update(search_config.search_space.keys())

            filtered_baseline = _filter_supported_parameters(raw_baseline, allowed_hyperparameter_keys)
            base_params.update(filtered_baseline)

            filtered_search_space = _filter_supported_parameters(search_config.search_space, allowed_hyperparameter_keys)
            if not filtered_search_space:
                raise ValueError(
                    "Search space is empty after filtering unsupported hyperparameters for this estimator."
                )

            splitter = _build_cv_splitter(resolved_problem_type, search_config.cross_validation, y_train)

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
                searcher = GridSearchCV(param_grid=filtered_search_space, **search_kwargs)
            else:
                iterations = search_config.n_iterations
                if iterations is None or iterations <= 0:
                    iterations = min(20, max(len(filtered_search_space), 1) * 5)
                searcher = RandomizedSearchCV(
                    param_distributions=filtered_search_space,
                    n_iter=int(iterations),
                    random_state=search_config.random_state,
                    **search_kwargs,
                )

            searcher.fit(X_train, y_train)

            summary = _summarize_results(searcher.cv_results_)

            best_estimator = searcher.best_estimator_
            metrics: Dict[str, Any] = {
                "search": {
                    "strategy": search_config.strategy,
                    "scoring": search_config.scoring or "default",
                    "n_candidates": len(searcher.cv_results_.get("params", [])),
                    "best_index": int(searcher.best_index_),
                    "best_score": float(searcher.best_score_),
                },
                "row_counts": {"train": int(y_train.shape[0])},
                "feature_columns": feature_columns,
                "target_column": target_column,
                "model_type": spec.key,
                "cross_validation": {
                    "strategy": search_config.cross_validation.strategy,
                    "folds": search_config.cross_validation.folds,
                    "shuffle": search_config.cross_validation.shuffle,
                    "random_state": search_config.cross_validation.random_state,
                },
            }

            X_train_array = X_train.to_numpy(dtype=np.float64)
            if resolved_problem_type == "classification":
                y_train_array = y_train.astype(int).to_numpy()
                metrics["train"] = _classification_metrics(best_estimator, X_train_array, y_train_array)
            else:
                y_train_array = y_train.astype(float).to_numpy()
                metrics["train"] = _regression_metrics(best_estimator, X_train_array, y_train_array)

            if X_validation is not None and y_validation is not None and not X_validation.empty and y_validation.shape[0] > 0:
                X_val_array = X_validation.to_numpy(dtype=np.float64)
                if resolved_problem_type == "classification":
                    y_val_array = y_validation.astype(int).to_numpy()
                    metrics.setdefault("row_counts", {})["validation"] = int(y_val_array.shape[0])
                    metrics["validation"] = _classification_metrics(best_estimator, X_val_array, y_val_array)
                else:
                    y_val_array = y_validation.astype(float).to_numpy()
                    metrics.setdefault("row_counts", {})["validation"] = int(y_val_array.shape[0])
                    metrics["validation"] = _regression_metrics(best_estimator, X_val_array, y_val_array)
            else:
                metrics.setdefault("row_counts", {})["validation"] = 0

            artifact_uri: Optional[str] = None
            try:
                artifact_dir = Path(_settings.TRAINING_ARTIFACT_DIR) / job.pipeline_id / "tuning"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                artifact_path = artifact_dir / f"{job.id}_run{job.run_number}.joblib"
                joblib.dump(best_estimator, artifact_path, compress=("gzip", 3))
                artifact_uri = str(artifact_path)
                metrics["artifact_uri"] = artifact_uri
            except Exception as exc:  # pragma: no cover - artifact persistence failure shouldn't abort job
                logger.warning("Failed to persist best estimator for tuning job %s: %s", job.id, exc)

            metadata_update = {
                "resolved_problem_type": resolved_problem_type,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "cross_validation": {
                    "enabled": True,
                    "strategy": search_config.cross_validation.strategy,
                    "folds": search_config.cross_validation.folds,
                    "shuffle": search_config.cross_validation.shuffle,
                    "random_state": search_config.cross_validation.random_state,
                    "refit_strategy": search_config.cross_validation.refit_strategy,
                },
            }
            if dataset_meta:
                metadata_update["dataset"] = dataset_meta

            await update_tuning_job_status(
                session,
                job,
                status=HyperparameterTuningJobStatus.SUCCEEDED,
                metrics=metrics,
                results=summary,
                best_params={key: _serialize_value(value) for key, value in (searcher.best_params_ or {}).items()},
                best_score=float(searcher.best_score_),
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
