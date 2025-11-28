"""Training execution helpers (fit, evaluate, persist)."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...shared import (
    ConvergenceWarning,
    CrossValidationConfig,
    _build_cv_splitter,
    _classification_metrics,
    _extract_warning_messages,
    _persist_training_artifact,
    _regression_metrics,
    _resolve_problem_type_hint,
    _write_transformer_debug_snapshot,
)
from ..registry import get_model_spec
from .transformers import _build_transformer_plan, _collect_transformers

logger = logging.getLogger(__name__)


def _merge_model_params(defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(defaults)
    if overrides:
        merged.update({key: value for key, value in overrides.items() if value is not None})
    return merged


def _normalize_model_params(spec, params: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(params)
    sanitizer = getattr(spec, "sanitize_params", None)
    if callable(sanitizer):
        try:
            normalized = sanitizer(normalized)
        except Exception:
            logger.exception("Failed to sanitize params for %s; using defaults", spec.key)
            normalized = dict(spec.default_params)
    return normalized


def _aggregate_cv_metrics(fold_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_values: Dict[str, List[float]] = {}
    for entry in fold_entries:
        metrics = entry.get("metrics") or {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                metric_values.setdefault(key, []).append(float(value))

    if not metric_values:
        return {"mean": {}, "std": {}}

    mean_summary = {key: float(np.mean(values)) for key, values in metric_values.items()}
    std_summary = {key: float(np.std(values)) for key, values in metric_values.items()}
    return {"mean": mean_summary, "std": std_summary}


def _run_cross_validation(
    spec,
    params: Dict[str, Any],
    resolved_problem_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_config: CrossValidationConfig,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Optional[Dict[str, Any]]:
    if not cv_config.enabled:
        return None

    if X_train.shape[0] < cv_config.folds:
        return {
            "status": "skipped",
            "reason": "insufficient_rows",
            "requested_folds": cv_config.folds,
            "available_rows": int(X_train.shape[0]),
        }

    try:
        splitter = _build_cv_splitter(resolved_problem_type, cv_config, y_train)
    except ValueError as exc:  # pragma: no cover - defensive guard
        logger.warning("Cross-validation splitter could not be constructed: %s", exc)
        return {
            "status": "skipped",
            "reason": "invalid_configuration",
            "message": str(exc),
        }

    y_array = y_train.astype(int).to_numpy() if resolved_problem_type == "classification" else y_train.astype(float).to_numpy()

    fold_entries: List[Dict[str, Any]] = []
    try:
        for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X_train, y_array), start=1):
            if progress_callback:
                pct = int(((fold_index - 1) / cv_config.folds) * 100)
                progress_callback(pct, f"Fold {fold_index}/{cv_config.folds}")

            if val_idx.size == 0:
                continue

            model = spec.factory(**params)
            model.fit(X_train.iloc[train_idx], y_array[train_idx])

            validation_features = X_train.iloc[val_idx]
            validation_target = y_array[val_idx]

            metrics = (
                _classification_metrics(model, validation_features, validation_target)
                if resolved_problem_type == "classification"
                else _regression_metrics(model, validation_features, validation_target)
            )

            fold_entries.append(
                {
                    "fold": fold_index,
                    "row_count": int(val_idx.size),
                    "metrics": metrics,
                }
            )
    except ValueError as exc:  # pragma: no cover - defensive guard
        logger.warning("Cross-validation split failed: %s", exc)
        return {
            "status": "skipped",
            "reason": "split_failed",
            "message": str(exc),
        }

    if not fold_entries:
        return {"status": "skipped", "reason": "no_valid_folds"}

    summary = _aggregate_cv_metrics(fold_entries)
    return {
        "status": "completed",
        "strategy": cv_config.strategy,
        "folds": len(fold_entries),
        "shuffle": cv_config.shuffle,
        "random_state": cv_config.random_state,
        "refit_strategy": cv_config.refit_strategy,
        "metrics": summary,
        "folds_detail": fold_entries,
    }


def _prepare_refit_dataset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: Optional[pd.DataFrame],
    y_validation: Optional[pd.Series],
    cv_config: CrossValidationConfig,
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    if (
        cv_config.refit_strategy == "train_plus_validation"
        and X_validation is not None
        and y_validation is not None
        and not X_validation.empty
        and y_validation.shape[0] > 0
    ):
        combined_X = pd.concat([X_train, X_validation], ignore_index=True)
        combined_y = pd.concat([y_train, y_validation], ignore_index=True)
        return combined_X, combined_y, True
    return X_train, y_train, False


def _prepare_target_array(target: pd.Series, problem_type: str) -> np.ndarray:
    return target.astype(int).to_numpy() if problem_type == "classification" else target.astype(float).to_numpy()


def _initialize_metrics(
    feature_columns: List[str],
    target_column: str,
    model_type: str,
    version: int,
    refit_strategy: str,
    used_validation: bool,
    cv_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "row_counts": {},
        "feature_columns": feature_columns,
        "target_column": target_column,
        "model_type": model_type,
        "version": version,
        "refit_strategy": refit_strategy,
        "validation_used_for_training": used_validation,
    }
    if cv_summary is not None:
        metrics["cross_validation"] = cv_summary
    return metrics


def _add_split_metrics(
    metrics: Dict[str, Any],
    split_name: str,
    model,
    X: Optional[pd.DataFrame],
    y: Optional[pd.Series],
    problem_type: str,
) -> None:
    if X is None or y is None or X.empty or y.shape[0] <= 0:
        metrics.setdefault("row_counts", {})[split_name] = 0
        return

    y_array = _prepare_target_array(y, problem_type)
    metrics["row_counts"][split_name] = int(y_array.shape[0])
    metrics[split_name] = (
        _classification_metrics(model, X, y_array)
        if problem_type == "classification"
        else _regression_metrics(model, X, y_array)
    )


def _train_and_save_model(
    model_type: str,
    hyperparameters: Optional[Dict[str, Any]],
    problem_type_hint: Optional[str],
    target_column: str,
    feature_columns: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: Optional[pd.DataFrame],
    y_validation: Optional[pd.Series],
    X_test: Optional[pd.DataFrame],
    y_test: Optional[pd.Series],
    artifact_root: str,
    pipeline_id: str,
    job_id: str,
    version: int,
    cv_config: CrossValidationConfig,
    upstream_node_order: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[str, Dict[str, Any], str, List[str]]:
    """Fit model synchronously, persist artifacts/metrics, and collect warnings."""

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        warnings.simplefilter("always", ConvergenceWarning)

        spec = get_model_spec(model_type)
        resolved_problem_type = _resolve_problem_type_hint(problem_type_hint, spec.problem_type)
        params = _merge_model_params(spec.default_params, hyperparameters)
        params = _normalize_model_params(spec, params)

        if progress_callback:
            progress_callback(0, "Starting training...")

        cv_summary = _run_cross_validation(
            spec, 
            params, 
            resolved_problem_type, 
            X_train, 
            y_train, 
            cv_config,
            progress_callback=progress_callback
        )

        fit_features, fit_target, used_validation = _prepare_refit_dataset(
            X_train,
            y_train,
            X_validation,
            y_validation,
            cv_config,
        )

        if progress_callback:
            progress_callback(90, "Training final model")

        final_model = spec.factory(**params)
        fit_target_array = _prepare_target_array(fit_target, resolved_problem_type)
        final_model.fit(fit_features, fit_target_array)

        if progress_callback:
            progress_callback(100, "Training completed")

        metrics = _initialize_metrics(
            feature_columns,
            target_column,
            spec.key,
            version,
            cv_config.refit_strategy,
            used_validation,
            cv_summary,
        )
        metrics["row_counts"]["train"] = int(fit_target_array.shape[0])
        metrics["train"] = (
            _classification_metrics(final_model, fit_features, fit_target_array)
            if resolved_problem_type == "classification"
            else _regression_metrics(final_model, fit_features, fit_target_array)
        )

        _add_split_metrics(metrics, "validation", final_model, X_validation, y_validation, resolved_problem_type)
        _add_split_metrics(metrics, "test", final_model, X_test, y_test, resolved_problem_type)

        transformers = _collect_transformers(pipeline_id)
        transformer_plan = _build_transformer_plan(transformers, upstream_node_order)

        artifact_data = {
            "model": final_model,
            "model_type": spec.key,
            "problem_type": resolved_problem_type,
            "feature_columns": feature_columns,
            "transformers": transformers,
            "transformer_plan": transformer_plan,
            "transformer_bundle_version": 1,
            "version": version,
        }

        _write_transformer_debug_snapshot(transformers, transformer_plan)

        artifact_path = _persist_training_artifact(
            artifact_root,
            pipeline_id,
            job_id,
            version,
            artifact_data,
        )

        metrics["artifact_uri"] = str(artifact_path)

    warning_messages = _extract_warning_messages(caught_warnings)
    if warning_messages:
        metrics.setdefault("warnings", warning_messages)

    return str(artifact_path), metrics, resolved_problem_type, warning_messages


__all__ = ["_train_and_save_model"]
