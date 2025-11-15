"""Search execution helpers for hyperparameter tuning."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from ...shared import (
    ConvergenceWarning,
    CrossValidationConfig,
    SearchConfiguration,
    _classification_metrics,
    _extract_warning_messages,
    _regression_metrics,
    _summarize_results,
)
from .data_bundle import TrainingDataBundle


@dataclass(frozen=True)
class SearchExecutionResult:
    searcher: Any
    summary: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    best_estimator: Any
    warnings: List[str]


def _execute_search(
    searcher: Any,
    training_data: TrainingDataBundle,
    spec_key: str,
    resolved_problem_type: str,
    target_column: str,
    search_config: SearchConfiguration,
    cv_config: CrossValidationConfig,
) -> SearchExecutionResult:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        warnings.simplefilter("always", ConvergenceWarning)
        searcher.fit(training_data.X_train, training_data.y_train)

    warning_messages = _extract_warning_messages(caught_warnings)
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
    if warning_messages:
        metrics.setdefault("warnings", warning_messages)
    return SearchExecutionResult(
        searcher=searcher,
        summary=summary,
        metrics=metrics,
        best_estimator=searcher.best_estimator_,
        warnings=warning_messages,
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


__all__ = [
    "SearchExecutionResult",
    "_build_search_metrics",
    "_execute_search",
]
