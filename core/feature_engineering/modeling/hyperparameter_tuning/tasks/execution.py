"""Search execution helpers for hyperparameter tuning."""

from __future__ import annotations

import sys
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

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


class StreamCapture:
    def __init__(self, original, line_callback: Callable[[str], None]):
        self.original = original
        self.line_callback = line_callback
        self.buffer = ""

    def write(self, text: str) -> None:
        if self.original:
            self.original.write(text)
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.line_callback(line)

    def flush(self) -> None:
        if self.original:
            self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


class StdoutProgressCapture(logging.Handler):
    def __init__(self, callback: Callable[[int, str], None]):
        logging.Handler.__init__(self)
        self.callback = callback
        self.total_fits = 0
        self.current_fit = 0
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stdout_capture: Optional[StreamCapture] = None
        self.stderr_capture: Optional[StreamCapture] = None
        self._processed_lines = set()

    def __enter__(self):
        self.stdout_capture = StreamCapture(self.original_stdout, self._parse_line)
        self.stderr_capture = StreamCapture(self.original_stderr, self._parse_line)
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        
        # Attach to root logger to capture everything
        root_logger = logging.getLogger()
        root_logger.addHandler(self)
        
        # Explicitly attach to optuna logger to ensure capture
        optuna_logger = logging.getLogger("optuna")
        optuna_logger.addHandler(self)
        # Ensure optuna propagates if it wasn't already
        optuna_logger.propagate = True
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        logging.getLogger().removeHandler(self)
        logging.getLogger("optuna").removeHandler(self)

    def emit(self, record):
        try:
            msg = self.format(record)
            self._parse_line(msg)
        except Exception:
            self.handleError(record)

    def _parse_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return

        # Deduplication logic:
        # Optuna uses logging, so messages might arrive via 'emit' AND 'write' (if propagated to stdout).
        # We must deduplicate Optuna lines to avoid double-counting.
        # Sklearn uses direct stdout/stderr writes, so it only arrives via 'write'.
        # Sklearn lines (like "[CV] END...") can be identical across folds, so we MUST NOT deduplicate them.
        
        is_optuna = "Trial" in line and ("finished with value" in line or "pruned" in line)
        
        if is_optuna:
            line_hash = hash(line)
            if line_hash in self._processed_lines:
                return
            self._processed_lines.add(line_hash)

        # Sklearn verbose=2 format:
        # "Fitting 5 folds for each of 10 candidates, totalling 50 fits"
        if "totalling" in line and "fits" in line:
            try:
                parts = line.split("totalling")
                self.total_fits = int(parts[1].strip().split()[0])
                # Reset current fit count when a new search starts (e.g. Halving search iterations)
                self.current_fit = 0
            except Exception:
                pass
        
        # "[CV] END ...; score=... total time=..." or "[CV 1/5] END ..."
        if "[CV" in line and "END" in line:
            self.current_fit += 1
            if self.total_fits > 0:
                pct = int((self.current_fit / self.total_fits) * 100)
                # Cap at 99 until done
                pct = min(pct, 99)
                self.callback(pct, f"Fit {self.current_fit}/{self.total_fits}")
        
        # Optuna format: "Trial 0 finished with value: ..."
        # Also handle "Trial 0 pruned."
        if is_optuna:
            # We don't know total trials easily from stdout, but we can increment
            self.current_fit += 1
            # Heuristic: if we don't know total, just show count
            if self.total_fits > 0:
                 pct = int((self.current_fit / self.total_fits) * 100)
                 pct = min(pct, 99)
                 self.callback(pct, f"Trial {self.current_fit}/{self.total_fits}")
            else:
                 # If we missed the total count, just show trial number
                 self.callback(0, f"Trial {self.current_fit}")

    def set_total_trials(self, n_trials: int):
        self.total_fits = n_trials


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
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> SearchExecutionResult:
    # Force Optuna verbosity to INFO so that StdoutProgressCapture can catch the logs
    if search_config.strategy == "optuna":
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.INFO)
        except ImportError:
            pass

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        warnings.simplefilter("always", ConvergenceWarning)
        
        if progress_callback:
            capture = StdoutProgressCapture(progress_callback)
            # Try to set total trials if available in searcher
            if hasattr(searcher, "n_trials"):
                capture.set_total_trials(searcher.n_trials)
            elif hasattr(searcher, "n_iter"):
                # For RandomizedSearchCV
                capture.set_total_trials(searcher.n_iter * cv_config.folds)
            elif hasattr(searcher, "param_grid"):
                 # For GridSearchCV
                 # This is harder to calculate without iterating, but sklearn prints it.
                 pass

            with capture:
                searcher.fit(training_data.X_train, training_data.y_train)
        else:
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
