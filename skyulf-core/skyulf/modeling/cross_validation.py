"""Cross-validation logic for V2 modeling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
)

from ..engines import SkyulfDataFrame
from ..engines.sklearn_bridge import SklearnBridge

if TYPE_CHECKING:
    from .base import BaseModelApplier, BaseModelCalculator

from .evaluation.common import sanitize_metrics
from .evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)


def _aggregate_metrics(
    fold_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregates metrics across folds (mean and std)."""
    if not fold_metrics:
        return {}

    keys = fold_metrics[0].keys()
    aggregated = {}

    for key in keys:
        values = [m.get(key, np.nan) for m in fold_metrics]
        # Filter nans
        values = [v for v in values if np.isfinite(v)]

        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return aggregated


def perform_cross_validation(
    calculator: BaseModelCalculator,
    applier: BaseModelApplier,
    X: Union[pd.DataFrame, SkyulfDataFrame],
    y: Union[pd.Series, Any],
    config: Dict[str, Any],
    n_folds: int = 5,
    cv_type: str = "k_fold",  # k_fold, stratified_k_fold, time_series_split, shuffle_split
    shuffle: bool = True,
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Performs K-Fold cross-validation.

    Args:
        calculator: The model calculator (fit logic).
        applier: The model applier (predict logic).
        X: Features.
        y: Target.
        config: Model configuration.
        n_folds: Number of folds.
        cv_type: Type of CV.
        shuffle: Whether to shuffle data before splitting (for KFold/Stratified).
        random_state: Random seed for shuffling.
        progress_callback: Optional callback(current_fold, total_folds).
        log_callback: Optional callback for logging messages.

    Returns:
        Dict containing aggregated metrics and per-fold details.
    """

    problem_type = calculator.problem_type

    if log_callback:
        log_callback(f"Starting Cross-Validation (Folds: {n_folds}, Type: {cv_type})")

    # 1. Setup Splitter
    if cv_type == "time_series_split":
        splitter = TimeSeriesSplit(n_splits=n_folds)
    elif cv_type == "shuffle_split":
        splitter = ShuffleSplit(
            n_splits=n_folds, test_size=0.2, random_state=random_state
        )
    elif cv_type == "stratified_k_fold" and problem_type == "classification":
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
    else:
        # Default to KFold
        splitter = KFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    fold_results = []

    # Ensure numpy for splitting using the Bridge
    X_arr, y_arr = SklearnBridge.to_sklearn((X, y))

    # 2. Iterate Folds
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_arr, y_arr)):
        if progress_callback:
            progress_callback(fold_idx + 1, n_folds)

        if log_callback:
            log_callback(f"Processing Fold {fold_idx + 1}/{n_folds}...")

        # Split Data
        # We slice the original X/y to preserve their type (Pandas/Polars) for the calculator
        # Polars supports slicing with numpy arrays via __getitem__
        # Pandas supports slicing via iloc
        
        if hasattr(X, "iloc"):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
        else:
            # Polars or other
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            
        if hasattr(y, "iloc"):
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
        else:
            # Polars Series or numpy array
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

        # Fit
        model_artifact = calculator.fit(X_train_fold, y_train_fold, config)

        # Evaluate
        if problem_type == "classification":
            metrics = calculate_classification_metrics(
                model_artifact, X_val_fold, y_val_fold
            )
        else:
            metrics = calculate_regression_metrics(
                model_artifact, X_val_fold, y_val_fold
            )

        if log_callback:
            # Log a key metric for the fold
            key_metric = "accuracy" if problem_type == "classification" else "r2"
            score = metrics.get(key_metric, 0.0)
            log_callback(f"Fold {fold_idx + 1} completed. {key_metric}: {score:.4f}")

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "metrics": sanitize_metrics(metrics),
                # We could store predictions here if needed, but might be too heavy
            }
        )

    # 3. Aggregate
    fold_metrics = [cast(Dict[str, float], r["metrics"]) for r in fold_results]
    aggregated = _aggregate_metrics(fold_metrics)

    if log_callback:
        log_callback(f"Cross-Validation Completed. Aggregated Metrics: {aggregated}")

    return {
        "aggregated_metrics": aggregated,
        "folds": fold_results,
        "cv_config": {
            "n_folds": n_folds,
            "cv_type": cv_type,
            "shuffle": shuffle,
            "random_state": random_state,
        },
    }
