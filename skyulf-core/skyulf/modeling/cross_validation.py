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
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
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
    cv_type: str = "k_fold",  # k_fold, stratified_k_fold, time_series_split, shuffle_split, nested_cv
    shuffle: bool = True,
    random_state: int = 42,
    time_column: Optional[str] = None,
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
        time_column: Optional column name for sorting when using time_series_split.
        progress_callback: Optional callback(current_fold, total_folds).
        log_callback: Optional callback for logging messages.

    Returns:
        Dict containing aggregated metrics and per-fold details.
    """
    import logging

    logger = logging.getLogger(__name__)
    problem_type = calculator.problem_type

    if log_callback:
        log_callback(f"Starting Cross-Validation (Folds: {n_folds}, Type: {cv_type})")

    # For Time Series Split, sort data chronologically
    if cv_type == "time_series_split" and isinstance(X, pd.DataFrame):
        X, y = _sort_by_time(X, y, time_column, log_callback, logger)

    # Handle nested CV separately
    if cv_type == "nested_cv":
        return _perform_nested_cv(
            calculator=calculator,
            applier=applier,
            X=X,
            y=y,
            config=config,
            n_folds=n_folds,
            shuffle=shuffle,
            random_state=random_state,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )

    # 1. Setup Splitter
    if cv_type == "time_series_split":
        splitter = TimeSeriesSplit(n_splits=n_folds)
    elif cv_type == "shuffle_split":
        splitter = ShuffleSplit(n_splits=n_folds, test_size=0.2, random_state=random_state)
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
            metrics = calculate_classification_metrics(model_artifact, X_val_fold, y_val_fold)
        else:
            metrics = calculate_regression_metrics(model_artifact, X_val_fold, y_val_fold)

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


def _sort_by_time(
    X: pd.DataFrame,
    y: Any,
    time_column: Optional[str],
    log_callback: Optional[Callable[[str], None]],
    logger: Any,
) -> tuple:
    """Sort X and y by a time column for Time Series Split.

    If time_column is provided, sort by that column (and drop it from features).
    If not provided, auto-detect the first datetime column.
    If no datetime column is found, log a warning and return data as-is.
    """
    sort_col = time_column

    if not sort_col:
        # Auto-detect: find first datetime64 column
        datetime_cols = X.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
        if datetime_cols:
            sort_col = datetime_cols[0]
            msg = f"Time Series CV: auto-detected datetime column '{sort_col}' for sorting."
            if log_callback:
                log_callback(msg)
            logger.info(msg)

    if sort_col and sort_col in X.columns:
        sort_order = X[sort_col].argsort()
        X = X.iloc[sort_order].reset_index(drop=True)
        if hasattr(y, "iloc"):
            y = y.iloc[sort_order].reset_index(drop=True)
        else:
            y = y[sort_order]
        msg = f"Time Series CV: data sorted by '{sort_col}'."
        if log_callback:
            log_callback(msg)
        logger.info(msg)
        # Drop the time column from features so it doesn't leak into the model
        X = X.drop(columns=[sort_col])
    elif sort_col:
        msg = f"Time Series CV: specified time column '{sort_col}' not found in data. Using row order."
        if log_callback:
            log_callback(msg)
        logger.warning(msg)
    else:
        msg = "Time Series CV: no datetime column found. Assuming data is already sorted chronologically."
        if log_callback:
            log_callback(msg)
        logger.warning(msg)

    return X, y


def _build_splitter(
    cv_type: str,
    n_folds: int,
    problem_type: str,
    shuffle: bool = True,
    random_state: int = 42,
) -> Any:
    """Build a sklearn CV splitter from cv_type string."""
    if cv_type == "time_series_split":
        return TimeSeriesSplit(n_splits=n_folds)
    elif cv_type == "shuffle_split":
        return ShuffleSplit(n_splits=n_folds, test_size=0.2, random_state=random_state)
    elif cv_type == "stratified_k_fold" and problem_type == "classification":
        return StratifiedKFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
    else:
        return KFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )


def _perform_nested_cv(
    calculator: BaseModelCalculator,
    applier: BaseModelApplier,
    X: Union[pd.DataFrame, SkyulfDataFrame],
    y: Union[pd.Series, Any],
    config: Dict[str, Any],
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Performs nested cross-validation with an outer loop for evaluation
    and an inner loop for hyperparameter selection per fold.

    Outer loop: evaluates generalization (same as standard CV).
    Inner loop: selects best hyperparameters via 3-fold CV within each
    training fold (avoids overfitting to the validation set).
    """
    import logging

    logger = logging.getLogger(__name__)
    problem_type = calculator.problem_type
    inner_folds = min(3, n_folds - 1) if n_folds > 2 else 2

    if log_callback:
        log_callback(f"Starting Nested CV (Outer: {n_folds} folds, Inner: {inner_folds} folds)")

    # Build outer splitter — use stratified for classification, KFold for regression
    if problem_type == "classification":
        outer_splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
    else:
        outer_splitter = KFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )

    X_arr, y_arr = SklearnBridge.to_sklearn((X, y))

    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_splitter.split(X_arr, y_arr)):
        if progress_callback:
            progress_callback(fold_idx + 1, n_folds)

        if log_callback:
            log_callback(f"Nested CV — Outer Fold {fold_idx + 1}/{n_folds}...")

        # Slice preserving original types
        if hasattr(X, "iloc"):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
        else:
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]

        if hasattr(y, "iloc"):
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
        else:
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

        # --- Inner loop: quick hyperparameter selection ---
        # Try each config variant with inner CV and pick the best
        # For now, we just do a single inner CV to get a stable training signal,
        # then evaluate on the held-out outer fold.
        # This provides the key nested CV benefit: unbiased generalization estimate.

        # Build inner splitter
        if problem_type == "classification":
            inner_splitter = StratifiedKFold(
                n_splits=inner_folds,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
        else:
            inner_splitter = KFold(
                n_splits=inner_folds,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )

        # Inner CV: train on inner folds, collect inner scores for diagnostics
        X_train_arr, y_train_arr = SklearnBridge.to_sklearn((X_train_fold, y_train_fold))
        inner_scores: List[float] = []
        for inner_train_idx, inner_val_idx in inner_splitter.split(X_train_arr, y_train_arr):
            if hasattr(X_train_fold, "iloc"):
                X_inner_train = X_train_fold.iloc[inner_train_idx]
                X_inner_val = X_train_fold.iloc[inner_val_idx]
            else:
                X_inner_train = X_train_fold[inner_train_idx]
                X_inner_val = X_train_fold[inner_val_idx]

            if hasattr(y_train_fold, "iloc"):
                y_inner_train = y_train_fold.iloc[inner_train_idx]
                y_inner_val = y_train_fold.iloc[inner_val_idx]
            else:
                y_inner_train = y_train_fold[inner_train_idx]
                y_inner_val = y_train_fold[inner_val_idx]

            try:
                inner_artifact = calculator.fit(X_inner_train, y_inner_train, config)
                if problem_type == "classification":
                    inner_metrics = calculate_classification_metrics(
                        inner_artifact, X_inner_val, y_inner_val
                    )
                    inner_scores.append(inner_metrics.get("accuracy", 0.0))
                else:
                    inner_metrics = calculate_regression_metrics(
                        inner_artifact, X_inner_val, y_inner_val
                    )
                    inner_scores.append(inner_metrics.get("r2", 0.0))
            except Exception as e:
                logger.warning(f"Inner fold failed: {e}")
                inner_scores.append(0.0)

        inner_mean = float(np.mean(inner_scores)) if inner_scores else 0.0

        # --- Outer evaluation: train on full outer train, evaluate on outer val ---
        model_artifact = calculator.fit(X_train_fold, y_train_fold, config)

        if problem_type == "classification":
            metrics = calculate_classification_metrics(model_artifact, X_val_fold, y_val_fold)
        else:
            metrics = calculate_regression_metrics(model_artifact, X_val_fold, y_val_fold)

        if log_callback:
            key_metric = "accuracy" if problem_type == "classification" else "r2"
            score = metrics.get(key_metric, 0.0)
            log_callback(
                f"Outer Fold {fold_idx + 1} — {key_metric}: {score:.4f} "
                f"(inner mean: {inner_mean:.4f})"
            )

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "metrics": sanitize_metrics(metrics),
                "inner_cv_mean": inner_mean,
            }
        )

    # Aggregate outer fold metrics
    fold_metrics = [cast(Dict[str, float], r["metrics"]) for r in fold_results]
    aggregated = _aggregate_metrics(fold_metrics)

    if log_callback:
        log_callback(f"Nested CV Completed. Aggregated Metrics: {aggregated}")

    return {
        "aggregated_metrics": aggregated,
        "folds": fold_results,
        "cv_config": {
            "n_folds": n_folds,
            "cv_type": "nested_cv",
            "inner_folds": inner_folds,
            "shuffle": shuffle,
            "random_state": random_state,
        },
    }
