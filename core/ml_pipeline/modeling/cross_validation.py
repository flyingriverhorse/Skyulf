"""Cross-validation logic for V2 modeling."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, ShuffleSplit

if TYPE_CHECKING:
    from .base import BaseModelCalculator, BaseModelApplier

from .evaluation.metrics import calculate_classification_metrics, calculate_regression_metrics
from .evaluation.common import _sanitize_structure

def _aggregate_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
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
                "max": float(np.max(values))
            }
            
    return aggregated

def perform_cross_validation(
    calculator: BaseModelCalculator,
    applier: BaseModelApplier,
    X: pd.DataFrame,
    y: pd.Series,
    config: Dict[str, Any],
    n_folds: int = 5,
    cv_type: str = "k_fold", # k_fold, stratified_k_fold, time_series_split, shuffle_split
    shuffle: bool = True,
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, int], None]] = None
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
        
    Returns:
        Dict containing aggregated metrics and per-fold details.
    """
    
    problem_type = calculator.problem_type
    
    # 1. Setup Splitter
    if cv_type == "time_series_split":
        splitter = TimeSeriesSplit(n_splits=n_folds)
    elif cv_type == "shuffle_split":
        splitter = ShuffleSplit(n_splits=n_folds, test_size=0.2, random_state=random_state)
    elif cv_type == "stratified_k_fold" and problem_type == "classification":
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
    else:
        # Default to KFold
        splitter = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state if shuffle else None)
        
    fold_results = []
    
    # Ensure numpy for splitting
    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else X
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else y
    
    # 2. Iterate Folds
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_arr, y_arr)):
        if progress_callback:
            progress_callback(fold_idx + 1, n_folds)

        # Split Data
        X_train_fold = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]
        
        # Fit
        model_artifact = calculator.fit(X_train_fold, y_train_fold, config)
        
        # Evaluate
        metrics = {}
        if problem_type == "classification":
            metrics = calculate_classification_metrics(model_artifact, X_val_fold, y_val_fold)
        elif problem_type == "regression":
            metrics = calculate_regression_metrics(model_artifact, X_val_fold, y_val_fold)
            
        fold_results.append({
            "fold": fold_idx + 1,
            "metrics": metrics
        })
        
    # 3. Aggregate
    metrics_list = [r["metrics"] for r in fold_results]
    aggregated = _aggregate_metrics(metrics_list)
    
    return {
        "aggregated": aggregated,
        "folds": fold_results
    }
