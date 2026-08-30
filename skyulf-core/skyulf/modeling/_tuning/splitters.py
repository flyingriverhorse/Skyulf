"""CV splitter construction for hyperparameter tuning.

Leaf module (F-18 split of ``engine.py``): the ``_build_*_cv`` family.
Splitters only need the problem type from the wrapped calculator, so it is
taken as an explicit ``problem_type`` argument.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    PredefinedSplit,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
)

from .schemas import TuningConfig


def build_cv_splitter(
    X: Any,
    y: Any,
    config: TuningConfig,
    validation_data: tuple[Any, Any] | None,
    problem_type: str,
) -> tuple[Any, Any, Any]:
    """Builds the CV splitter (or ``PredefinedSplit``) plus the ``X``/``y`` to search over.

    When ``validation_data`` is provided, it is concatenated with ``X``/``y`` and a
    ``PredefinedSplit`` is used so the searcher trains on ``X`` and validates on it.
    Otherwise a CV splitter is chosen from ``config`` (holdout, nested CV inner folds,
    time series, shuffle, stratified, or plain K-fold).
    """
    if validation_data is not None:
        return build_predefined_split_cv(X, y, validation_data)

    return select_cv_by_type(config, problem_type), X, y


def build_predefined_split_cv(
    X: Any,
    y: Any,
    validation_data: tuple[Any, Any],
) -> tuple[Any, Any, Any]:
    """Concatenates train/val data and builds a ``PredefinedSplit`` over it.

    Numpy (frameless) variant — the frame-based per-fold-refit variant is
    ``build_predefined_split_cv_frames``.

    The search treats ``X`` (train) as always-in-training-set (-1) and the concatenated
    ``validation_data`` as the single test fold (0), so the searcher trains on ``X`` and
    validates on ``validation_data``.
    """
    X_val, y_val = validation_data

    # Concatenate Train and Val (Numpy arrays)
    X_for_search = np.concatenate([X, X_val], axis=0)
    y_for_search = np.concatenate([y, y_val], axis=0)

    # Create test_fold array: -1 for train, 0 for val
    # -1 means "never in test set" (so always in training set)
    # 0 means "in test set for fold 0"
    test_fold = np.concatenate([np.full(len(X), -1), np.full(len(X_val), 0)])

    cv = PredefinedSplit(test_fold)
    return cv, X_for_search, y_for_search


def build_predefined_split_cv_frames(
    preprocessing_frames: tuple[Any, Any],
    validation_frames: tuple[Any, Any],
) -> tuple[Any, Any, Any]:
    """Frame variant of ``build_predefined_split_cv`` for per-fold refit.

    Concatenates the pre-transform train and validation frames (positional
    index reset so the mask aligns) and builds a ``PredefinedSplit`` where
    train rows are always in training (-1) and validation rows form the
    single scoring fold (0): the preprocessing chain refits on train rows
    only and candidates score against untouched validation rows.
    """

    def _as_pandas(frame: Any) -> Any:
        return frame.to_pandas() if hasattr(frame, "to_pandas") else frame

    X_train, y_train = preprocessing_frames
    X_val, y_val = validation_frames

    X_for_search = pd.concat([_as_pandas(X_train), _as_pandas(X_val)], ignore_index=True)
    y_for_search = pd.concat([_as_pandas(y_train), _as_pandas(y_val)], ignore_index=True)

    test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])

    cv = PredefinedSplit(test_fold)
    return cv, X_for_search, y_for_search


def build_holdout_cv(config: TuningConfig) -> Any:
    """Builds the single-split (20% holdout) CV used when ``cv_enabled`` is False."""
    return ShuffleSplit(n_splits=1, test_size=0.2, random_state=config.cv_random_state)


def build_shuffle_split_cv(config: TuningConfig) -> Any:
    """Builds a repeated shuffle-split CV splitter for ``cv_type == "shuffle_split"``."""
    return ShuffleSplit(
        n_splits=config.cv_folds,
        test_size=0.2,
        random_state=config.cv_random_state,
    )


def build_stratified_kfold_cv(config: TuningConfig) -> Any:
    """Builds a StratifiedKFold splitter for ``cv_type == "stratified_k_fold"``."""
    return StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=config.cv_shuffle,
        random_state=config.cv_random_state if config.cv_shuffle else None,
    )


def build_kfold_cv(config: TuningConfig) -> Any:
    """Builds the default plain KFold splitter (also the regression fallback for stratified)."""
    return KFold(
        n_splits=config.cv_folds,
        shuffle=config.cv_shuffle,
        random_state=config.cv_random_state if config.cv_shuffle else None,
    )


def select_cv_by_type(config: TuningConfig, problem_type: str) -> Any:
    """Picks a CV splitter from ``config`` (holdout, nested CV inner folds, time series,
    shuffle, stratified, or plain K-fold), based on ``cv_enabled``/``cv_type``.
    """
    if not config.cv_enabled:
        # Single split validation (20% holdout)
        return build_holdout_cv(config)

    if config.cv_type == "nested_cv":
        # Nested CV during tuning: use fewer inner folds for
        # candidate scoring. The outer evaluation loop runs
        # post-tuning in engine.py (as stratified_k_fold).
        return build_nested_inner_cv(config, problem_type)

    if config.cv_type == "time_series_split":
        return TimeSeriesSplit(n_splits=config.cv_folds)

    if config.cv_type == "shuffle_split":
        return build_shuffle_split_cv(config)

    if config.cv_type == "stratified_k_fold" and problem_type == "classification":
        return build_stratified_kfold_cv(config)

    # Default to KFold (also fallback for stratified if regression)
    return build_kfold_cv(config)


def build_nested_inner_cv(config: TuningConfig, problem_type: str) -> Any:
    """Builds the inner-fold CV splitter used for candidate scoring during nested CV tuning."""
    inner_folds = min(3, config.cv_folds - 1) if config.cv_folds > 2 else 2
    inner_cv_random_state = config.cv_random_state if config.cv_shuffle else None
    if problem_type == "classification":
        return StratifiedKFold(
            n_splits=inner_folds,
            shuffle=config.cv_shuffle,
            random_state=inner_cv_random_state,
        )
    return KFold(
        n_splits=inner_folds,
        shuffle=config.cv_shuffle,
        random_state=inner_cv_random_state,
    )
