"""Train/Test/Validation Split preprocessing helpers.

This module provides functionality to split a dataset into training, testing,
and optionally validation sets. It uses sklearn's train_test_split with support
for stratification and tracks which split each row belongs to via metadata.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from core.feature_engineering.schemas import TrainTestSplitNodeSignal

logger = logging.getLogger(__name__)

# Split type metadata column name
SPLIT_TYPE_COLUMN = "__split_type__"


def apply_train_test_split(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, TrainTestSplitNodeSignal]:
    """Apply train/test/validation split to the dataset."""
    node_id = node.get("id") if isinstance(node, dict) else None
    signal = TrainTestSplitNodeSignal(node_id=str(node_id) if node_id is not None else None)

    if frame.empty:
        return frame, "Train/Test Split: no data available", signal

    data = node.get("data") or {}
    config = data.get("config") or {}

    test_size = config.get("test_size", 0.2)
    validation_size = config.get("validation_size", 0.0)
    random_state = config.get("random_state", 42)
    shuffle = config.get("shuffle", True)
    stratify_enabled = config.get("stratify", False)
    target_column = config.get("target_column", "")

    if test_size <= 0 or test_size >= 1:
        return frame, "Train/Test Split: test_size must be between 0 and 1", signal

    if validation_size < 0 or validation_size >= 1:
        return frame, "Train/Test Split: validation_size must be between 0 and 1", signal

    if test_size + validation_size >= 1:
        return frame, "Train/Test Split: test_size + validation_size must be less than 1", signal

    signal.test_ratio = float(test_size)
    signal.validation_ratio = float(validation_size)
    signal.stratified = stratify_enabled
    signal.target_column = str(target_column) if target_column else None
    signal.random_state = random_state if isinstance(random_state, int) else None
    signal.shuffle = shuffle
    signal.total_size = len(frame)

    stratify_col = None
    if stratify_enabled and target_column and target_column in frame.columns:
        stratify_col = frame[target_column]
        
        # Check for missing values in target which breaks stratification
        if stratify_col.isnull().any():
            logger.warning(
                "Cannot stratify with target column '%s': contains missing values",
                target_column,
            )
            stratify_col = None
        else:
            unique_values = stratify_col.nunique()
            if unique_values < 2:
                logger.warning(
                    "Cannot stratify with target column '%s': only %s unique values",
                    target_column,
                    unique_values,
                )
                stratify_col = None

    working_frame = frame.copy()

    try:
        if validation_size > 0:
            train_val, test = train_test_split(
                working_frame,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify_col if stratify_col is not None else None,
            )

            val_size_adjusted = validation_size / (1 - test_size)

            stratify_col_train_val = None
            if stratify_enabled and target_column and target_column in train_val.columns:
                stratify_col_train_val = train_val[target_column]

            train, validation = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify_col_train_val if stratify_col_train_val is not None else None,
            )

            train[SPLIT_TYPE_COLUMN] = "train"
            validation[SPLIT_TYPE_COLUMN] = "validation"
            test[SPLIT_TYPE_COLUMN] = "test"

            result_frame = pd.concat([train, validation, test], ignore_index=False)

            signal.train_size = len(train)
            signal.validation_size = len(validation)
            signal.test_size = len(test)
            signal.splits_created = ["train", "validation", "test"]

            summary = (
                "Train/Test/Validation Split: "
                f"Train={len(train)} ({len(train)/len(frame)*100:.1f}%), "
                f"Validation={len(validation)} ({len(validation)/len(frame)*100:.1f}%), "
                f"Test={len(test)} ({len(test)/len(frame)*100:.1f}%)"
            )

        else:
            train, test = train_test_split(
                working_frame,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify_col if stratify_col is not None else None,
            )

            train[SPLIT_TYPE_COLUMN] = "train"
            test[SPLIT_TYPE_COLUMN] = "test"

            result_frame = pd.concat([train, test], ignore_index=False)

            signal.train_size = len(train)
            signal.test_size = len(test)
            signal.splits_created = ["train", "test"]

            summary = (
                "Train/Test Split: "
                f"Train={len(train)} ({len(train)/len(frame)*100:.1f}%), "
                f"Test={len(test)} ({len(test)/len(frame)*100:.1f}%)"
            )

        return result_frame, summary, signal

    except Exception as exc:  # pragma: no cover - defensive guard for worker runtime
        logger.error("Train/Test Split failed: %s", exc)
        return frame, f"Train/Test Split: error - {exc}", signal


def get_split_type(frame: pd.DataFrame) -> Optional[str]:
    """Get the split type from a DataFrame if it has split metadata."""
    if SPLIT_TYPE_COLUMN in frame.columns:
        split_types = frame[SPLIT_TYPE_COLUMN].value_counts()
        if not split_types.empty:
            return str(split_types.index[0])
    return None


def remove_split_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove split metadata column from DataFrame."""
    if SPLIT_TYPE_COLUMN in frame.columns:
        return frame.drop(columns=[SPLIT_TYPE_COLUMN])
    return frame
