"""Training data bundling utilities for hyperparameter tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ...shared import _prepare_training_data


@dataclass(frozen=True)
class TrainingDataBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: Optional[pd.DataFrame]
    y_validation: Optional[pd.Series]
    feature_columns: List[str]
    target_meta: Optional[Dict[str, Any]]


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


__all__ = ["TrainingDataBundle", "_build_training_data_bundle"]
