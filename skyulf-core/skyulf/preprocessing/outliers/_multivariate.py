"""Shared numeric matrix and positional mask operations for joint outlier detectors."""

from typing import Any

import numpy as np
import polars as pl

from ...utils import detect_numeric_columns
from .._fitted_validation import _columns, local_state_fields
from .._helpers import resolve_columns_then_to_numpy, select_rows_by_position
from ._common import validate_detector_warnings


def detector_matrix(X: Any, config: dict) -> tuple[np.ndarray, list[str]]:
    """Select the fixed axes before converting just their values to float64."""
    values, columns = resolve_columns_then_to_numpy(X, config, detect_numeric_columns)
    _columns(columns)
    return np.asarray(values, dtype=float), columns


def fit_detector(X: Any, y: Any, config: dict, detector: Any, kind: str) -> dict:
    """Fit jointly on complete finite rows, leaving unusable rows to downstream steps."""
    target = config.get("target_column") or getattr(y, "name", None)
    values, columns = detector_matrix(X, {**config, "target_column": target})
    if not columns:
        return {}
    finite = np.isfinite(values).all(axis=1)
    if finite.sum() < 2:
        raise ValueError(
            "Multivariate outliers require at least two complete finite training rows."
        )
    detector.fit(values[finite])
    warnings = (
        [] if finite.all() else [f"Kept {int((~finite).sum())} nonfinite rows without scoring."]
    )
    return {"type": kind, "columns": columns, "detector_object": detector, "warnings": warnings}


def validate_detector(raw: dict, kind: str, expected: type) -> Any:
    """Check exact saved ownership and axes without fitting or predicting."""
    if not local_state_fields(
        raw, kind, {"type", "columns", "detector_object", "warnings"}, allow_empty=True
    ):
        return None
    columns = _columns(raw["columns"])
    model = raw["detector_object"]
    if (
        not columns
        or type(model) is not expected
        or getattr(model, "n_features_in_", None) != len(columns)
    ):
        raise ValueError("Fitted detector type or feature width disagrees with selected columns.")
    if any(callable(value) for value in vars(model).values()):
        raise ValueError("Overridden detector methods are unsupported.")
    offset = getattr(model, "offset_", None)
    if (
        isinstance(offset, (bool, np.bool_))
        or not isinstance(offset, (int, float, np.number))
        or not np.isfinite(offset)
    ):
        raise ValueError("Fitted detector offset must be finite and numeric.")
    validate_detector_warnings(raw["warnings"])
    return model


def fitted_array(value: Any, shape: tuple[int, ...], *, positive: bool = False) -> None:
    """Require finite numeric fitted arrays with their original feature/sample axes."""
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.kind not in "fiu":
        raise ValueError("Fitted detector arrays have invalid shape or dtype.")
    if not np.isfinite(value).all() or (positive and np.any(value <= 0)):
        raise ValueError("Fitted detector arrays contain invalid numeric values.")


def detector_keep(X: Any, state: dict, *, training: bool = False) -> np.ndarray:
    """Keep nonfinite rows and score each finite row against the learned population only."""
    missing = set(state["columns"]) - set(X.columns)
    if missing:
        raise ValueError(f"Fitted detector requires columns: {sorted(missing)}.")
    values, _ = detector_matrix(X, {"columns": state["columns"]})
    finite = np.isfinite(values).all(axis=1)
    keep = np.ones(len(X), dtype=bool)
    if finite.any():
        model = state["detector_object"]
        keep[finite] = (
            model.negative_outlier_factor_ >= model.offset_
            if training
            else model.predict(values[finite]) == 1
        )
    return keep


def filter_detector_rows(X: Any, y: Any, state: dict, *, training: bool = False) -> tuple[Any, Any]:
    """Slice native features and every supported target shape using one positional mask."""
    if not state:
        return X, y
    keep = detector_keep(X, state, training=training)
    output = (
        X.filter(pl.Series(keep)) if isinstance(X, pl.DataFrame) else X.iloc[np.flatnonzero(keep)]
    )
    return output, select_rows_by_position(y, np.flatnonzero(keep))


def reject_boolean_options(config: dict, names: tuple[str, ...]) -> None:
    """Reject booleans where sklearn accepts them as integer-shaped configuration."""
    if any(isinstance(config.get(name), (bool, np.bool_)) for name in names):
        raise ValueError("Multivariate detector numeric options must not be booleans.")
