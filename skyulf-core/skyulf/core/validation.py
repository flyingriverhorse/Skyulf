"""Shared validation of row counts at prediction boundaries."""

from typing import Any


def prediction_row_count(data: Any) -> int:
    """Count observations in frames, sparse/dense matrices, or prediction sequences."""
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return len(data)


def validate_prediction_rows(expected: int, actual: int, *, stage: str) -> None:
    """Reject row-count changes when predictions have no input-row provenance.

    Args:
        expected: Number of observations supplied to this stage.
        actual: Number of observations or predictions returned by the stage.
        stage: Operation name included in the diagnostic.

    Raises:
        ValueError: If the stage loses or adds observations.
    """
    if actual != expected:
        raise ValueError(
            f"{stage} changed row count from {expected} to {actual}. "
            "Prediction requires one result per input row. Use row-preserving "
            "preprocessing, or filter inputs explicitly before requesting predictions."
        )
