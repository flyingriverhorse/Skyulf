"""Implement the functions selected by features/scoring.py.

This file contains executable rules; scoring.py chooses rules and their params.
All callbacks receive pandas frames, even for a model fitted with Polars.

BEFORE the model: eligibility(frame, params) returns one reason per input row.
None means accepted; a nonempty string explains exclusion. Never drop rows here.
AFTER the model: output(frame, predictions, params) returns declared new columns,
with the same row count and index. It runs only for accepted rows.

The function path in scoring.py links directly to a function below, e.g.
custom.scoring_custom.require_value_range -> require_value_range(frame, params).
No separate call or registration is needed. Names/versions/params are saved with
that model version; changing the current source does not alter an older model.
"""

import numpy as np
import pandas as pd

# BEFORE PREDICTION: input eligibility checks.


def require_observed_values(frame, params):
    """Explain which required observed fields are missing without dropping records."""
    columns = params["columns"]
    missing = frame.loc[:, columns].isna()
    return pd.Series(
        [
            "missing:" + ",".join(missing.columns[row].tolist()) if row.any() else None
            for row in missing.to_numpy()
        ],
        index=frame.index,
        dtype="string",
    )


def require_value_range(frame, params):
    """Accept finite numeric values within inclusive limits; numeric strings are allowed."""
    column = params["column"]
    lower, upper = params.get("minimum"), params.get("maximum")
    _validate_range(lower, upper)
    values = pd.to_numeric(frame[column], errors="coerce")
    accepted = values.notna() & np.isfinite(values)
    if lower is not None:
        accepted &= values >= lower
    if upper is not None:
        accepted &= values <= upper
    return pd.Series(None, index=frame.index, dtype="string").mask(
        ~accepted.fillna(False), f"outside_range:{column}"
    )


def _validate_range(lower, upper):
    """Reject missing, nonfinite or contradictory bounds before evaluating rows."""
    bounds = [value for value in (lower, upper) if value is not None]
    invalid = any(type(value) not in (int, float) or not np.isfinite(value) for value in bounds)
    if not bounds or invalid or (lower is not None and upper is not None and lower > upper):
        raise ValueError("Range checks require finite ordered limits and at least one bound.")


# AFTER PREDICTION: additional business output columns.


def prediction_band(frame, predictions, params):
    """Map an estimate or class probability to declared ordered business bands."""
    thresholds = np.asarray(params["thresholds"], dtype=float)
    labels = params["labels"]
    if (
        thresholds.ndim != 1
        or not np.isfinite(thresholds).all()
        or (np.diff(thresholds) <= 0).any()
        or len(labels) != len(thresholds) + 1
    ):
        raise ValueError("Bands require increasing finite thresholds and one extra label.")
    values = predictions[params.get("column", "prediction")]
    positions = np.searchsorted(thresholds, values.to_numpy(dtype=float), side="right")
    bands = pd.Series(
        [labels[position] for position in positions], index=frame.index, dtype="string"
    )
    return pd.DataFrame({params.get("output", "band"): bands.mask(values.isna())})
