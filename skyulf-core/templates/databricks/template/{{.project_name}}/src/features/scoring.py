"""Optional custom callbacks selected by config/scoring.yml.

Built-in required_values, value_range and prediction_band rules need no Python.
These helpers show the equivalent callback signatures for project-specific rules.
Eligibility returns one nullable reason per row; outputs return declared columns.
All callbacks receive pandas frames, preserve rows/index and must be row-local
for Spark workers and endpoints. The model saves their source and parameters;
editing this file affects new model versions only. Keep policy ownership in YAML.
"""

import numpy as np
import pandas as pd


def profit(inputs, predictions, params):
    """Example callback: subtract cost from revenue, selected explicitly in config/scoring.yml."""
    return (predictions["revenue__prediction"] - predictions["cost__prediction"]).to_frame("profit")


# Custom eligibility helpers selected by custom: scoring.<function>.


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


# Custom output helpers selected by custom: scoring.<function>.


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
