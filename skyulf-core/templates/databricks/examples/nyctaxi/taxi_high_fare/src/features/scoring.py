"""Scoring rules for the high-fare classifier.

SCORING_MODE="custom": rows outside a plausible trip distance stay in the result
table with an empty prediction and a reason; accepted rows get a fare_band from
the high-fare probability.
"""

import numpy as np
import pandas as pd

SCORING_MODE = "custom"  # Options: pre_split, custom, combined


def build_model_rules():
    """Use the custom eligibility and output rules below."""
    if SCORING_MODE != "custom":
        raise ValueError("This project's rules are written for SCORING_MODE='custom'.")
    return {"eligibility": build_eligibility_rules(), "outputs": build_output_rules()}


build_scoring = build_model_rules


def build_eligibility_rules():
    """Exclude trips whose distance is not a plausible taxi ride."""
    return [
        {
            "name": "plausible_distance",
            "version": "1",
            "function": "scoring.require_value_range",
            "params": {"column": "trip_distance", "minimum": 0.01, "maximum": 100.0},
        },
    ]


def build_output_rules():
    """Publish a low/medium/high band of the high-fare probability."""
    return [
        {
            "name": "fare_risk_band",
            "version": "1",
            "function": "scoring.prediction_band",
            "params": {
                "column": "probability_1",
                "thresholds": [0.3, 0.7],
                "labels": ["low", "medium", "high"],
                "output": "fare_band",
            },
            "columns": [{"name": "fare_band", "dtype": "string"}],
        },
    ]


def build_combined_rules():
    """A competition scores one champion; there are no cross-model results."""
    return []


def require_value_range(frame, params):
    """Accept finite numeric values within inclusive limits; numeric strings are allowed."""
    column = params["column"]
    lower, upper = params.get("minimum"), params.get("maximum")
    values = pd.to_numeric(frame[column], errors="coerce")
    accepted = values.notna() & np.isfinite(values) & (values >= lower) & (values <= upper)
    return pd.Series(None, index=frame.index, dtype="string").mask(
        ~accepted.fillna(False), f"outside_range:{column}"
    )


def prediction_band(frame, predictions, params):
    """Map the class probability to the declared ordered bands."""
    thresholds = np.asarray(params["thresholds"], dtype=float)
    values = predictions[params["column"]]
    positions = np.searchsorted(thresholds, values.to_numpy(dtype=float), side="right")
    bands = pd.Series(
        [params["labels"][position] for position in positions], index=frame.index, dtype="string"
    )
    return pd.DataFrame({params["output"]: bands.mask(values.isna())})
