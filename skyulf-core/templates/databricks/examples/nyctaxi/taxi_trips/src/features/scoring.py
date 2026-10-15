"""Scoring rules for the trips model set (fare and long_trip).

SCORING_MODE="combined": each model first reuses its saved pre-split recipe,
skipping filters that read its target, then the custom eligibility rule below.
After both models predict, long_trip_fare combines their results.
"""

import pandas as pd

SCORING_MODE = "combined"  # Options: pre_split, custom, combined
SKIP_TARGET_PRE_SPLIT_STEPS = True


def build_model_rules():
    """Reuse the pre-split recipe, then apply the custom eligibility rule."""
    if SCORING_MODE != "combined":
        raise ValueError("This project's rules are written for SCORING_MODE='combined'.")
    return {
        "reuse_pre_split": True,
        "skip_target_steps": SKIP_TARGET_PRE_SPLIT_STEPS,
        "eligibility": build_eligibility_rules(),
        "outputs": [],
    }


build_scoring = build_model_rules


def build_eligibility_rules():
    """Both models need a recorded pickup zone."""
    return [
        {
            "name": "observed_zones",
            "version": "1",
            "function": "scoring.require_observed_values",
            "params": {"columns": ["pickup_zip"]},
        },
    ]


def build_combined_rules():
    """Expected fare of long trips, from both models' results."""
    return [
        {
            "name": "long_trip_fare",
            "version": "1",
            "function": "scoring.long_trip_fare",
            "params": {},
            "columns": [{"name": "expected_long_trip_fare", "dtype": "float64"}],
            "required_components": ["fare", "long_trip"],
        },
    ]


def require_observed_values(frame, params):
    """Explain which required fields are missing without dropping records."""
    missing = frame.loc[:, params["columns"]].isna()
    return pd.Series(
        [
            "missing:" + ",".join(missing.columns[row].tolist()) if row.any() else None
            for row in missing.to_numpy()
        ],
        index=frame.index,
        dtype="string",
    )


def long_trip_fare(inputs, predictions, params):
    """Weight the fare estimate by the long-trip probability from the second model."""
    expected = predictions["fare__prediction"] * predictions["long_trip__probability_1"]
    return expected.astype("float64").to_frame("expected_long_trip_fare")
