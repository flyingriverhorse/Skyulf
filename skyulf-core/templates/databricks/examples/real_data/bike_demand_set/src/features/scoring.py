"""Shared component eligibility, component output and custom demand composition."""

import pandas as pd


def build_model_rules():
    """Apply the same raw-input policy to both demand models."""
    return {
        "reuse_pre_split": True,
        "skip_target_steps": True,
        "eligibility": [
            {"name": "known_hour", "version": "1", "function": "scoring.known_hour", "params": {}}
        ],
        "outputs": [
            {
                "name": "nonnegative_demand",
                "version": "1",
                "function": "scoring.nonnegative_demand",
                "params": {},
                "columns": [{"name": "nonnegative_demand", "dtype": "float64"}],
            }
        ],
    }


def known_hour(frame, params):
    """Explain missing-hour exclusions without dropping the source record."""
    return pd.Series(None, index=frame.index, dtype="string").mask(
        frame["hr"].isna(), "missing_hour"
    )


def nonnegative_demand(frame, predictions, params):
    """Expose a nonnegative demand estimate while retaining the raw model prediction."""
    return predictions["prediction"].clip(lower=0).astype("float64").to_frame("nonnegative_demand")


def build_combined_rules():
    """Require both component results before publishing total predicted demand."""
    return [
        {
            "name": "total_demand",
            "version": "1",
            "function": "scoring.total_demand",
            "params": {},
            "required_components": ["casual", "registered"],
            "columns": [{"name": "total_demand", "dtype": "float64"}],
        }
    ]


def total_demand(inputs, predictions, params):
    """Compose the two model outputs; the actual cnt target is never read."""
    total = (
        predictions["casual__nonnegative_demand"] + predictions["registered__nonnegative_demand"]
    )
    return total.astype("float64").to_frame("total_demand")
