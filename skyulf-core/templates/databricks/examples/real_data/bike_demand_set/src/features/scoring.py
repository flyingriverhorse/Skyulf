"""Optional custom callbacks selected by config/scoring.yml."""

import pandas as pd


def known_hour(frame, params):
    """Explain missing-hour exclusions without dropping the source record."""
    return pd.Series(None, index=frame.index, dtype="string").mask(
        frame["hr"].isna(), "missing_hour"
    )


def nonnegative_demand(frame, predictions, params):
    """Expose a nonnegative demand estimate while retaining the raw model prediction."""
    return predictions["prediction"].clip(lower=0).astype("float64").to_frame("nonnegative_demand")


def total_demand(inputs, predictions, params):
    """Compose the two model outputs; the actual cnt target is never read."""
    total = (
        predictions["casual__nonnegative_demand"] + predictions["registered__nonnegative_demand"]
    )
    return total.astype("float64").to_frame("total_demand")
