"""Optional custom callbacks selected by config/scoring.yml."""

import pandas as pd


def required_shape(frame, params):
    """Preserve missing-shape rows with an exclusion reason instead of losing them."""
    missing = frame[["mean_area", "mean_perimeter"]].isna().any(axis=1)
    return pd.Series(None, index=frame.index, dtype="string").mask(missing, "missing_shape")


def probability_band(frame, predictions, params):
    """Expose the model's positive-class probability as a simple demonstration label."""
    probability = predictions["probability_1"]
    band = pd.Series("middle", index=predictions.index, dtype="string")
    band = band.mask(probability < 0.25, "low").mask(probability >= 0.75, "high")
    return band.mask(probability.isna()).to_frame("probability_band")
