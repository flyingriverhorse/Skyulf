"""Row eligibility and an educational probability band, shared by Spark and REST."""

import pandas as pd


def build_model_rules():
    """Reuse feature filters but skip the training target filter on unlabeled requests."""
    return {
        "reuse_pre_split": True,
        "skip_target_steps": True,
        "eligibility": [
            {
                "name": "required_shape",
                "version": "1",
                "function": "scoring.required_shape",
                "params": {},
            }
        ],
        "outputs": [
            {
                "name": "probability_band",
                "version": "1",
                "function": "scoring.probability_band",
                "params": {},
                "columns": [{"name": "probability_band", "dtype": "string"}],
            }
        ],
    }


build_scoring = build_model_rules


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


def build_combined_rules():
    """This project contains one model, so there is no composition."""
    return []
