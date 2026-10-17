"""Combined scoring: reuse the pre-split outage filter, then custom eligibility and an NO2 band."""

import pandas as pd

# The measured_no2 pre-split step reads the target, which requests do not carry.
SKIP_TARGET_PRE_SPLIT_STEPS = True


def build_model_rules():
    """Reuse pre-split filters first, then check humidity and publish an NO2 band."""
    return {
        "reuse_pre_split": True,
        "skip_target_steps": SKIP_TARGET_PRE_SPLIT_STEPS,
        "eligibility": [
            {
                "name": "plausible_humidity",
                "version": "1",
                "function": "scoring.plausible_humidity",
                "params": {"minimum": 10.0, "maximum": 100.0},
            }
        ],
        "outputs": [
            {
                "name": "no2_band",
                "version": "1",
                "function": "scoring.no2_band",
                "params": {"moderate": 40.0, "high": 200.0},
                "columns": [{"name": "no2_band", "dtype": "string"}],
            }
        ],
    }


def plausible_humidity(frame, params):
    """Exclude hours whose relative humidity is outside the sensor's plausible range."""
    humidity = frame["rel_humidity"]
    outside = humidity.lt(params["minimum"]) | humidity.gt(params["maximum"])
    return pd.Series(None, index=frame.index, dtype="string").mask(outside, "humidity_out_of_range")


def no2_band(frame, predictions, params):
    """Label each prediction: below 40, up to the EU 200 ug/m3 hourly limit, or above it."""
    value = predictions["prediction"]
    band = pd.Series("moderate", index=predictions.index, dtype="string")
    band = band.mask(value < params["moderate"], "low").mask(value >= params["high"], "high")
    return band.mask(value.isna()).to_frame("no2_band")


def build_combined_rules():
    """This project contains one model, so there is no composition."""
    return []
