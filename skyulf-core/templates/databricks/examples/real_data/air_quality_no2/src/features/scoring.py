"""Optional custom callbacks selected by config/scoring.yml."""

import pandas as pd


def plausible_humidity(frame, params):
    """Exclude hours whose relative humidity is outside the sensor's plausible range."""
    humidity = frame["rel_humidity"]
    outside = humidity.lt(params["minimum"]) | humidity.gt(params["maximum"])
    return pd.Series(None, index=frame.index, dtype="string").mask(outside, "humidity_out_of_range")
