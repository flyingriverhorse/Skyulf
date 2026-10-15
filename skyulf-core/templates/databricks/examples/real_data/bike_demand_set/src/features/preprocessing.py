"""Hourly clock features and training-fitted weather imputation."""

import pandas as pd

from skyulf.preprocessing import column_step, fitted_step

from .numeric import cycle


def clock_values(frame):
    """Keep midnight adjacent to 23:00 without using neighboring observations."""
    sine, cosine = cycle(frame["hr"])
    return pd.DataFrame({"hour_sin": sine, "hour_cos": cosine}, index=frame.index)


def clock_cycle():
    """Add two row-local periodic clock features."""
    return column_step("clock_cycle", clock_values, output=["hour_sin", "hour_cos"])


def learn_weather(frame, target, params):
    """Fit one median per weather measurement on training rows only."""
    return {column: float(frame[column].median()) for column in params["columns"]}


def apply_weather(frame, state, params):
    """Use the fitted medians in every worker regardless of batch size or order."""
    return frame.loc[:, params["columns"]].fillna(state).astype("float64")


def weather_imputation(columns):
    """Replace weather columns with consistently typed, filled measurements."""
    return fitted_step(
        "weather_imputation",
        learn_weather,
        apply_weather,
        output=columns,
        replace=True,
        params={"columns": columns},
    )
