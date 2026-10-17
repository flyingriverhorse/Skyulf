"""Custom air-quality steps; each apply reads only its own row, learn may use training rows."""

from skyulf.preprocessing import column_step, fitted_step

from .sensors import day_part_labels, without_marker


def unmark_missing(frame, params):
    """Replace the -200 marker in the listed columns with missing values for the imputer."""
    return without_marker(frame.loc[:, params["columns"]])


def missing_marker(columns):
    """Turn the station's -200 marker into NaN in place, so later built-ins see real gaps."""
    return column_step(
        "missing_marker", unmark_missing, output=columns, replace=True, params={"columns": columns}
    )


def day_part_column(frame):
    """Derive a categorical part of day from the hour of this row."""
    return day_part_labels(frame["hour"])


def day_part():
    """Add a categorical day_part column for the one-hot encoder."""
    return column_step("day_part", day_part_column, output="day_part")


def learn_hourly_temperature(frame, target):
    """Learn the training mean temperature for each hour; batch statistics belong in learn."""
    means = frame.groupby("hour")["temperature"].mean()
    hourly = {str(int(hour)): float(value) for hour, value in means.items()}
    return {"overall": float(frame["temperature"].mean()), "hourly": hourly}


def apply_temperature_anomaly(frame, state):
    """Subtract the saved mean for this row's hour; an unseen hour uses the overall mean."""
    hour = frame["hour"].astype("int64").astype("string")
    expected = hour.map(state["hourly"]).astype("float64").fillna(state["overall"])
    return frame["temperature"] - expected


def temperature_anomaly():
    """Add how much warmer or colder this hour is than the training mean for that hour."""
    return fitted_step(
        "temperature_anomaly",
        learn_hourly_temperature,
        apply_temperature_anomaly,
        output="temperature_anomaly",
    )
