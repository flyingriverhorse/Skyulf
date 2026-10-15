"""Project-specific trip factories selected from config YAML as trip_rules.<factory>."""

from skyulf.preprocessing import filter_step, fitted_step


def learn_bounds(df, y, params):
    """Learn clip limits from the training fold only."""
    values = df[params["column"]]
    return {"low": float(values.quantile(0.01)), "high": float(values.quantile(0.99))}


def clip_values(df, state, params):
    """Clip every later batch with the saved training limits."""
    return df[params["column"]].clip(state["low"], state["high"])


def clip_outliers(column):
    """Replace column with its value clipped to training 1%-99% quantiles."""
    params = {"column": column}
    return fitted_step(
        f"clip_{column}", learn_bounds, clip_values, output=column, replace=True, params=params
    )


def zips_known(df, params):
    """Keep trips whose pickup and dropoff zones are both recorded."""
    return df[params["columns"]].notna().all(axis=1)


def known_zips(columns):
    """Fixed eligibility filter: both zone columns must be present."""
    return filter_step(
        "known_zips", zips_known, columns=list(columns), params={"columns": list(columns)}
    )
