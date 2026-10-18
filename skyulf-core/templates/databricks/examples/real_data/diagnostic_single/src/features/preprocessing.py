"""Custom shape features and training-only learned clipping for the diagnostic example."""

from skyulf.preprocessing import column_step, fitted_step

from .numeric import circularity_values


def shape_values(frame):
    """Call a second source file so fresh worker loading is exercised too."""
    return circularity_values(frame["mean_area"], frame["mean_perimeter"])


def circularity():
    """Add circularity using the measured area and perimeter of the same sample."""
    return column_step("circularity", shape_values, output="circularity")


def learn_bounds(frame, target, params):
    """Learn bounds only from training rows; batch statistics belong in learn."""
    return {
        column: [float(frame[column].quantile(0.01)), float(frame[column].quantile(0.99))]
        for column in params["columns"]
    }


def apply_bounds(frame, state, params):
    """Clip against saved bounds without recomputing them on each scoring batch."""
    return frame.loc[:, params["columns"]].assign(
        **{column: frame[column].clip(*state[column]) for column in params["columns"]}
    )


def clip_measurements(columns):
    """Replace measurements with values clipped to the fitted percentile interval."""
    return fitted_step(
        "clip_measurements",
        learn_bounds,
        apply_bounds,
        output=columns,
        replace=True,
        params={"columns": columns},
    )
