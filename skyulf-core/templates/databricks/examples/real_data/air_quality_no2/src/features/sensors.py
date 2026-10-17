"""Pure helpers shared by the custom steps; loaded with the saved source on every worker."""

import numpy as np
import pandas as pd

# The UCI Air Quality station records an hour without a reading as -200.
MISSING_READING = -200.0


def without_marker(values):
    """Turn the station's missing-reading marker into a real missing value, cell by cell."""
    numbers = values.astype("float64")
    return numbers.mask(numbers.eq(MISSING_READING))


def day_part_labels(hours):
    """Name the part of day for each hour without consulting other rows."""
    return pd.Series(
        np.select(
            [hours < 6, hours < 12, hours < 18], ["night", "morning", "afternoon"], "evening"
        ),
        index=hours.index,
        dtype="object",
    )
