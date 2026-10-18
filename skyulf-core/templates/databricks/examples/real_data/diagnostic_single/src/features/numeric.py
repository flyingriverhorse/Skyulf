"""Pure numeric helpers loaded with the project's saved source on every worker."""

import numpy as np


def circularity_values(area, perimeter):
    """Return dimensionless shape circularity without consulting other rows."""
    return (4.0 * np.pi * area / perimeter.where(perimeter > 0)).astype("float64")
