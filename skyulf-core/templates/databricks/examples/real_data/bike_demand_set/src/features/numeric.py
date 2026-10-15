"""Pure clock transformation shared by freshly loaded worker modules."""

import numpy as np


def cycle(hour):
    """Map hour-of-day onto a circle using only each row's hour."""
    angle = hour * (2.0 * np.pi / 24.0)
    return np.sin(angle), np.cos(angle)
