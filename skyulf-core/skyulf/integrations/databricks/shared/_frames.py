"""Package-internal memory and scalar contracts for pandas/Polars frames."""

import math
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


def frame_bytes(frame: pd.DataFrame | pl.DataFrame) -> int:
    """Count the actual frame allocation used for the memory guard."""
    if isinstance(frame, pd.DataFrame):
        return int(frame.memory_usage(index=True, deep=True).sum())
    return int(frame.estimated_size())


def output_scalar(value: Any) -> Any:
    """Convert bounded pandas/NumPy scalars without changing their logical type."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError("Prediction output contains a nonfinite number.")
    return value
