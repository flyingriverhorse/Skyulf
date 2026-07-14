import copy as _copy
from dataclasses import dataclass
from typing import Any

import pandas as pd

from skyulf.engines import SkyulfDataFrame

# Payload for a split slot: engine-neutral frame, pandas frame, or (X, y) tuple.
SplitPayload = SkyulfDataFrame | pd.DataFrame | tuple[SkyulfDataFrame | pd.DataFrame, Any]


@dataclass
class SplitDataset:
    train: SplitPayload
    test: SplitPayload
    validation: SplitPayload | None = None

    def copy(self) -> "SplitDataset":
        def copy_leaf(value):
            # Fall back to a real copy (not the same reference) for
            # generic objects with neither `.copy()` nor `.clone()` (e.g.
            # a plain list/dict), so `SplitDataset.copy()` always returns
            # an independent object instead of silently aliasing.
            if hasattr(value, "copy"):
                return value.copy()
            if hasattr(value, "clone"):
                return value.clone()
            return _copy.copy(value)

        def copy_data(data):
            if isinstance(data, tuple):
                # Handle target copy safely (Series/Array/List)
                y = data[1]
                y_copy = copy_leaf(y)

                X = data[0]
                X_copy = copy_leaf(X)

                return (X_copy, y_copy)

            return copy_leaf(data)

        return SplitDataset(
            train=copy_data(self.train),
            test=copy_data(self.test),
            validation=(copy_data(self.validation) if self.validation is not None else None),
        )
