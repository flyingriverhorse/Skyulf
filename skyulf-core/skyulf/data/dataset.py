"""Split-slot payload types and the ``SplitDataset`` container passed between nodes.

A split slot is deliberately engine-neutral: it may hold a Skyulf wrapper, a
pandas frame, a raw polars frame, or an ``(X, y)`` tuple, so consumers must
dispatch on what they are given rather than assume one frame library.
"""

import copy as _copy
from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from skyulf.engines import SkyulfDataFrame

# Payload for a split slot: engine-neutral frame, pandas frame, raw Polars
# frame, or (X, y) tuple. Raw Polars frames appear when the backend runs with
# SKYULF_ENGINE=polars and wraps a plain frame for evaluation.
SplitPayload = (
    SkyulfDataFrame
    | pd.DataFrame
    | pl.DataFrame
    | tuple[SkyulfDataFrame | pd.DataFrame | pl.DataFrame, Any]
)


@dataclass
class SplitDataset:
    """The train/test (and optional validation) slots a splitter node produces."""

    train: SplitPayload
    test: SplitPayload
    validation: SplitPayload | None = None

    def copy(self) -> "SplitDataset":
        """Return a ``SplitDataset`` whose slots are copies, not shared references.

        Each leaf is copied through its own ``copy()``/``clone()`` so pandas and
        polars frames are both handled, and ``(X, y)`` tuples are copied
        element-wise. A leaf with neither method falls back to
        :func:`copy.copy`, which is shallow: the container is independent, but
        mutable contents inside such a leaf are still shared.
        """

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
