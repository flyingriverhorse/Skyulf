"""Split-slot payload types and the ``SplitDataset`` container passed between nodes.

A split slot is deliberately engine-neutral: it may hold a Skyulf wrapper, a
pandas frame, a raw polars frame, or an ``(X, y)`` tuple, so consumers must
dispatch on what they are given rather than assume one frame library.
"""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
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
    """Hold train/test slots and optional positional training-only sample weights.

    ``train_sample_weight`` follows the row order of ``train``. Test and
    validation metrics remain unweighted.
    ``evaluation_coverage`` carries held-out input and retained row counts
    through preprocessing, without storing row identifiers or observations.
    """

    train: SplitPayload
    test: SplitPayload
    validation: SplitPayload | None = None
    train_sample_weight: Any = None
    evaluation_coverage: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Load older saved split containers without inventing missing coverage evidence."""
        self.__dict__.update(state)
        self.__dict__.setdefault("evaluation_coverage", {})

    def copy(self) -> SplitDataset:
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
            train_sample_weight=copy_leaf(self.train_sample_weight),
            test=copy_data(self.test),
            validation=(copy_data(self.validation) if self.validation is not None else None),
            evaluation_coverage=_copy.deepcopy(self.evaluation_coverage),
        )
