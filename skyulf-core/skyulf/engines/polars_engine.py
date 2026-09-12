"""Polars implementation of the ``SkyulfDataFrame`` protocol and its ``PolarsEngine`` adapter."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from .protocol import SkyulfDataFrame
from .registry import BaseEngine, EngineName, EngineRegistry

# Polars dtypes treated as numeric/bool for feature-matrix purposes (e.g.
# correlation and clustering evaluation), mirroring pandas'
# `select_dtypes(include=["number", "bool"])`. Shared so call sites stay
# consistent as new native-Polars paths are added.
POLARS_NUMERIC_BOOL_DTYPES: frozenset = frozenset(
    (
        pl.Boolean,
        pl.Float32,
        pl.Float64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )
)

# Polars numeric dtypes (float/int/uint), excluding bool — mirrors pandas'
# `select_dtypes(include=["number"])`. Shared so numeric-only column detection
# stays consistent as new native-Polars paths are added.
POLARS_NUMERIC_DTYPES: frozenset = POLARS_NUMERIC_BOOL_DTYPES - frozenset((pl.Boolean,))


class SkyulfPolarsWrapper:
    """Wrapper for Polars DataFrame to implement SkyulfDataFrame protocol."""

    def __init__(self, df: Any):
        """Store ``df``, the ``polars.DataFrame`` this wrapper adapts."""
        # df is pl.DataFrame
        self._df = df

    @property
    def columns(self) -> Sequence[str]:
        """Column names as a ``list[str]``."""
        return self._df.columns

    @property
    def shape(self) -> tuple[int, int]:
        """Row and column counts as a ``(rows, cols)`` tuple."""
        return self._df.shape

    def select(self, columns: list[str] | str) -> "SkyulfDataFrame":
        """Return a new wrapper with only the selected column(s)."""
        return SkyulfPolarsWrapper(self._df.select(columns))

    def drop(self, columns: list[str]) -> "SkyulfDataFrame":
        """Return a new wrapper without the given columns."""
        return SkyulfPolarsWrapper(self._df.drop(columns))

    def with_column(self, name: str, values: Any) -> "SkyulfDataFrame":
        """Return a new wrapper with ``name`` set to ``values``; scalars broadcast to all rows."""
        # Polars with_columns takes expressions or series. Passing a bare
        # scalar to pl.Series(name, values) creates a length-1 Series, which
        # then fails to broadcast against a taller frame (unlike pandas'
        # assign(), which broadcasts scalars automatically). Use pl.lit()
        # for scalars so polars broadcasts it across all rows, matching
        # pandas semantics.
        if isinstance(values, pl.Series | np.ndarray | list | tuple):
            expr = pl.Series(name, values)
        else:
            expr = pl.lit(values).alias(name)
        return SkyulfPolarsWrapper(self._df.with_columns(expr))

    def to_native(self) -> Any:
        """Return the underlying ``polars.DataFrame`` without conversion."""
        return self._df

    def to_pandas(self) -> Any:
        """Convert the frame to a ``pandas.DataFrame``."""
        return self._df.to_pandas()

    def to_arrow(self) -> Any:
        """Convert the frame to an Arrow table."""
        return self._df.to_arrow()

    def copy(self) -> "SkyulfDataFrame":
        """Return a new wrapper around a clone of the underlying frame."""
        return SkyulfPolarsWrapper(self._df.clone())

    def __getitem__(self, key):
        """Delegate column/row selection to the underlying ``polars.DataFrame``."""
        return self._df[key]

    def __setitem__(self, key, value):
        """Assign a scalar cell by ``(row, col)`` key; other keys raise ``NotImplementedError``."""
        # polars.DataFrame supports scalar cell assignment via a (row, col)
        # index tuple (e.g. ``df[0, "a"] = 99``), which mutates in place and
        # works fine. It does NOT support pandas-style whole-column
        # assignment (e.g. ``df["a"] = series``) — that raises a TypeError
        # from polars internals. Delegate the supported case, and raise a
        # clear, actionable error for the unsupported one instead of letting
        # a confusing polars TypeError bubble up.
        if isinstance(key, tuple):
            self._df[key] = value
            return
        raise NotImplementedError(
            "SkyulfPolarsWrapper does not support whole-column assignment "
            "(polars.DataFrame has no pandas-style `df[col] = values`). "
            "Use `with_column(name, values)` instead, which returns a new "
            "wrapper with the column set/replaced."
        )

    def __len__(self) -> int:
        """Return the row count of the underlying frame."""
        return self._df.height

    def __getattr__(self, name):
        """Delegate unknown attribute access to the underlying ``polars.DataFrame``."""
        # Restoration probes attributes before _df exists; bypass this fallback
        # so missing state raises AttributeError instead of recursing.
        return getattr(object.__getattribute__(self, "_df"), name)


class PolarsEngine(BaseEngine):
    """Engine adapter for polars-backed data; registered under ``"polars"``."""

    name = EngineName.POLARS

    @classmethod
    def is_compatible(cls, data: Any) -> bool:
        """Return ``True`` if ``data`` is a ``polars.DataFrame``."""
        return isinstance(data, pl.DataFrame)

    @classmethod
    def from_pandas(cls, df: Any) -> Any:
        """Convert a pandas DataFrame to a ``polars.DataFrame``."""
        return pl.from_pandas(df)

    @classmethod
    def to_numpy(cls, df: Any) -> Any:
        """Convert ``df`` to a NumPy array, mirroring pandas' ``(n, 0)`` shape for empty frames."""
        # `SkyulfPolarsWrapper.__getattr__` delegates to the wrapped
        # `pl.DataFrame`, so `df.to_numpy()` and `df._df.to_numpy()` are
        # identical -- no need for a separate `isinstance` branch.
        if hasattr(df, "to_numpy"):
            # polars' to_numpy() raises "need at least one array to
            # concatenate" on a 0-column frame; pandas yields (n, 0) float64,
            # so mirror that to keep engine parity for empty selections.
            if getattr(df, "width", None) == 0:
                return np.empty((df.height, 0), dtype=np.float64)
            return df.to_numpy()
        return np.array(df)

    @classmethod
    def wrap(cls, data: Any) -> "SkyulfDataFrame":
        """Wrap ``data`` in a ``SkyulfPolarsWrapper`` (idempotent for wrappers)."""
        if isinstance(data, SkyulfPolarsWrapper):
            return data
        return SkyulfPolarsWrapper(data)

    @classmethod
    def create_dataframe(cls, data: Any) -> Any:
        """Build a ``polars.DataFrame`` from ``data`` (dict, list, etc.)."""
        return pl.DataFrame(data)


# Register automatically
EngineRegistry.register("polars", PolarsEngine)
