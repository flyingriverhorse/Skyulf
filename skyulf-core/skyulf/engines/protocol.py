from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class SkyulfDataFrame(Protocol):
    """
    The Universal DataFrame Interface for Skyulf.

    This protocol defines the minimum set of operations that any compute engine
    (Pandas, Polars, Spark, Dask) must support to be used within Skyulf nodes.
    """

    @property
    def columns(self) -> Any:
        """Column names (pandas `Index` or `List[str]` depending on engine)."""
        ...

    @property
    def dtypes(self) -> Any:
        """Column dtypes (pandas `Series` or polars `list` depending on engine)."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the dataframe (rows, cols)."""
        ...

    # Item access (column or row selection)
    def __getitem__(self, key: Any) -> Any:
        """Column or row selection. Returns a Series, scalar, or DataFrame."""
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        """Column or row assignment."""
        ...

    def __len__(self) -> int:
        """Return the number of rows."""
        ...

    # Engine-agnostic column ops (implemented by every engine adapter).
    def select(self, columns: list[str]) -> "SkyulfDataFrame":
        """Return a new frame with only the given columns."""
        ...

    def drop(self, columns: list[str]) -> "SkyulfDataFrame":
        """Return a new frame without the given columns."""
        ...

    def with_column(self, name: str, values: Any) -> "SkyulfDataFrame":
        """Return a new frame with a column added or replaced."""
        ...

    # Bridges
    def to_native(self) -> Any:
        """
        Return the underlying engine-native frame (a ``pandas.DataFrame`` or
        ``polars.DataFrame``) without any conversion.

        This is the documented way to escape the wrapper when native-engine
        APIs are required (e.g. ``pl.concat``, ``write_parquet``). It replaces
        reaching into the private ``._df`` attribute. Unlike ``to_pandas()``,
        which always yields a pandas frame (a no-op for pandas-backed data but
        a conversion/copy for polars-backed data), ``to_native()`` hands back
        whatever engine backs the wrapper, as-is.
        """
        ...

    def to_pandas(self) -> pd.DataFrame:
        """Convert to a Pandas DataFrame."""
        ...

    def to_arrow(self) -> Any:
        """
        Convert to an Arrow Table/RecordBatch.
        Critical for zero-copy data transfer between engines.
        """
        ...

    def copy(self) -> "SkyulfDataFrame":
        """Return a copy of the dataframe."""
        ...


@runtime_checkable
class PandasBackedFrame(SkyulfDataFrame, Protocol):
    """
    A :class:`SkyulfDataFrame` backed by pandas, exposing pandas-only
    attributes (``.loc``, ``.iloc``, ``.select_dtypes``) that are not part of
    the engine-agnostic base protocol.

    Use this in signatures when a function genuinely needs pandas semantics;
    the type checker then verifies the attribute exists instead of silently
    accepting ``Any``.
    """

    @property
    def loc(self) -> Any:
        """Label-based indexer (pandas ``.loc``)."""
        ...

    @property
    def iloc(self) -> Any:
        """Positional indexer (pandas ``.iloc``)."""
        ...

    def select_dtypes(self, include: Any, exclude: Any = None) -> "PandasBackedFrame":
        """Filter columns by dtype (pandas ``.select_dtypes``)."""
        ...


@runtime_checkable
class PolarsBackedFrame(SkyulfDataFrame, Protocol):
    """
    A :class:`SkyulfDataFrame` backed by polars, exposing polars-only
    attributes (``.with_columns``, ``.filter``, ``.to_polars``) that are not
    part of the engine-agnostic base protocol.

    Use this in signatures when a function genuinely needs polars semantics;
    the type checker then verifies the attribute exists instead of silently
    accepting ``Any``.
    """

    def with_columns(self, *expressions: Any) -> "PolarsBackedFrame":
        """Add/replace columns with expressions (polars ``.with_columns``)."""
        ...

    def filter(self, predicate: Any) -> "PolarsBackedFrame":
        """Filter rows by predicate (polars ``.filter``)."""
        ...

    def to_polars(self) -> Any:
        """Return the underlying ``polars.DataFrame``."""
        ...
