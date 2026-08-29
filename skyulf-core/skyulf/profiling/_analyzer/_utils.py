"""Shared utilities for the analyzer mixins."""

import importlib.util
from typing import Any, Protocol, cast, runtime_checkable

import polars as pl


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Narrow ``LazyFrame.collect()`` back to ``DataFrame`` (sync path only)."""
    return cast(pl.DataFrame, lf.collect())


_INT_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)


def _dtype_to_semantic_bucket(dtype: Any, ratio: float, n_unique: int) -> str:
    """Map a Polars dtype + cardinality ratio/count to a semantic bucket.

    Shared by ``ColumnMixin._get_semantic_type`` (per-series, used during the
    main per-column analysis pass) and ``EDAAnalyzer._semantic_type_for_column``
    (dtype-only, used by the inline/vectorized inference pass that reuses
    already-computed ``n_unique`` counts) so the Numeric/Categorical/Boolean/
    DateTime/Text buckets never drift apart between the two call sites.

    ``ratio`` is ``n_unique / row_count`` (0 when there are no rows).
    Low-cardinality ints (``ratio < 0.05`` and ``n_unique < 20``) and
    low-cardinality strings (``ratio < 0.05``) are treated as Categorical.
    """
    if dtype in (pl.Float32, pl.Float64):
        return "Numeric"
    if dtype in _INT_DTYPES:
        return "Categorical" if (ratio < 0.05 and n_unique < 20) else "Numeric"
    if dtype == pl.Boolean:
        return "Boolean"
    if dtype in (pl.Date, pl.Datetime, pl.Duration):
        return "DateTime"
    if dtype in (pl.Utf8, pl.String):
        return "Categorical" if ratio < 0.05 else "Text"
    if str(dtype) == "Categorical":
        return "Categorical"
    return "Text"


@runtime_checkable
class _AnalyzerState(Protocol):
    """Structural type for the shared :class:`EDAAnalyzer` state.

    Mixins inherit this Protocol so type checkers know ``self.df``,
    ``self.lazy_df`` and the cross-mixin helpers exist. Real implementations
    live in :class:`EDAAnalyzer` and the concrete mixins.
    """

    df: pl.DataFrame
    lazy_df: pl.LazyFrame
    row_count: int
    columns: list[str]

    # Cross-mixin helper signatures (real implementations live in their mixins).
    def _get_semantic_type(self, series: pl.Series) -> str: ...

    def _analyze_numeric(self, col: str, row: dict) -> Any: ...

    def _analyze_categorical(self, col: str, row: dict, basic: dict) -> Any: ...

    def _analyze_text(self, col: str, advanced_stats: dict) -> Any: ...

    def _analyze_sentiment(self, text_series: pl.Series) -> Any: ...

    def _check_pii(self, col: str) -> Any: ...

    def _analyze_date(self, col: str, row: dict) -> Any: ...


# Optional dependency probes — kept here so each mixin imports a single flag
# instead of re-running the try/except dance. `find_spec` answers the
# "is it installed" question without importing (and initialising) the package.

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
STATSMODELS_AVAILABLE = importlib.util.find_spec("statsmodels") is not None
VADER_AVAILABLE = importlib.util.find_spec("vaderSentiment") is not None
