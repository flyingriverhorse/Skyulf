"""Shared utilities for the analyzer mixins."""

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
# instead of re-running the try/except dance.

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        _tree,  # noqa: F401  # ty: ignore[unresolved-import]
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import f_oneway, kstest, shapiro

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import adfuller

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import (  # noqa: F401  # ty: ignore[unresolved-import]
        SentimentIntensityAnalyzer,
    )

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
