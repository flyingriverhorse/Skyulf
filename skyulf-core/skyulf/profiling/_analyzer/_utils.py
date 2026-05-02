"""Shared utilities for the analyzer mixins."""

from typing import Any, List, Protocol, cast, runtime_checkable

import polars as pl


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Narrow ``LazyFrame.collect()`` back to ``DataFrame`` (sync path only)."""
    return cast(pl.DataFrame, lf.collect())


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
    columns: List[str]

    # Cross-mixin helper signatures (real implementations live in their mixins).
    def _get_semantic_type(self, series: pl.Series) -> str:
        ...

    def _analyze_numeric(self, col: str, row: dict) -> Any:
        ...

    def _analyze_categorical(self, col: str, row: dict, basic: dict) -> Any:
        ...

    def _analyze_text(self, col: str, advanced_stats: dict) -> Any:
        ...

    def _analyze_sentiment(self, text_series: pl.Series) -> Any:
        ...

    def _check_pii(self, col: str) -> Any:
        ...

    def _analyze_date(self, col: str, row: dict) -> Any:
        ...


# Optional dependency probes — kept here so each mixin imports a single flag
# instead of re-running the try/except dance.

try:
    from sklearn.cluster import KMeans  # noqa: F401
    from sklearn.decomposition import PCA  # noqa: F401
    from sklearn.ensemble import IsolationForest  # noqa: F401
    from sklearn.impute import SimpleImputer  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: F401
    from sklearn.tree import _tree  # noqa: F401  # ty: ignore[unresolved-import]

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import f_oneway, kstest, shapiro  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import adfuller  # noqa: F401

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
