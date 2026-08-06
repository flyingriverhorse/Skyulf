"""Correlation-threshold feature selector."""

import inspect
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from ...core.meta.decorators import node_meta
from ...engines.polars_engine import SkyulfPolarsWrapper
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, resolve_columns
from .._artifacts import CorrelationThresholdArtifact
from .._helpers import to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine

_NATIVE_POLARS_METHODS = frozenset(("pearson", "spearman"))
NativePolarsCorrelationMethod = Literal["pearson", "spearman"]
_POLARS_CORRELATION_DTYPES = frozenset(
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


def _corr_drop_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Polars apply path: drop the precomputed ``columns_to_drop`` list."""
    if not params.get("drop_columns", True):
        return X, y
    to_drop = [c for c in params.get("columns_to_drop", []) if c in X.columns]
    if to_drop:
        X = X.drop(to_drop)
    return X, y


def _corr_drop_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Pandas apply path: drop the precomputed ``columns_to_drop`` list."""
    if not params.get("drop_columns", True):
        return X, y
    to_drop = [c for c in params.get("columns_to_drop", []) if c in X.columns]
    if to_drop:
        X = X.drop(columns=to_drop)
    return X, y


def _as_polars_frame(X: Any) -> pl.DataFrame | None:
    """Return raw Polars data for native correlation fitting when available."""
    if isinstance(X, pl.DataFrame):
        return X
    if isinstance(X, SkyulfPolarsWrapper):
        return X._df
    return None


def _polars_corr_accepts_method() -> bool:
    """Return whether the installed Polars correlation API accepts ``method``."""
    corr = getattr(pl, "corr", None)
    if not callable(corr):
        return False
    try:
        return "method" in inspect.signature(corr).parameters
    except (TypeError, ValueError):
        return False


def _fit_correlation_threshold_pandas(
    X: Any,
    config: dict[str, Any],
) -> CorrelationThresholdArtifact:
    """Run the retained Pandas-compatible correlation fit path."""
    X_pd = to_pandas(X)
    threshold = config.get("threshold", 0.95)
    drop_columns = config.get("drop_columns", True)
    # Prefer "correlation_method" — falling back to "method" can collide with the
    # facade's own "method" key (e.g. "correlation_threshold").
    method = config.get("correlation_method", "pearson")
    cols = resolve_columns(X_pd, config, detect_numeric_columns)
    if len(cols) < 2:
        return cast(CorrelationThresholdArtifact, {})

    corr_matrix = X_pd[cols].corr(method=method).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return cast(
        CorrelationThresholdArtifact,
        {
            "type": "correlation_threshold",
            "columns_to_drop": to_drop,
            "threshold": threshold,
            "method": method,
            "drop_columns": drop_columns,
        },
    )


def _native_polars_correlation_eligible(
    frame: pl.DataFrame,
    columns: list[str],
    method: Any,
    threshold: Any,
) -> bool:
    """Return whether this fit can preserve its contract on the native path."""
    return (
        isinstance(method, str)
        and method in _NATIVE_POLARS_METHODS
        and isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and _polars_corr_accepts_method()
        and all(frame.get_column(column).dtype in _POLARS_CORRELATION_DTYPES for column in columns)
    )


def _polars_correlation_columns_to_drop(
    frame: pl.DataFrame,
    columns: list[str],
    method: NativePolarsCorrelationMethod,
    threshold: int | float,
) -> list[str]:
    """Return upper-triangle columns whose pairwise-complete correlation exceeds threshold."""
    normalized = frame.select([pl.col(column).cast(pl.Float64).alias(column) for column in columns])
    expressions: list[pl.Expr] = []
    for right_index, right_column in enumerate(columns):
        for left_column in columns[:right_index]:
            left = pl.col(left_column)
            right = pl.col(right_column)
            complete = (
                left.is_not_null() & right.is_not_null() & left.is_not_nan() & right.is_not_nan()
            )
            expressions.append(
                pl.corr(
                    left.filter(complete),
                    right.filter(complete),
                    method=method,
                )
                .abs()
                .alias(f"__skyulf_correlation_{len(expressions)}")
            )

    values = normalized.select(expressions).row(0)
    to_drop: list[str] = []
    offset = 0
    for right_index, right_column in enumerate(columns):
        pair_values = values[offset : offset + right_index]
        offset += right_index
        if any(value is not None and value > threshold for value in pair_values):
            to_drop.append(right_column)
    return to_drop


class CorrelationThresholdApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, _corr_drop_polars, _corr_drop_pandas)


@NodeRegistry.register("CorrelationThreshold", CorrelationThresholdApplier)
@node_meta(
    id="CorrelationThreshold",
    name="Correlation Threshold",
    category="Feature Selection",
    description="Remove features highly correlated with others.",
    params={"threshold": 0.95, "method": "pearson"},
)
class CorrelationThresholdCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> CorrelationThresholdArtifact:  # pylint: disable=arguments-differ
        threshold = config.get("threshold", 0.95)
        drop_columns = config.get("drop_columns", True)
        # Prefer "correlation_method" — falling back to "method" can collide with the
        # facade's own "method" key (e.g. "correlation_threshold").
        method = config.get("correlation_method", "pearson")
        frame = _as_polars_frame(X)

        if frame is not None:
            columns = resolve_columns(SkyulfPolarsWrapper(frame), config, detect_numeric_columns)
            if len(columns) < 2:
                return cast(CorrelationThresholdArtifact, {})
            if _native_polars_correlation_eligible(frame, columns, method, threshold):
                native_method = cast(NativePolarsCorrelationMethod, method)
                return cast(
                    CorrelationThresholdArtifact,
                    {
                        "type": "correlation_threshold",
                        "columns_to_drop": _polars_correlation_columns_to_drop(
                            frame,
                            columns,
                            native_method,
                            threshold,
                        ),
                        "threshold": threshold,
                        "method": method,
                        "drop_columns": drop_columns,
                    },
                )

        # Retain this compatibility route until Polars supports Kendall and callable correlations.
        return _fit_correlation_threshold_pandas(X, config)
