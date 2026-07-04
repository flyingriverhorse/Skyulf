"""Correlation-threshold feature selector."""

from typing import Any, Dict, Tuple, cast

import numpy as np

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, resolve_columns
from .._artifacts import CorrelationThresholdArtifact
from .._helpers import to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine


def _corr_drop_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    """Polars apply path: drop the precomputed ``columns_to_drop`` list."""
    if not params.get("drop_columns", True):
        return X, y
    to_drop = [c for c in params.get("columns_to_drop", []) if c in X.columns]
    if to_drop:
        X = X.drop(to_drop)
    return X, y


def _corr_drop_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    """Pandas apply path: drop the precomputed ``columns_to_drop`` list."""
    if not params.get("drop_columns", True):
        return X, y
    to_drop = [c for c in params.get("columns_to_drop", []) if c in X.columns]
    if to_drop:
        X = X.drop(columns=to_drop)
    return X, y


class CorrelationThresholdApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
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
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> CorrelationThresholdArtifact:  # pylint: disable=arguments-differ
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
