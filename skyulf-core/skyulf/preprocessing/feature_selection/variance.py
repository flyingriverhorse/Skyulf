"""Variance-threshold feature selector."""

from typing import Any, cast

import numpy as np
from sklearn.feature_selection import VarianceThreshold

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns
from .._artifacts import VarianceThresholdArtifact
from .._helpers import resolve_columns_then_to_numpy
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _drop_selected_pandas, _drop_selected_polars


class VarianceThresholdApplier(BaseApplier):
    """Remove the columns a fitted variance threshold rejected."""

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Drop ``candidate_columns - selected_columns`` from ``X``.

        Honours the artifact's ``drop_columns`` flag; when it is false the frame
        passes through untouched.
        """
        return apply_dual_engine(
            X, params, {"polars": _drop_selected_polars, "pandas": _drop_selected_pandas}
        )


@NodeRegistry.register("VarianceThreshold", VarianceThresholdApplier)
@node_meta(
    id="VarianceThreshold",
    name="Variance Threshold",
    category="Feature Selection",
    description="Remove features with low variance.",
    params={"threshold": 0.0},
    learns_from_data=True,
)
class VarianceThresholdCalculator(BaseCalculator):
    """Fit ``sklearn.feature_selection.VarianceThreshold`` on numeric columns."""

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> VarianceThresholdArtifact:  # pylint: disable=arguments-differ
        """Record which numeric columns clear ``config["threshold"]`` variance.

        Binary and constant columns are kept as candidates rather than excluded
        up front, so the threshold itself decides their fate. Returns an empty
        artifact — a no-op passthrough — when no numeric column is resolved.
        If no candidate clears the threshold, records an empty selection so
        apply can remove all candidates while preserving other columns.
        """
        threshold = config.get("threshold", 0.0)
        drop_columns = config.get("drop_columns", True)

        X_np, cols = resolve_columns_then_to_numpy(
            X,
            config,
            lambda d: detect_numeric_columns(d, exclude_binary=False, exclude_constant=False),
        )
        if not cols:
            return cast(VarianceThresholdArtifact, {})

        selector = VarianceThreshold(threshold=threshold)
        try:
            selector.fit(X_np)
        except ValueError as exc:
            # sklearn has computed variances but refuses an empty selection.
            if not str(exc).startswith("No feature in X meets the variance threshold"):
                raise
            all_missing = np.isnan(np.asarray(X_np, dtype=np.float64)).all(axis=0)
            if np.any(~np.isfinite(selector.variances_) & ~all_missing):
                # Only all-missing columns may have an undefined variance.
                raise
            selected_cols = []
        else:
            support = selector.get_support()
            selected_cols = [c for c, s in zip(cols, support, strict=True) if s]
        variances = (
            dict(zip(cols, selector.variances_.tolist(), strict=True))
            if hasattr(selector, "variances_")
            else {}
        )
        return cast(
            VarianceThresholdArtifact,
            {
                "type": "variance_threshold",
                "selected_columns": selected_cols,
                "candidate_columns": cols,
                "threshold": threshold,
                "drop_columns": drop_columns,
                "variances": variances,
            },
        )
