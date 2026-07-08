"""Variance-threshold feature selector."""

from typing import Any, cast

from sklearn.feature_selection import VarianceThreshold

from ...core.meta.decorators import node_meta
from ...engines.sklearn_bridge import SklearnBridge
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, resolve_columns
from .._artifacts import VarianceThresholdArtifact
from .._helpers import to_pandas
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _drop_selected_pandas, _drop_selected_polars


class VarianceThresholdApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, _drop_selected_polars, _drop_selected_pandas)


@NodeRegistry.register("VarianceThreshold", VarianceThresholdApplier)
@node_meta(
    id="VarianceThreshold",
    name="Variance Threshold",
    category="Feature Selection",
    description="Remove features with low variance.",
    params={"threshold": 0.0},
)
class VarianceThresholdCalculator(BaseCalculator):
    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> VarianceThresholdArtifact:  # pylint: disable=arguments-differ
        threshold = config.get("threshold", 0.0)
        drop_columns = config.get("drop_columns", True)

        X_pd = to_pandas(X)
        cols = resolve_columns(
            X_pd,
            config,
            lambda d: detect_numeric_columns(d, exclude_binary=False, exclude_constant=False),
        )
        if not cols:
            return cast(VarianceThresholdArtifact, {})

        selector = VarianceThreshold(threshold=threshold)
        X_np, _ = SklearnBridge.to_sklearn(X_pd[cols])
        selector.fit(X_np)

        support = selector.get_support()
        selected_cols = [c for c, s in zip(cols, support, strict=True) if s]
        variances = (
            dict(zip(cols, selector.variances_.tolist(), strict=True)) if hasattr(selector, "variances_") else {}
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
