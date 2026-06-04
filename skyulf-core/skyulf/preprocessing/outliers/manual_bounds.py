"""Manual-bounds outlier node (user-specified lower/upper per column)."""

from typing import Any, Dict, Tuple

import pandas as pd

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import ManualBoundsArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _apply_pandas_mask, _filter_y_polars


def _manual_bounds_col_mask_polars(col: str, bound: Dict[str, Any]) -> Any:
    """Build a Polars per-column inlier mask from optional lower/upper bounds."""
    import polars as pl

    lower = bound.get("lower")
    upper = bound.get("upper")
    col_mask = pl.lit(True)
    if lower is not None:
        col_mask = col_mask & (pl.col(col) >= lower)
    if upper is not None:
        col_mask = col_mask & (pl.col(col) <= upper)
    return col_mask | pl.col(col).is_null()


def _manual_bounds_col_mask_pandas(series: pd.Series, bound: Dict[str, Any]) -> pd.Series:
    """Build a Pandas per-column inlier mask from optional lower/upper bounds."""
    lower = bound.get("lower")
    upper = bound.get("upper")
    col_mask = pd.Series(True, index=series.index)
    if lower is not None:
        col_mask = col_mask & (series >= lower)
    if upper is not None:
        col_mask = col_mask & (series <= upper)
    return col_mask | series.isna()


class ManualBoundsApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        mask = pl.lit(True)
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
            mask = mask & _manual_bounds_col_mask_polars(col, bound)

        mask_series = X.select(mask.alias("mask")).get_column("mask")
        return X.filter(mask_series), _filter_y_polars(y, mask_series)

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        mask = pd.Series(True, index=X.index)
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
            series = pd.to_numeric(X[col], errors="coerce")
            mask = mask & _manual_bounds_col_mask_pandas(series, bound)

        return _apply_pandas_mask(X, y, mask)


@NodeRegistry.register("ManualBounds", ManualBoundsApplier)
@node_meta(
    id="ManualBounds",
    name="Manual Bounds",
    category="Preprocessing",
    description="Filter outliers by manually specifying lower and upper bounds for columns.",
    params={"bounds": {}},
)
class ManualBoundsCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Manual bounds filter rows; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, _X: Any, _y: Any, config: Dict[str, Any]) -> ManualBoundsArtifact:
        # Config: {'bounds': {'col1': {'lower': 0, 'upper': 100}, ...}}
        return {"type": "manual_bounds", "bounds": config.get("bounds", {})}
