"""IQR outlier-removal node (Interquartile Range)."""

from typing import Any

import pandas as pd

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, resolve_columns, user_picked_no_columns
from .._artifacts import IQRArtifact
from .._helpers import to_pandas
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _apply_pandas_mask, _filter_y_polars


class IQRApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        # apply_method already unpacked (X, y); re-wrap so apply_dual_engine's
        # own unpack_pipeline_input doesn't silently drop y (leaving it
        # unfiltered when X rows are removed). Omit the wrap when y is None
        # to avoid apply_dual_engine's tuple-with-no-y warning log.
        input_data = (X, y) if y is not None else X
        return apply_dual_engine(input_data, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        import polars as pl

        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        mask = pl.lit(True)
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
            col_mask = (pl.col(col) >= bound["lower"]) & (pl.col(col) <= bound["upper"])
            mask = mask & (col_mask | pl.col(col).is_null())

        mask_series = X.select(mask.alias("mask")).get_column("mask")
        return X.filter(mask_series), _filter_y_polars(y, mask_series)

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        mask = pd.Series(True, index=X.index)
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
            series = pd.to_numeric(X[col], errors="coerce")
            col_mask = (series >= bound["lower"]) & (series <= bound["upper"])
            mask = mask & (col_mask | series.isna())

        return _apply_pandas_mask(X, y, mask)


@NodeRegistry.register("IQR", IQRApplier)
@node_meta(
    id="IQR",
    name="IQR Outlier Removal",
    category="Preprocessing",
    description="Remove outliers using Interquartile Range.",
    params={"multiplier": 1.5, "columns": []},
)
class IQRCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        # IQR removes outlier *rows*; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> IQRArtifact:  # pylint: disable=arguments-differ
        if user_picked_no_columns(config):
            return {}

        X_pd = to_pandas(X)
        multiplier = config.get("multiplier", 1.5)
        cols = resolve_columns(X_pd, config, detect_numeric_columns)
        if not cols:
            return {}

        bounds: dict[str, dict[str, float]] = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.empty:
                warnings.append(f"Column '{col}': Empty or non-numeric")
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            bounds[col] = {"lower": q1 - (multiplier * iqr), "upper": q3 + (multiplier * iqr)}

        return {
            "type": "iqr",
            "bounds": bounds,
            "multiplier": multiplier,
            "warnings": warnings,
        }
