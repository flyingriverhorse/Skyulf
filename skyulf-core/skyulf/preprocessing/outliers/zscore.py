"""Z-Score outlier-removal node."""

from typing import Any

import pandas as pd

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, resolve_columns, user_picked_no_columns
from .._artifacts import ZScoreArtifact
from .._helpers import to_pandas
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine
from ._common import _apply_pandas_mask, _filter_y_polars


class ZScoreApplier(BaseApplier):
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

        stats = params.get("stats", {})
        threshold = params.get("threshold", 3.0)
        if not stats:
            return X, y

        mask = pl.lit(True)
        for col, stat in stats.items():
            if col not in X.columns or stat["std"] == 0:
                continue
            z = (pl.col(col) - stat["mean"]) / stat["std"]
            col_mask = z.abs() <= threshold
            mask = mask & (col_mask | pl.col(col).is_null())

        mask_series = X.select(mask.alias("mask")).get_column("mask")
        return X.filter(mask_series), _filter_y_polars(y, mask_series)

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        stats = params.get("stats", {})
        threshold = params.get("threshold", 3.0)
        if not stats:
            return X, y

        mask = pd.Series(True, index=X.index)
        for col, stat in stats.items():
            if col not in X.columns or stat["std"] == 0:
                continue
            series = pd.to_numeric(X[col], errors="coerce")
            z = (series - stat["mean"]) / stat["std"]
            col_mask = z.abs() <= threshold
            mask = mask & (col_mask | series.isna())

        return _apply_pandas_mask(X, y, mask)


@NodeRegistry.register("ZScore", ZScoreApplier)
@node_meta(
    id="ZScore",
    name="Z-Score Outlier Removal",
    category="Preprocessing",
    description="Remove outliers using Z-Score.",
    params={"threshold": 3.0, "columns": []},
)
class ZScoreCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema:
        # Z-score removes outlier *rows*; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: dict[str, Any]) -> ZScoreArtifact:  # pylint: disable=arguments-differ
        if user_picked_no_columns(config):
            return {}

        X_pd = to_pandas(X)
        threshold = config.get("threshold", 3.0)
        cols = resolve_columns(X_pd, config, detect_numeric_columns)
        if not cols:
            return {}

        stats: dict[str, dict[str, float]] = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.empty:
                warnings.append(f"Column '{col}': Empty or non-numeric")
                continue
            std = series.std(ddof=0)
            if std == 0:
                warnings.append(f"Column '{col}': Zero variance (std=0)")
                continue
            stats[col] = {"mean": series.mean(), "std": std}

        return {
            "type": "zscore",
            "stats": stats,
            "threshold": threshold,
            "warnings": warnings,
        }
