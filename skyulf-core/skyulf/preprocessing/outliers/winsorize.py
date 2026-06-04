"""Winsorize node (clip extreme values to percentile bounds)."""

from typing import Any, Dict, Tuple

import pandas as pd

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, resolve_columns, user_picked_no_columns
from .._artifacts import WinsorizeArtifact
from .._helpers import to_pandas
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine


class WinsorizeApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        exprs = []
        for col, bound in bounds.items():
            if col not in X.columns:
                continue
            # Cast to float to avoid integer truncation when bounds are float.
            exprs.append(
                pl.col(col).cast(pl.Float64).clip(bound["lower"], bound["upper"]).alias(col)
            )
        return X.with_columns(exprs), y

    @staticmethod
    def _apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        bounds = params.get("bounds", {})
        if not bounds:
            return X, y

        df_out = X.copy()
        for col, bound in bounds.items():
            if col not in df_out.columns:
                continue
            if pd.api.types.is_numeric_dtype(df_out[col]):
                df_out[col] = df_out[col].clip(lower=bound["lower"], upper=bound["upper"])
        return df_out, y


@NodeRegistry.register("Winsorize", WinsorizeApplier)
@node_meta(
    id="Winsorize",
    name="Winsorization",
    category="Preprocessing",
    description="Limit extreme values in the data.",
    params={"limits": [0.05, 0.05], "columns": []},
)
class WinsorizeCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Winsorize clips values in place; column set is preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> WinsorizeArtifact:
        if user_picked_no_columns(config):
            return {}

        X_pd = to_pandas(X)
        lower_p = config.get("lower_percentile", 5.0)
        upper_p = config.get("upper_percentile", 95.0)
        cols = resolve_columns(X_pd, config, detect_numeric_columns)
        if not cols:
            return {}

        bounds: Dict[str, Dict[str, float]] = {}
        warnings = []
        for col in cols:
            series = pd.to_numeric(X_pd[col], errors="coerce").dropna()
            if series.empty:
                warnings.append(f"Column '{col}': Empty or non-numeric")
                continue
            bounds[col] = {
                "lower": series.quantile(lower_p / 100.0),
                "upper": series.quantile(upper_p / 100.0),
            }

        return {
            "type": "winsorize",
            "bounds": bounds,
            "lower_percentile": lower_p,
            "upper_percentile": upper_p,
            "warnings": warnings,
        }
