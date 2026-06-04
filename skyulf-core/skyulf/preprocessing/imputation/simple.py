"""Simple imputer node (mean / median / most_frequent / constant)."""

from typing import Any, Dict, List, Tuple, cast

from sklearn.impute import SimpleImputer

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...utils import detect_numeric_columns, user_picked_no_columns
from .._artifacts import SimpleImputerArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from ..dispatcher import apply_dual_engine, fit_dual_engine
from ._common import (
    _compute_polars_fill_values,
    _polars_missing_counts,
    _resolve_simple_columns,
)


class SimpleImputerApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, self._apply_polars, self._apply_pandas)

    @staticmethod
    def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        import polars as pl

        cols = params.get("columns", [])
        fill_values = params.get("fill_values", {})
        if not cols:
            return X, _y

        exprs: List[Any] = []
        for col in X.columns:
            if col in cols and col in fill_values:
                exprs.append(pl.col(col).fill_null(fill_values[col]).alias(col))
            else:
                exprs.append(pl.col(col))

        # Restore columns that were present at fit time but missing in input X.
        for col in cols:
            if col not in X.columns and col in fill_values:
                exprs.append(pl.lit(fill_values[col]).alias(col))

        return X.select(exprs), _y

    @staticmethod
    def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
        cols = params.get("columns", [])
        fill_values = params.get("fill_values", {})
        if not cols:
            return X, _y

        X_out = X.copy()
        for col in cols:
            val = fill_values.get(col)
            if val is None:
                continue
            if col not in X_out.columns:
                X_out[col] = val
            else:
                X_out[col] = X_out[col].fillna(val)
        return X_out, _y


@NodeRegistry.register("SimpleImputer", SimpleImputerApplier)
@node_meta(
    id="SimpleImputer",
    name="Simple Imputer",
    category="Preprocessing",
    description="Imputes missing values using mean, median, or constant.",
    params={"strategy": "mean", "fill_value": None, "columns": []},
)
class SimpleImputerCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> SkyulfSchema:
        # Imputers fill NaNs in place; column set and order are preserved.
        return input_schema

    @fit_method
    def fit(self, X: Any, _y: Any, config: Dict[str, Any]) -> SimpleImputerArtifact:
        if user_picked_no_columns(config):
            return {}

        strategy = config.get("strategy", "mean")
        if strategy == "mode":
            strategy = "most_frequent"
        fill_value = config.get("fill_value", None)

        cols = _resolve_simple_columns(X, config, strategy)
        if not cols:
            return {}

        # Stash resolved-once values into params so dispatched fits don't redo work.
        merged = dict(config)
        merged["_resolved_strategy"] = strategy
        merged["_resolved_cols"] = cols
        merged["_resolved_fill_value"] = fill_value

        return cast(
            SimpleImputerArtifact,
            fit_dual_engine(X, merged, self._fit_polars, self._fit_pandas),
        )

    @staticmethod
    def _fit_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        cols: List[str] = params["_resolved_cols"]
        strategy: str = params["_resolved_strategy"]
        fill_value = params["_resolved_fill_value"]

        fill_values = _compute_polars_fill_values(X, cols, strategy, fill_value)
        missing_counts, total_missing = _polars_missing_counts(X, cols)

        return {
            "type": "simple_imputer",
            "strategy": strategy,
            "fill_values": fill_values,
            "columns": cols,
            "missing_counts": missing_counts,
            "total_missing": total_missing,
        }

    @staticmethod
    def _fit_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        cols: List[str] = params["_resolved_cols"]
        strategy: str = params["_resolved_strategy"]
        fill_value = params["_resolved_fill_value"]

        # Mean/median: extra safety filter to numeric columns only.
        if strategy in ("mean", "median"):
            numeric = set(detect_numeric_columns(X))
            cols = [c for c in cols if c in numeric]
            if not cols:
                return {}

        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        imputer.fit(X[cols])

        statistics = imputer.statistics_.tolist()
        fill_values = dict(zip(cols, statistics))
        missing_counts = X[cols].isnull().sum().to_dict()
        total_missing = int(sum(missing_counts.values()))

        return {
            "type": "simple_imputer",
            "strategy": strategy,
            "fill_values": fill_values,
            "columns": cols,
            "missing_counts": missing_counts,
            "total_missing": total_missing,
        }
