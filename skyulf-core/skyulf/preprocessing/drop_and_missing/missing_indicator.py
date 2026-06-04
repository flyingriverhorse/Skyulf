"""Missing-indicator node (binary flags for missing values)."""

from typing import Any, Dict, Optional, Tuple, cast

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import MissingIndicatorArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine, fit_dual_engine


def _missing_indicator_apply_polars(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    import polars as pl

    cols = params.get("columns", [])
    if not cols:
        return X, y
    exprs = [
        pl.col(c).is_null().cast(pl.Int64).alias(f"{c}_missing") for c in cols if c in X.columns
    ]
    return (X.with_columns(exprs) if exprs else X), y


def _missing_indicator_apply_pandas(X: Any, y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    cols = params.get("columns", [])
    if not cols:
        return X, y
    X_out = X.copy()
    for col in cols:
        if col in X.columns:
            X_out[f"{col}_missing"] = X[col].isna().astype(int)
    return X_out, y


class MissingIndicatorApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(
            X, params, _missing_indicator_apply_polars, _missing_indicator_apply_pandas
        )


def _missing_cols_polars(X: Any) -> list:
    null_counts = X.null_count()
    return [c for c in X.columns if null_counts[c][0] > 0]


def _missing_cols_pandas(X: Any) -> list:
    return X.columns[X.isna().any()].tolist()


def _missing_indicator_fit_polars(
    X: Any, _y: Any, config: Dict[str, Any]
) -> MissingIndicatorArtifact:
    explicit = config.get("columns")
    cols = [c for c in explicit if c in X.columns] if explicit else _missing_cols_polars(X)
    return {"type": "missing_indicator", "columns": cols}


def _missing_indicator_fit_pandas(
    X: Any, _y: Any, config: Dict[str, Any]
) -> MissingIndicatorArtifact:
    explicit = config.get("columns")
    cols = [c for c in explicit if c in X.columns] if explicit else _missing_cols_pandas(X)
    return {"type": "missing_indicator", "columns": cols}


@NodeRegistry.register("MissingIndicator", MissingIndicatorApplier)
@node_meta(
    id="MissingIndicator",
    name="Missing Indicator",
    category="Feature Engineering",
    description="Create binary indicators for missing values.",
    params={"features": "missing-only", "sparse": "auto"},
)
class MissingIndicatorCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> Optional[SkyulfSchema]:
        # Adds one boolean column "<col>_missing" per indicator column.
        # Only predictable when the user supplied an explicit column list;
        # otherwise the set depends on which columns actually contain NaNs.
        explicit = config.get("columns") or []
        if not explicit:
            return None
        new_schema = input_schema
        for col in explicit:
            new_schema = new_schema.add(f"{col}_missing", "bool")
        return new_schema

    def fit(self, df: Any, config: Dict[str, Any]) -> MissingIndicatorArtifact:
        return cast(
            MissingIndicatorArtifact,
            fit_dual_engine(
                df, config, _missing_indicator_fit_polars, _missing_indicator_fit_pandas
            ),
        )
