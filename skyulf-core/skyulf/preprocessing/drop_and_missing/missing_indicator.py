"""Missing-indicator node (binary flags for missing values)."""

from typing import Any, cast

from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from .._artifacts import MissingIndicatorArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine, fit_dual_engine

_DEFAULT_FLAG_SUFFIX = "_missing"


def _missing_indicator_apply_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    import polars as pl

    cols = params.get("columns", [])
    if not cols:
        return X, y
    suffix = params.get("flag_suffix") or _DEFAULT_FLAG_SUFFIX
    exprs = [
        pl.col(c).is_null().cast(pl.Int64).alias(f"{c}{suffix}") for c in cols if c in X.columns
    ]
    return (X.with_columns(exprs) if exprs else X), y


def _missing_indicator_apply_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    cols = params.get("columns", [])
    if not cols:
        return X, y
    suffix = params.get("flag_suffix") or _DEFAULT_FLAG_SUFFIX
    X_out = X.copy()
    for col in cols:
        if col in X.columns:
            X_out[f"{col}{suffix}"] = X[col].isna().astype(int)
    return X_out, y


class MissingIndicatorApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(
            X, params, _missing_indicator_apply_polars, _missing_indicator_apply_pandas
        )


def _missing_cols_polars(X: Any) -> list:
    null_counts = X.null_count()
    return [c for c in X.columns if null_counts[c][0] > 0]


def _missing_cols_pandas(X: Any) -> list:
    return X.columns[X.isna().any()].tolist()


def _missing_indicator_fit_polars(
    X: Any, _y: Any, config: dict[str, Any]
) -> MissingIndicatorArtifact:
    explicit = config.get("columns")
    cols = [c for c in explicit if c in X.columns] if explicit else _missing_cols_polars(X)
    return {
        "type": "missing_indicator",
        "columns": cols,
        "flag_suffix": config.get("flag_suffix") or _DEFAULT_FLAG_SUFFIX,
    }


def _missing_indicator_fit_pandas(
    X: Any, _y: Any, config: dict[str, Any]
) -> MissingIndicatorArtifact:
    explicit = config.get("columns")
    cols = [c for c in explicit if c in X.columns] if explicit else _missing_cols_pandas(X)
    return {
        "type": "missing_indicator",
        "columns": cols,
        "flag_suffix": config.get("flag_suffix") or _DEFAULT_FLAG_SUFFIX,
    }


@NodeRegistry.register("MissingIndicator", MissingIndicatorApplier)
@node_meta(
    id="MissingIndicator",
    name="Missing Indicator",
    category="Feature Engineering",
    description="Create binary indicators for missing values.",
    params={"columns": [], "flag_suffix": _DEFAULT_FLAG_SUFFIX},
)
class MissingIndicatorCalculator(BaseCalculator):
    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema | None:
        # Adds one boolean column "<col><flag_suffix>" per indicator column.
        # Only predictable when the user supplied an explicit column list;
        # otherwise the set depends on which columns actually contain NaNs.
        explicit = config.get("columns") or []
        if not explicit:
            return None
        suffix = config.get("flag_suffix") or _DEFAULT_FLAG_SUFFIX
        new_schema = input_schema
        for col in explicit:
            new_schema = new_schema.add(f"{col}{suffix}", "bool")
        return new_schema

    def fit(self, df: Any, config: dict[str, Any]) -> MissingIndicatorArtifact:
        return cast(
            MissingIndicatorArtifact,
            fit_dual_engine(
                df, config, _missing_indicator_fit_polars, _missing_indicator_fit_pandas
            ),
        )
