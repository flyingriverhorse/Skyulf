"""Lag features: shift columns by N rows to expose past values to the model."""

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...registry import NodeRegistry
from .._artifacts import LagFeaturesArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from ._common import coerce_lags, resolve_columns, sort_pandas


def _lag_name(col: str, lag: int) -> str:
    return f"{col}_lag_{lag}"


def _polars_lag_exprs(
    columns: List[str], available: List[str], lags: List[int], group_by: Optional[List[str]]
) -> list:
    import polars as pl

    exprs = []
    for col in columns:
        if col not in available:
            continue
        for lag in lags:
            expr = pl.col(col).shift(lag)
            if group_by:
                expr = expr.over(group_by)
            exprs.append(expr.alias(_lag_name(col, lag)))
    return exprs


def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    columns: List[str] = params.get("columns", [])
    lags: List[int] = params.get("lags", [])
    sort_by: Optional[str] = params.get("sort_by")
    if not columns or not lags:
        return X, _y

    X_out = X.sort(sort_by) if sort_by and sort_by in X.columns else X
    exprs = _polars_lag_exprs(columns, list(X_out.columns), lags, params.get("group_by") or None)
    if exprs:
        X_out = X_out.with_columns(exprs)
    if params.get("drop_na"):
        X_out = X_out.drop_nulls()
    return X_out, _y


def _pandas_lag_column(df: Any, col: str, lags: List[int], group_by: Optional[List[str]]) -> None:
    source = df.groupby(group_by)[col] if group_by else df[col]
    for lag in lags:
        df[_lag_name(col, lag)] = source.shift(lag)


def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    columns: List[str] = params.get("columns", [])
    lags: List[int] = params.get("lags", [])
    group_by: Optional[List[str]] = params.get("group_by") or None
    if not columns or not lags:
        return X, _y

    df = sort_pandas(X.copy(), params.get("sort_by"))
    for col in columns:
        if col in df.columns:
            _pandas_lag_column(df, col, lags, group_by)
    if params.get("drop_na"):
        df = df.dropna()
    return df, _y


class LagFeaturesApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, _apply_polars, _apply_pandas)


@NodeRegistry.register("LagFeatures", LagFeaturesApplier)
@node_meta(
    id="LagFeatures",
    name="Lag Features",
    category="Preprocessing",
    description="Create lagged copies of columns to expose past values (time series).",
    params={"columns": [], "lags": [1], "group_by": None, "sort_by": None, "drop_na": False},
    tags=["time-series"],
)
class LagFeaturesCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> LagFeaturesArtifact:
        return {
            "type": "lag_features",
            "columns": config.get("columns", []),
            "lags": coerce_lags(config.get("lags", [1])),
            "group_by": config.get("group_by"),
            "sort_by": config.get("sort_by"),
            "drop_na": bool(config.get("drop_na", False)),
        }

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> Optional[SkyulfSchema]:
        # Lag columns mirror the dtype of their source column, so the output
        # schema is derivable from config alone (shape is data-independent).
        cols = resolve_columns(config.get("columns", []), input_schema.column_list())
        lags = coerce_lags(config.get("lags", [1]))
        schema = input_schema
        for col in cols:
            dtype = input_schema.dtypes.get(col, "unknown")
            for lag in lags:
                schema = schema.add(_lag_name(col, lag), dtype)
        return schema
