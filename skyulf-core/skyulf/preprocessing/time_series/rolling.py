"""Rolling-window aggregates (mean/sum/min/max/std/median) over time series."""

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...registry import NodeRegistry
from .._artifacts import RollingAggregateArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from ._common import coerce_aggregations, resolve_columns, sort_pandas


def _roll_name(col: str, agg: str, window: int) -> str:
    return f"{col}_roll_{agg}_{window}"


def _polars_rolling_expr(col_expr: Any, agg: str, window: int, min_periods: int) -> Any:
    # polars >= 1.21 renamed ``min_periods`` to ``min_samples``.
    builders = {
        "mean": lambda e: e.rolling_mean(window, min_samples=min_periods),
        "sum": lambda e: e.rolling_sum(window, min_samples=min_periods),
        "min": lambda e: e.rolling_min(window, min_samples=min_periods),
        "max": lambda e: e.rolling_max(window, min_samples=min_periods),
        "std": lambda e: e.rolling_std(window, min_samples=min_periods),
        "median": lambda e: e.rolling_median(window, min_samples=min_periods),
    }
    return builders[agg](col_expr)


def _polars_rolling_exprs(
    columns: List[str],
    available: List[str],
    aggs: List[str],
    window: int,
    min_periods: int,
    group_by: Optional[List[str]],
) -> list:
    import polars as pl

    exprs = []
    for col in columns:
        if col not in available:
            continue
        for agg in aggs:
            expr = _polars_rolling_expr(pl.col(col), agg, window, min_periods)
            if group_by:
                expr = expr.over(group_by)
            exprs.append(expr.alias(_roll_name(col, agg, window)))
    return exprs


def _apply_polars(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    columns: List[str] = params.get("columns", [])
    aggs: List[str] = params.get("aggregations", [])
    sort_by: Optional[str] = params.get("sort_by")
    if not columns or not aggs:
        return X, _y

    X_out = X.sort(sort_by) if sort_by and sort_by in X.columns else X
    exprs = _polars_rolling_exprs(
        columns,
        list(X_out.columns),
        aggs,
        int(params.get("window", 3)),
        int(params.get("min_periods", 1)),
        params.get("group_by") or None,
    )
    if exprs:
        X_out = X_out.with_columns(exprs)
    return X_out, _y


def _pandas_rolling(series: Any, agg: str, window: int, min_periods: int) -> Any:
    roller = series.rolling(window=window, min_periods=min_periods)
    return getattr(roller, agg)()


def _pandas_roll_column(
    df: Any,
    col: str,
    aggs: List[str],
    window: int,
    min_periods: int,
    group_by: Optional[List[str]],
) -> None:
    numeric = pd.to_numeric(df[col], errors="coerce")
    for agg in aggs:
        if group_by:
            grouped = numeric.groupby([df[g] for g in group_by])
            rolled = grouped.transform(lambda s: _pandas_rolling(s, agg, window, min_periods))
        else:
            rolled = _pandas_rolling(numeric, agg, window, min_periods)
        df[_roll_name(col, agg, window)] = rolled


def _apply_pandas(X: Any, _y: Any, params: Dict[str, Any]) -> Tuple[Any, Any]:
    columns: List[str] = params.get("columns", [])
    aggs: List[str] = params.get("aggregations", [])
    if not columns or not aggs:
        return X, _y

    window = int(params.get("window", 3))
    min_periods = int(params.get("min_periods", 1))
    group_by: Optional[List[str]] = params.get("group_by") or None
    df = sort_pandas(X.copy(), params.get("sort_by"))
    for col in columns:
        if col in df.columns:
            _pandas_roll_column(df, col, aggs, window, min_periods, group_by)
    return df, _y


class RollingAggregateApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: Dict[str, Any]) -> Any:
        return apply_dual_engine(X, params, _apply_polars, _apply_pandas)


@NodeRegistry.register("RollingAggregate", RollingAggregateApplier)
@node_meta(
    id="RollingAggregate",
    name="Rolling Aggregate",
    category="Preprocessing",
    description="Rolling-window aggregates (mean/sum/min/max/std/median) for time series.",
    params={
        "columns": [],
        "window": 3,
        "aggregations": ["mean"],
        "min_periods": 1,
        "group_by": None,
        "sort_by": None,
    },
    tags=["time-series"],
)
class RollingAggregateCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, SkyulfDataFrame, Tuple[Any, ...], Any],
        config: Dict[str, Any],
    ) -> RollingAggregateArtifact:
        return {
            "type": "rolling_aggregate",
            "columns": config.get("columns", []),
            "window": int(config.get("window", 3)),
            "aggregations": coerce_aggregations(config.get("aggregations", ["mean"])),
            "min_periods": int(config.get("min_periods", 1)),
            "group_by": config.get("group_by"),
            "sort_by": config.get("sort_by"),
        }

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: Dict[str, Any]
    ) -> Optional[SkyulfSchema]:
        # Rolling outputs are float64 regardless of source dtype.
        cols = resolve_columns(config.get("columns", []), input_schema.column_list())
        aggs = coerce_aggregations(config.get("aggregations", ["mean"]))
        window = int(config.get("window", 3))
        schema = input_schema
        for col in cols:
            for agg in aggs:
                schema = schema.add(_roll_name(col, agg, window), "float64")
        return schema
