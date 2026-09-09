"""Rolling-window aggregates (mean/sum/min/max/std/median) over time series."""

from typing import Any

import pandas as pd
import polars as pl

from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...registry import NodeRegistry
from .._artifacts import RollingAggregateArtifact
from .._helpers import select_rows_by_position
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from ._common import (
    coerce_aggregations,
    filter_existing_columns,
    sort_with_positions_pandas,
    sort_with_positions_polars,
)


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
    columns: list[str],
    available: dict[str, Any],
    aggs: list[str],
    window: int,
    min_periods: int,
    group_by: list[str] | None,
) -> list:
    """Build rolling expressions that treat floating NaN as missing observations."""
    exprs = []
    for col in columns:
        if col not in available:
            continue
        source = pl.col(col)
        if available[col].is_float():
            source = source.fill_nan(None)
        for agg in aggs:
            expr = _polars_rolling_expr(source, agg, window, min_periods)
            if group_by:
                expr = expr.over(group_by)
            exprs.append(expr.alias(_roll_name(col, agg, window)))
    return exprs


def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    columns: list[str] = params.get("columns", [])
    aggs: list[str] = params.get("aggregations", [])
    if not columns or not aggs:
        return X, _y

    X_out, sort_positions = sort_with_positions_polars(X, params.get("sort_by"))
    _y = select_rows_by_position(_y, sort_positions)
    exprs = _polars_rolling_exprs(
        columns,
        X_out.schema,
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
    aggs: list[str],
    window: int,
    min_periods: int,
    group_by: list[str] | None,
) -> None:
    numeric = pd.to_numeric(df[col], errors="coerce")
    for agg in aggs:
        if group_by:
            # `dropna=False` matches Polars' `.over(group_by)` semantics,
            # where a null group key is a normal (self-equal) group rather
            # than excluded. Pandas' `groupby` defaults to `dropna=True`,
            # which would otherwise force every null-group row's rolling
            # value to NaN, diverging from the Polars apply path.
            grouped = numeric.groupby([df[g] for g in group_by], dropna=False)
            rolled = grouped.transform(
                lambda s, agg=agg: _pandas_rolling(s, agg, window, min_periods)
            )
        else:
            rolled = _pandas_rolling(numeric, agg, window, min_periods)
        df[_roll_name(col, agg, window)] = rolled


def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    columns: list[str] = params.get("columns", [])
    aggs: list[str] = params.get("aggregations", [])
    if not columns or not aggs:
        return X, _y

    window = int(params.get("window", 3))
    min_periods = int(params.get("min_periods", 1))
    group_by: list[str] | None = params.get("group_by") or None
    df, sort_positions = sort_with_positions_pandas(X.copy(), params.get("sort_by"))
    _y = select_rows_by_position(_y, sort_positions)
    for col in columns:
        if col in df.columns:
            _pandas_roll_column(df, col, aggs, window, min_periods, group_by)
    return df, _y


class RollingAggregateApplier(BaseApplier):
    """Append rolling-window aggregate columns for the configured columns."""

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Compute the rolling aggregates on the active engine; ``y`` passes through."""
        return apply_dual_engine(
            (X, _y) if _y is not None else X,
            params,
            {"polars": _apply_polars, "pandas": _apply_pandas},
        )


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
    learns_from_data=False,
)
class RollingAggregateCalculator(BaseCalculator):
    """Normalize the rolling-window configuration into the artifact."""

    def fit(
        self,
        df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
        config: dict[str, Any],
    ) -> RollingAggregateArtifact:
        """Record the window, recognized aggregations, and sort/group settings."""
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
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema | None:
        """Add one float64 ``{col}_roll_{agg}_{window}`` column per (column, aggregation)."""
        # Rolling outputs are float64 regardless of source dtype.
        cols = filter_existing_columns(config.get("columns", []), input_schema.column_list())
        aggs = coerce_aggregations(config.get("aggregations", ["mean"]))
        window = int(config.get("window", 3))
        schema = input_schema
        for col in cols:
            for agg in aggs:
                schema = schema.add(_roll_name(col, agg, window), "float64")
        return schema
