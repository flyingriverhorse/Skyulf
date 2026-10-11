"""Lag features: shift columns by N rows to expose past values to the model."""

from numbers import Integral
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...registry import NodeRegistry
from .._artifacts import LagFeaturesArtifact
from .._helpers import select_rows_by_position
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from ._common import (
    coerce_lags,
    filter_existing_columns,
    sort_with_positions_pandas,
    sort_with_positions_polars,
)
from ._history_apply import apply_history
from ._history_state import fit_history


def _lag_name(col: str, lag: int) -> str:
    return f"{col}_lag_{lag}"


def _polars_lag_exprs(
    columns: list[str], available: list[str], lags: list[int], group_by: list[str] | None
) -> list:
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


def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Add lag features and filter X/y by the same null-or-NaN keep positions."""
    if params.get("history_mode") == "carry":
        return apply_history(X, _y, params, _apply_polars)
    columns: list[str] = params.get("columns", [])
    lags: list[int] = params.get("lags", [])
    if not columns or not lags:
        return X, _y

    X_out, sort_positions = sort_with_positions_polars(X, params.get("sort_by"))
    _y = select_rows_by_position(_y, sort_positions)
    exprs = _polars_lag_exprs(columns, list(X_out.columns), lags, params.get("group_by") or None)
    if exprs:
        X_out = X_out.with_columns(exprs)
    if params.get("drop_na"):
        return _drop_missing_polars(X_out, _y)
    return X_out, _y


def _drop_missing_polars(X_out: Any, _y: Any) -> tuple[Any, Any]:
    """Drop incomplete observations while retaining their paired target positions."""
    if X_out.columns:
        missing = [pl.col(c).is_null() for c in X_out.columns]
        missing.extend(pl.col(c).is_nan() for c, dtype in X_out.schema.items() if dtype.is_float())
        keep = X_out.select(~pl.any_horizontal(missing)).to_series().arg_true()
        X_out = X_out.gather(keep)
        _y = select_rows_by_position(_y, keep)
    return X_out, _y


def _pandas_lag_column(
    df: Any, source: Any, col: str, lags: list[int], group_by: list[str] | None
) -> None:
    """Assign one source column's lags without reading previously generated values."""
    # `dropna=False` matches Polars' `.over(group_by)` semantics, where a null
    # group key is treated as a normal (self-equal) group rather than
    # excluded. Pandas' `groupby` defaults to `dropna=True`, which would
    # otherwise force every null-group row's lag to NaN regardless of what
    # preceded it, diverging from the Polars apply path on the same data.
    values = source.groupby(group_by, dropna=False)[col] if group_by else source[col]
    for lag in lags:
        df[_lag_name(col, lag)] = values.shift(lag)


def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    if params.get("history_mode") == "carry":
        return apply_history(X, _y, params, _apply_pandas)
    columns: list[str] = params.get("columns", [])
    lags: list[int] = params.get("lags", [])
    group_by: list[str] | None = params.get("group_by") or None
    if not columns or not lags:
        return X, _y

    df, sort_positions = sort_with_positions_pandas(X.copy(), params.get("sort_by"))
    _y = select_rows_by_position(_y, sort_positions)
    source = df.copy(deep=False)
    for col in columns:
        if col in source.columns:
            _pandas_lag_column(df, source, col, lags, group_by)
    if params.get("drop_na"):
        keep = np.flatnonzero(df.notna().to_numpy().all(axis=1))
        df = df.iloc[keep]
        _y = select_rows_by_position(_y, keep)
    return df, _y


class LagFeaturesApplier(BaseApplier):
    """Append lagged copies of the configured columns, optionally within groups."""

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Retain the window dependency even when training history was saved."""
        if engine not in ("pandas", "polars"):
            return None
        effect = "filter" if state.get("drop_na", False) else "preserve"
        return ExecutionCapability(engine, "apply", "local", effect, "window")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Add lag columns; ``drop_na`` drops rows containing nulls or floating NaN."""
        if any(
            not isinstance(lag, Integral) or isinstance(lag, bool) or lag <= 0
            for lag in params.get("lags", [])
        ):
            raise ValueError("LagFeatures artifact lags must be positive integers.")
        return apply_dual_engine(
            (X, _y) if _y is not None else X,
            params,
            {"polars": _apply_polars, "pandas": _apply_pandas},
        )


@NodeRegistry.register("LagFeatures", LagFeaturesApplier)
@node_meta(
    id="LagFeatures",
    name="Lag Features",
    category="Preprocessing",
    description="Create lagged copies of columns to expose past values (time series).",
    params={"columns": [], "lags": [1], "group_by": None, "sort_by": None, "drop_na": False},
    tags=["time-series"],
    learns_from_data=False,
)
class LagFeaturesCalculator(BaseCalculator):
    """Save lag configuration and optional bounded training history.

    Default batch mode uses only supplied rows. Explicit carry mode seeds future
    batches from training observations, without mutating the artifact on apply.
    """

    def fit(
        self,
        df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
        config: dict[str, Any],
    ) -> LagFeaturesArtifact:
        """Record the columns, deduplicated positive lags, and sort/group/drop settings."""
        params: dict[str, Any] = {
            "type": "lag_features",
            "columns": config.get("columns", []),
            "lags": coerce_lags(config.get("lags", [1])),
            "group_by": config.get("group_by"),
            "sort_by": config.get("sort_by"),
            "drop_na": bool(config.get("drop_na", False)),
        }
        return cast(
            LagFeaturesArtifact, fit_history(df, config, params, max(params["lags"], default=0))
        )

    def fit_transform_train(self, df: Any, config: dict[str, Any]) -> tuple[Any, Any]:
        """Save the training tail while computing training lags without future context."""
        params = self.fit(df, config)
        return params, LagFeaturesApplier().apply(df, dict(params) | {"_history_training": True})

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema | None:
        """Add one ``{col}_lag_{n}`` column per lag, as ``float64``."""
        # Shifted lag columns are nullable by construction; pandas/polars keep
        # them as ``float64`` because of the inserted missing row.
        cols = filter_existing_columns(config.get("columns", []), input_schema.column_list())
        lags = coerce_lags(config.get("lags", [1]))
        schema = input_schema
        for col in cols:
            for lag in lags:
                schema = schema.add(_lag_name(col, lag), "float64")
        return schema
