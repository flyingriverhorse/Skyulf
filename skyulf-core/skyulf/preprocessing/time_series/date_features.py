"""Calendar feature extraction from datetime columns (year, month, dow, ...)."""

from typing import Any

import pandas as pd

from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...registry import NodeRegistry
from .._artifacts import DateFeaturesArtifact
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from ._common import DATE_FEATURE_ACCESSORS, resolve_columns

# Default calendar parts when the user does not specify any.
DEFAULT_FEATURES: list[str] = ["year", "month", "day", "dayofweek"]


def _feat_name(col: str, feature: str) -> str:
    return f"{col}_{feature}"


def _resolve_features(config: dict[str, Any]) -> list[str]:
    requested = config.get("features") or DEFAULT_FEATURES
    return [f for f in requested if f in DATE_FEATURE_ACCESSORS]


def _pandas_feature(dt: Any, feature: str) -> Any:
    if feature == "weekofyear":
        return dt.isocalendar().week.astype("int64")
    if feature == "is_weekend":
        return (dt.dayofweek >= 5).astype("int64")
    if feature in ("is_month_start", "is_month_end"):
        return getattr(dt, feature).astype("int64")
    return getattr(dt, feature)


def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    columns: list[str] = params.get("columns", [])
    features: list[str] = params.get("features", [])
    drop_original: bool = params.get("drop_original", False)
    if not columns or not features:
        return X, _y

    df = X.copy()
    for col in columns:
        if col not in df.columns:
            continue
        dt = pd.to_datetime(df[col], errors="coerce").dt
        for feature in features:
            df[_feat_name(col, feature)] = _pandas_feature(dt, feature)
        if drop_original:
            df = df.drop(columns=[col])
    return df, _y


def _polars_feature(col_expr: Any, feature: str) -> Any:
    dt = col_expr.dt
    builders = {
        "year": lambda: dt.year(),
        "month": lambda: dt.month(),
        "day": lambda: dt.day(),
        "dayofweek": lambda: dt.weekday() - 1,
        "dayofyear": lambda: dt.ordinal_day(),
        "quarter": lambda: dt.quarter(),
        "weekofyear": lambda: dt.week(),
        "hour": lambda: dt.hour(),
        "minute": lambda: dt.minute(),
        "is_weekend": lambda: (dt.weekday() >= 6).cast(int),
        "is_month_start": lambda: (dt.day() == 1).cast(int),
        "is_month_end": lambda: (dt.month() != col_expr.dt.offset_by("1d").dt.month()).cast(int),
    }
    return builders[feature]()


def _polars_date_exprs(columns: list[str], available: list[str], features: list[str]) -> list:
    import polars as pl

    exprs = []
    for col in columns:
        if col not in available:
            continue
        base = pl.col(col).cast(pl.Datetime, strict=False)
        for feature in features:
            exprs.append(_polars_feature(base, feature).alias(_feat_name(col, feature)))
    return exprs


def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    columns: list[str] = params.get("columns", [])
    features: list[str] = params.get("features", [])
    if not columns or not features:
        return X, _y

    X_out = X
    exprs = _polars_date_exprs(columns, list(X_out.columns), features)
    if exprs:
        X_out = X_out.with_columns(exprs)
    if params.get("drop_original"):
        drop_cols = [c for c in columns if c in X_out.columns]
        if drop_cols:
            X_out = X_out.drop(drop_cols)
    return X_out, _y


class DateFeaturesApplier(BaseApplier):
    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        return apply_dual_engine(X, params, _apply_polars, _apply_pandas)


@NodeRegistry.register("DateFeatures", DateFeaturesApplier)
@node_meta(
    id="DateFeatures",
    name="Date Features",
    category="Preprocessing",
    description="Extract calendar parts (year, month, day-of-week, ...) from datetime columns.",
    params={"columns": [], "features": DEFAULT_FEATURES, "drop_original": False},
    tags=["time-series"],
)
class DateFeaturesCalculator(BaseCalculator):
    def fit(
        self,
        df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
        config: dict[str, Any],
    ) -> DateFeaturesArtifact:
        return {
            "type": "date_features",
            "columns": config.get("columns", []),
            "features": _resolve_features(config),
            "drop_original": bool(config.get("drop_original", False)),
        }

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema | None:
        # Calendar parts are integers; shape derivable from config alone.
        cols = resolve_columns(config.get("columns", []), input_schema.column_list())
        features = _resolve_features(config)
        schema = input_schema
        for col in cols:
            for feature in features:
                schema = schema.add(_feat_name(col, feature), "int64")
        if config.get("drop_original"):
            schema = schema.drop(cols)
        return schema
