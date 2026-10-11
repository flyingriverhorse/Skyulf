"""Calendar feature extraction from datetime columns (year, month, dow, ...)."""

from typing import Any

import pandas as pd
import polars as pl

from ...core.capabilities import ExecutionCapability
from ...core.meta.decorators import node_meta
from ...engines import SkyulfDataFrame
from ...registry import NodeRegistry
from .._artifacts import DateFeaturesArtifact
from .._fitted_validation import local_boolean, local_state_fields
from .._schema import SkyulfSchema
from ..base import BaseApplier, BaseCalculator, apply_method
from ..dispatcher import apply_dual_engine
from ._common import DATE_FEATURE_ACCESSORS, filter_existing_columns, parse_datetime_scalar

# Default calendar parts when the user does not specify any.
DEFAULT_FEATURES: list[str] = ["year", "month", "day", "dayofweek"]
_EPOCH_NS_MULTIPLIERS = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}


def _epoch_unit(params: dict[str, Any]) -> str | None:
    """Validate an explicit epoch unit without guessing it from value magnitude."""
    unit = params.get("epoch_unit")
    if unit is not None and (type(unit) is not str or unit not in _EPOCH_NS_MULTIPLIERS):
        raise ValueError("DateFeatures epoch_unit must be one of s, ms, us or ns.")
    return unit


def _require_epoch_unit(unit: str | None) -> str:
    """Reject numeric timestamps whose artifact does not establish their unit."""
    if unit is None:
        raise ValueError(
            "DateFeatures numeric timestamps require explicit epoch_unit (s, ms, us, ns); "
            "set the source unit and refit legacy artifacts."
        )
    return unit


def _epoch_bounds(unit: str, integral: bool) -> tuple[int | float, int | float]:
    """Bound source values before conversion to the common nanosecond timestamp range."""
    multiplier = _EPOCH_NS_MULTIPLIERS[unit]
    maximum = pd.Timestamp.max.value
    if integral:
        return -(maximum // multiplier), maximum // multiplier
    return -maximum / multiplier, maximum / multiplier


def _pandas_datetime_series(series: pd.Series, params: dict[str, Any]) -> pd.Series:
    """Parse numeric epochs with explicit units and coerce invalid magnitudes before pandas."""
    options: dict[str, Any] = {"format": "mixed"}
    if pd.api.types.is_numeric_dtype(series):
        unit = _require_epoch_unit(_epoch_unit(params))
        lower, upper = _epoch_bounds(unit, pd.api.types.is_integer_dtype(series))
        if pd.api.types.is_integer_dtype(series):
            nullable_dtype = "UInt64" if pd.api.types.is_unsigned_integer_dtype(series) else "Int64"
            series = series.astype(nullable_dtype)
        series = series.where(series.between(lower, upper))
        options = {"unit": unit}
    return pd.to_datetime(series, errors="coerce", utc=params.get("timezone") == "UTC", **options)


def _validate_numeric_date_columns(df: Any, columns: list[str], unit: str | None) -> None:
    """Check native or wrapped fit input before recording ambiguous numeric date features."""
    frame = df[0] if isinstance(df, tuple) else df
    frame = frame.to_native() if hasattr(frame, "to_native") else frame
    for col in columns:
        if col not in frame.columns:
            continue
        dtype = frame[col].dtype
        numeric = (
            dtype.is_numeric()
            if isinstance(frame, pl.DataFrame)
            else pd.api.types.is_numeric_dtype(dtype)
        )
        if numeric:
            _require_epoch_unit(unit)


def _feat_name(col: str, feature: str) -> str:
    return f"{col}_{feature}"


def _validate_date_names(value: Any) -> None:
    """Retain native string sequences, including duplicates preserved by date fit."""
    if type(value) not in (list, tuple) or any(not isinstance(item, str) for item in value):
        raise ValueError("Fitted date columns and features must be string sequences.")


def _resolve_features(config: dict[str, Any]) -> list[str]:
    requested = config.get("features") or DEFAULT_FEATURES
    return [f for f in requested if f in DATE_FEATURE_ACCESSORS]


def _pandas_feature(dt: Any, feature: str, is_null: Any) -> Any:
    """Compute one calendar feature from a pandas ``.dt`` accessor.

    All numeric features are returned as nullable ``Int64`` (not plain
    ``int64``/``float64``) so an invalid/unparseable date produces a proper
    null - matching polars' `Int32`/null semantics for the same input,
    instead of silently becoming `NaN` (for plain integer properties) or a
    misleading `0`/`False` (for the boolean-derived features below, whose
    NaN-vs-comparison behavior doesn't propagate nulls on its own).
    """
    if feature == "weekofyear":
        return dt.isocalendar().week.astype("Int64")
    if feature == "is_weekend":
        result = (dt.dayofweek >= 5).astype("Int64")
        result[is_null] = pd.NA
        return result
    if feature in ("is_month_start", "is_month_end"):
        result = getattr(dt, feature).astype("Int64")
        result[is_null] = pd.NA
        return result
    return getattr(dt, feature).astype("Int64")


def _apply_pandas(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    columns: list[str] = params.get("columns", [])
    features: list[str] = params.get("features", [])
    drop_original: bool = params.get("drop_original", False)
    if not columns or not features:
        return X, _y

    df = X.copy()
    _epoch_unit(params)
    for col in columns:
        if col not in df.columns:
            continue
        # Older fitted artifacts keep their original local-calendar semantics.
        parsed = _pandas_datetime_series(df[col], params)
        is_null = parsed.isna()
        dt = parsed.dt
        for feature in features:
            df[_feat_name(col, feature)] = _pandas_feature(dt, feature, is_null)
        if drop_original:
            df = df.drop(columns=[col])
    return df, _y


def _polars_feature(col_expr: Any, feature: str) -> Any:
    dt = col_expr.dt
    builders = {
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "dayofweek": lambda: dt.weekday() - 1,
        "dayofyear": dt.ordinal_day,
        "quarter": dt.quarter,
        "weekofyear": dt.week,
        "hour": dt.hour,
        "minute": dt.minute,
        "is_weekend": lambda: (dt.weekday() >= 6).cast(int),
        "is_month_start": lambda: (dt.day() == 1).cast(int),
        "is_month_end": lambda: (dt.month() != col_expr.dt.offset_by("1d").dt.month()).cast(int),
    }
    return builders[feature]()


def _polars_base_expr(col: str, dtype: Any, epoch_unit: str | None = None) -> Any:
    """Build the datetime expression for ``col``, dispatching on its source dtype.

    Plain ``cast(pl.Datetime, strict=False)`` only parses columns that are already
    temporal (or full ISO-8601 datetime strings); it silently returns null for
    ordinary date strings like "2021-01-01". String/Utf8 columns must instead go
    be parsed independently so companion rows cannot determine their format.
    """
    if dtype in (pl.Utf8, pl.String):
        return pl.col(col).map_elements(
            parse_datetime_scalar, return_dtype=pl.Datetime(time_zone="UTC")
        )
    if dtype.is_numeric():
        unit = _require_epoch_unit(epoch_unit)
        # Widen integer arithmetic before unit multiplication so overflow becomes
        # null on the final bounded timestamp cast instead of wrapping a date.
        values = pl.col(col).cast(pl.Int128) if dtype.is_integer() else pl.col(col)
        lower, upper = _epoch_bounds(unit, dtype.is_integer())
        values = pl.when(pl.col(col).is_between(lower, upper)).then(values).otherwise(None)
        return (
            (values * _EPOCH_NS_MULTIPLIERS[unit])
            .cast(pl.Int64, strict=False)
            .cast(pl.Datetime("ns", "UTC"))
        )
    return pl.col(col).cast(pl.Datetime, strict=False)


def _polars_date_exprs(
    columns: list[str], schema: dict[str, Any], features: list[str], epoch_unit: str | None = None
) -> list:
    exprs = []
    for col in columns:
        if col not in schema:
            continue
        base = _polars_base_expr(col, schema[col], epoch_unit)
        exprs.extend(
            _polars_feature(base, feature).alias(_feat_name(col, feature)) for feature in features
        )
    return exprs


def _apply_polars(X: Any, _y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    columns: list[str] = params.get("columns", [])
    features: list[str] = params.get("features", [])
    if not columns or not features:
        return X, _y

    X_out = X
    exprs = _polars_date_exprs(columns, dict(X_out.schema), features, _epoch_unit(params))
    if exprs:
        X_out = X_out.with_columns(exprs)
    if params.get("drop_original"):
        drop_cols = [c for c in columns if c in X_out.columns]
        if drop_cols:
            X_out = X_out.drop(drop_cols)
    return X_out, _y


class DateFeaturesApplier(BaseApplier):
    """Append calendar-part columns extracted from the configured datetime columns."""

    @staticmethod
    def validate_inference_state(raw: dict) -> dict:
        """Inspect saved calendar settings without parsing data or changing legacy defaults."""
        optional = {"drop_original", "timezone", "epoch_unit"}
        fields = {"type", "columns", "features"}
        if type(raw) is dict:
            fields.update(optional.intersection(raw))
        local_state_fields(raw, "date_features", fields)
        _validate_date_names(raw["columns"])
        _validate_date_names(raw["features"])
        if any(feature not in DATE_FEATURE_ACCESSORS for feature in raw["features"]):
            raise ValueError("Fitted date features contain unsupported calendar parts.")
        local_boolean(raw.get("drop_original", False), "drop_original")
        if raw.get("timezone") is not None and type(raw["timezone"]) is not str:
            raise ValueError("Fitted date timezone must be a string or None.")
        _epoch_unit(raw)
        return raw

    @staticmethod
    def inference_capability(state: dict, *, engine: str) -> ExecutionCapability | None:
        """Describe saved UTC parsing without promising legacy mixed-timezone behavior."""
        if engine not in ("pandas", "polars"):
            return None
        DateFeaturesApplier.validate_inference_state(state)
        if state.get("timezone") != "UTC":
            return None
        return ExecutionCapability(engine, "apply", "local", "preserve", "row")

    @apply_method
    def apply(self, X: Any, _y: Any, params: dict[str, Any]) -> Any:  # pylint: disable=arguments-differ
        """Extract the configured calendar features on the active engine; ``y`` passes through."""
        return apply_dual_engine(X, params, {"polars": _apply_polars, "pandas": _apply_pandas})


@NodeRegistry.register("DateFeatures", DateFeaturesApplier)
@node_meta(
    id="DateFeatures",
    name="Date Features",
    category="Preprocessing",
    description="Extract calendar parts (year, month, day-of-week, ...) from datetime columns.",
    params={
        "columns": [],
        "features": DEFAULT_FEATURES,
        "drop_original": False,
        "epoch_unit": None,
    },
    tags=["time-series"],
    learns_from_data=False,
)
class DateFeaturesCalculator(BaseCalculator):
    """Record calendar features using UTC for every newly fitted artifact.

    Offset-aware timestamps are converted to UTC before extracting their
    calendar parts; naive values are treated as UTC without shifting their
    clock time. Unparseable values produce null features. Artifacts fitted
    before this contract retain their original engine-specific interpretation
    so an existing model's input features do not silently change on reload.
    Refit the complete pipeline to adopt UTC for such models. Numeric epochs
    require an explicit ``epoch_unit`` (s, ms, us or ns), including on replay
    of legacy artifacts; units are never inferred from numeric magnitude.
    """

    def fit(
        self,
        df: pd.DataFrame | SkyulfDataFrame | tuple[Any, ...] | Any,
        config: dict[str, Any],
    ) -> DateFeaturesArtifact:
        """Record the columns and supported features, defaulting to year/month/day/dayofweek."""
        unit = _epoch_unit(config)
        _validate_numeric_date_columns(df, config.get("columns", []), unit)
        return {
            "type": "date_features",
            "columns": config.get("columns", []),
            "features": _resolve_features(config),
            "drop_original": bool(config.get("drop_original", False)),
            "timezone": "UTC",
            "epoch_unit": unit,
        }

    def infer_output_schema(
        self, input_schema: SkyulfSchema, config: dict[str, Any]
    ) -> SkyulfSchema | None:
        """Add a nullable-int column per (column, feature); drop originals on request."""
        # Calendar parts are nullable integers (an unparseable date yields a
        # null feature value, not 0/NaN) - "Int64" here is a best-effort,
        # engine-agnostic label communicating nullability; the *actual*
        # column width still varies per engine/feature (e.g. polars uses
        # Int8/Int16/Int32 for narrower fields like month/day/hour), so this
        # is intentionally not asserted via check_dtypes=True comparisons.
        cols = filter_existing_columns(config.get("columns", []), input_schema.column_list())
        features = _resolve_features(config)
        schema = input_schema
        for col in cols:
            for feature in features:
                schema = schema.add(_feat_name(col, feature), "Int64")
        if config.get("drop_original"):
            schema = schema.drop(cols)
        return schema
