"""Feature math node for advanced feature engineering combinations."""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import pandas as pd

from core.feature_engineering.schemas import (
    FeatureMathNodeSignal,
    FeatureMathOperationResult,
)
from core.utils.datetime import utcnow

from ...shared.utils import _coerce_config_boolean, _coerce_string_list

fuzz: Any
try:  # pragma: no cover - optional dependency
    from rapidfuzz import fuzz as _rapidfuzz_fuzz

    fuzz = _rapidfuzz_fuzz
    _HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover - graceful fallback
    fuzz = None
    _HAS_RAPIDFUZZ = False


logger = logging.getLogger(__name__)


FEATURE_MATH_ALLOWED_TYPES = {
    "arithmetic",
    "ratio",
    "stat",
    "similarity",
    "datetime_extract",
}

ARITHMETIC_METHODS = {"add", "subtract", "multiply", "divide"}
STAT_METHODS = {"sum", "mean", "min", "max", "std", "median", "count", "range"}
SIMILARITY_METHODS = {"token_sort_ratio", "token_set_ratio", "ratio"}
ALLOWED_DATETIME_FEATURES = {
    "year",
    "quarter",
    "month",
    "month_name",
    "week",
    "day",
    "day_name",
    "weekday",
    "is_weekend",
    "hour",
    "minute",
    "second",
    "season",
    "time_of_day",
}

DEFAULT_EPSILON = 1e-9

FeatureMathStatus = Literal["applied", "skipped", "failed"]


class FeatureMathOperationError(Exception):
    """Custom error for feature math operations."""

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


@dataclass
class FeatureMathOperation:
    """Normalized feature math operation configuration."""

    operation_id: str
    operation_type: str
    method: Optional[str] = None
    input_columns: List[str] = field(default_factory=list)
    secondary_columns: List[str] = field(default_factory=list)
    constants: List[float] = field(default_factory=list)
    output_column: Optional[str] = None
    output_prefix: Optional[str] = None
    datetime_features: List[str] = field(default_factory=list)
    timezone: Optional[str] = None
    fillna: Optional[float] = None
    round_digits: Optional[int] = None
    normalize: bool = False
    epsilon_override: Optional[float] = None
    allow_overwrite: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureMathConfig:
    """Container for node-level configuration."""

    operations: List[FeatureMathOperation] = field(default_factory=list)
    error_handling: str = "skip"
    epsilon: float = DEFAULT_EPSILON
    default_timezone: Optional[str] = "UTC"
    allow_overwrite: bool = False


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):  # pragma: no cover - defensive
        return None
    return numeric


def _coerce_int(value: Any) -> Optional[int]:
    float_value = _coerce_float(value)
    if float_value is None:
        return None
    try:
        return int(round(float_value))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _normalize_method(operation_type: str, raw_method: Any) -> Optional[str]:
    if raw_method is None:
        return None
    method = str(raw_method).strip().lower()
    if not method:
        return None

    if operation_type == "arithmetic" and method in ARITHMETIC_METHODS:
        return method
    if operation_type == "stat" and method in STAT_METHODS:
        return method
    if operation_type == "similarity" and method in SIMILARITY_METHODS:
        return method
    if operation_type == "ratio":
        return "ratio"
    if operation_type == "datetime_extract":
        return "datetime_extract"
    return None


def _normalize_datetime_features(raw: Any) -> List[str]:
    values = _coerce_string_list(raw)
    normalized = []
    for item in values:
        feature = item.lower()
        if feature in ALLOWED_DATETIME_FEATURES and feature not in normalized:
            normalized.append(feature)
    return normalized


def _normalize_constants(raw: Any) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, (int, float)):
        coerced = _coerce_float(raw)
        return [coerced] if coerced is not None else []
    if isinstance(raw, str):
        try:
            return [float(segment) for segment in raw.split(",") if segment.strip()]
        except ValueError:
            return []
    constants: List[float] = []
    if isinstance(raw, Iterable):
        for entry in raw:
            coerced = _coerce_float(entry)
            if coerced is not None:
                constants.append(coerced)
    return constants


def _normalize_feature_math_config(raw_config: Any) -> FeatureMathConfig:
    if not isinstance(raw_config, dict):
        raw_config = {}

    error_handling_raw = str(raw_config.get("error_handling") or "skip").strip().lower()
    error_handling = "fail" if error_handling_raw == "fail" else "skip"

    epsilon_value = _coerce_float(raw_config.get("epsilon")) or DEFAULT_EPSILON
    default_timezone = str(raw_config.get("default_timezone") or raw_config.get("timezone") or "UTC").strip() or None
    allow_overwrite = _coerce_config_boolean(raw_config.get("allow_overwrite"), default=False)

    operations_payload = raw_config.get("operations")
    if not isinstance(operations_payload, list):
        operations_payload = []

    normalized_operations: List[FeatureMathOperation] = []

    for index, raw_operation in enumerate(operations_payload):
        if not isinstance(raw_operation, dict):
            continue

        operation_type = str(
            raw_operation.get("operation_type")
            or raw_operation.get("type")
            or "arithmetic"
        ).strip().lower()
        if operation_type not in FEATURE_MATH_ALLOWED_TYPES:
            continue

        operation_id = str(raw_operation.get("id") or raw_operation.get("operation_id") or f"op_{index + 1}")

        method = _normalize_method(operation_type, raw_operation.get("method"))
        input_columns = _coerce_string_list(raw_operation.get("input_columns"))
        secondary_columns = _coerce_string_list(raw_operation.get("secondary_columns"))
        constants = _normalize_constants(raw_operation.get("constants"))
        output_column = str(raw_operation.get("output_column") or "").strip() or None
        output_prefix = str(raw_operation.get("output_prefix") or "").strip() or None
        datetime_features = _normalize_datetime_features(raw_operation.get("datetime_features"))
        timezone = str(raw_operation.get("timezone") or "").strip() or None
        fillna_value = _coerce_float(raw_operation.get("fillna"))
        round_digits = _coerce_int(raw_operation.get("round") or raw_operation.get("round_digits"))
        normalize_similarity = _coerce_config_boolean(raw_operation.get("normalize"), default=False)
        epsilon_override = _coerce_float(raw_operation.get("epsilon"))
        allow_overwrite_op = raw_operation.get("allow_overwrite")
        metadata_raw = raw_operation.get("metadata")
        metadata: Dict[str, Any] = {}
        if isinstance(metadata_raw, dict):
            metadata = {str(key): value for key, value in metadata_raw.items()}

        normalized_operations.append(
            FeatureMathOperation(
                operation_id=operation_id,
                operation_type=operation_type,
                method=method,
                input_columns=input_columns,
                secondary_columns=secondary_columns,
                constants=constants,
                output_column=output_column,
                output_prefix=output_prefix,
                datetime_features=datetime_features,
                timezone=timezone,
                fillna=fillna_value,
                round_digits=round_digits,
                normalize=normalize_similarity,
                epsilon_override=epsilon_override,
                allow_overwrite=allow_overwrite_op if isinstance(allow_overwrite_op, bool) else None,
                metadata=metadata,
            )
        )

    return FeatureMathConfig(
        operations=normalized_operations,
        error_handling=error_handling,
        epsilon=epsilon_value,
        default_timezone=default_timezone,
        allow_overwrite=allow_overwrite,
    )


def _ensure_columns_present(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureMathOperationError(f"Missing required column(s): {', '.join(missing)}")


def _prepare_numeric_series(
    frame: pd.DataFrame,
    column: str,
    *,
    fill_value: Optional[float] = None,
) -> pd.Series:
    series = pd.to_numeric(frame[column], errors="coerce")
    if fill_value is not None:
        series = series.fillna(fill_value)
    return series


def _apply_rounding(series: pd.Series, digits: Optional[int]) -> pd.Series:
    if digits is None:
        return series
    try:
        return series.round(int(digits))
    except Exception:  # pragma: no cover - defensive
        return series


def _resolve_output_name(
    frame: pd.DataFrame,
    op: FeatureMathOperation,
    *,
    fallback: str,
    allow_overwrite: bool,
) -> str:
    candidate = (op.output_column or fallback).strip()
    if not candidate:
        candidate = fallback

    should_overwrite = op.allow_overwrite if op.allow_overwrite is not None else allow_overwrite

    if should_overwrite:
        return candidate

    if candidate not in frame.columns:
        return candidate

    raise FeatureMathOperationError(
        f"Output column '{candidate}' already exists. Enable overwrite to replace it."
    )


def _placeholder_base_name(op: FeatureMathOperation) -> str:
    if op.output_column:
        return op.output_column
    base_parts = [op.operation_type, op.operation_id]
    candidate = "_".join(part for part in base_parts if part).strip("_")
    return candidate or "feature_math_placeholder"


def _infer_placeholder_value(op: FeatureMathOperation) -> Any:
    if op.fillna is not None:
        return op.fillna

    metadata = op.metadata or {}

    if "placeholder_value" in metadata:
        return metadata["placeholder_value"]

    declared_kind = str(
        metadata.get("output_kind")
        or metadata.get("output_type")
        or metadata.get("value_type")
        or ""
    ).strip().lower()

    if declared_kind in {"text", "string", "category", "categorical"}:
        return ""

    if op.operation_type == "similarity":
        return 0.0

    return None


def _create_placeholder_column(
    frame: pd.DataFrame,
    op: FeatureMathOperation,
    config: FeatureMathConfig,
) -> Optional[str]:
    """Create a placeholder output column when an operation is skipped."""

    value = _infer_placeholder_value(op)
    if value is None:
        return None

    fallback = _placeholder_base_name(op)

    try:
        output_name = _resolve_output_name(
            frame,
            op,
            fallback=fallback,
            allow_overwrite=config.allow_overwrite,
        )
    except FeatureMathOperationError:
        base = fallback or "feature_math_placeholder"
        candidate = base
        suffix = 1
        while candidate in frame.columns:
            candidate = f"{base}_{suffix}"
            suffix += 1
        output_name = candidate

    frame[output_name] = value
    return output_name


def _safe_divide(numerator: pd.Series, denominator: pd.Series, epsilon: float) -> pd.Series:
    adjusted = denominator.copy()
    adjusted = adjusted.replace({0: epsilon, -0.0: epsilon})
    adjusted = adjusted.fillna(epsilon)
    adjusted = adjusted.mask(adjusted.abs() < epsilon, epsilon)
    return numerator / adjusted


def _apply_arithmetic_operation(
    frame: pd.DataFrame,
    op: FeatureMathOperation,
    config: FeatureMathConfig,
) -> Tuple[List[str], str]:
    if not op.input_columns and not op.constants:
        raise FeatureMathOperationError("Arithmetic operations require at least one column or constant.")

    _ensure_columns_present(frame, op.input_columns)

    series_list = [
        _prepare_numeric_series(frame, column, fill_value=op.fillna)
        for column in op.input_columns
    ]

    constants = list(op.constants)

    result: pd.Series
    method = op.method or "add"

    if method == "add":
        result = pd.Series(0.0, index=frame.index)
        for series in series_list:
            result = result.add(series, fill_value=0.0)
        for constant in constants:
            result = result + constant
    elif method == "subtract":
        if series_list:
            result = series_list[0].copy()
            operands = series_list[1:]
        else:
            result = pd.Series(0.0, index=frame.index)
            operands = []
        for series in operands:
            result = result.subtract(series, fill_value=0.0)
        for constant in constants:
            result = result - constant
    elif method == "multiply":
        result = pd.Series(1.0, index=frame.index)
        for series in series_list:
            result = result.multiply(series, fill_value=1.0)
        for constant in constants:
            result = result * constant
    elif method == "divide":
        epsilon = op.epsilon_override or config.epsilon or DEFAULT_EPSILON
        if series_list:
            result = series_list[0].copy()
            operands = series_list[1:]
        else:
            if not constants:
                raise FeatureMathOperationError("Division requires at least one numeric operand.")
            initial_constant = constants[0]
            result = pd.Series(initial_constant, index=frame.index, dtype=float)
            operands = []
            constants = constants[1:]
        for series in operands:
            result = _safe_divide(result, series, epsilon)
        for constant in constants:
            denominator = pd.Series(constant, index=result.index, dtype=float)
            result = _safe_divide(result, denominator, epsilon)
    else:  # pragma: no cover - defensive
        raise FeatureMathOperationError(f"Unsupported arithmetic method '{method}'")

    result = _apply_rounding(result, op.round_digits)
    if op.fillna is not None:
        result = result.fillna(op.fillna)

    fallback_name = f"{method}_{'_'.join(op.input_columns[:2])}".strip("_") or "arithmetic_result"
    output_name = _resolve_output_name(frame, op, fallback=fallback_name, allow_overwrite=config.allow_overwrite)
    frame[output_name] = result
    return [output_name], f"Applied {method} operation -> {output_name}"


def _apply_ratio_operation(
    frame: pd.DataFrame,
    op: FeatureMathOperation,
    config: FeatureMathConfig,
) -> Tuple[List[str], str]:
    all_columns = list(dict.fromkeys(op.input_columns + op.secondary_columns))
    if len(all_columns) < 2:
        raise FeatureMathOperationError("Ratio requires numerator and denominator columns.")

    _ensure_columns_present(frame, all_columns)

    numerator_columns = op.input_columns or all_columns[:1]
    denominator_columns = op.secondary_columns or all_columns[1:]

    numerator = pd.Series(0.0, index=frame.index)
    for column in numerator_columns:
        numerator = numerator.add(
            _prepare_numeric_series(frame, column, fill_value=op.fillna),
            fill_value=0.0,
        )

    denominator = pd.Series(0.0, index=frame.index)
    for column in denominator_columns:
        denominator = denominator.add(
            _prepare_numeric_series(frame, column, fill_value=op.fillna),
            fill_value=0.0,
        )

    epsilon = op.epsilon_override or config.epsilon or DEFAULT_EPSILON
    result = _safe_divide(numerator, denominator, epsilon)
    result = _apply_rounding(result, op.round_digits)
    if op.fillna is not None:
        result = result.fillna(op.fillna)

    numerator_label = "+".join(numerator_columns[:2])
    denominator_label = "+".join(denominator_columns[:2])
    fallback_name = f"ratio_{numerator_label}_to_{denominator_label}".strip("_")
    output_name = _resolve_output_name(frame, op, fallback=fallback_name, allow_overwrite=config.allow_overwrite)
    frame[output_name] = result
    return [output_name], f"Computed ratio {output_name}"


def _apply_stat_operation(
    frame: pd.DataFrame,
    op: FeatureMathOperation,
    config: FeatureMathConfig,
) -> Tuple[List[str], str]:
    if not op.input_columns:
        raise FeatureMathOperationError("Statistical operations require at least one input column.")

    _ensure_columns_present(frame, op.input_columns)
    method = op.method or "sum"

    values = [
        _prepare_numeric_series(frame, column, fill_value=op.fillna)
        for column in op.input_columns
    ]

    if method == "sum":
        result = sum(values)
    elif method == "mean":
        result = sum(values) / max(len(values), 1)
    elif method == "min":
        result = pd.concat(values, axis=1).min(axis=1, skipna=True)
    elif method == "max":
        result = pd.concat(values, axis=1).max(axis=1, skipna=True)
    elif method == "std":
        result = pd.concat(values, axis=1).std(axis=1, ddof=0)
    elif method == "median":
        result = pd.concat(values, axis=1).median(axis=1)
    elif method == "count":
        result = pd.concat(values, axis=1).count(axis=1)
    elif method == "range":
        temp = pd.concat(values, axis=1)
        result = temp.max(axis=1, skipna=True) - temp.min(axis=1, skipna=True)
    else:  # pragma: no cover - defensive
        raise FeatureMathOperationError(f"Unsupported stat method '{method}'")

    result = _apply_rounding(result, op.round_digits)
    if op.fillna is not None:
        result = result.fillna(op.fillna)

    fallback_name = f"{method}_{'_'.join(op.input_columns[:3])}".strip("_") or "stat_feature"
    output_name = _resolve_output_name(frame, op, fallback=fallback_name, allow_overwrite=config.allow_overwrite)
    frame[output_name] = result
    return [output_name], f"Computed {method} across columns -> {output_name}"


def _compute_similarity_score(a: Any, b: Any, method: str) -> float:
    text_a = "" if a is None else str(a)
    text_b = "" if b is None else str(b)
    if not text_a and not text_b:
        return 100.0
    if not text_a or not text_b:
        return 0.0

    if _HAS_RAPIDFUZZ:
        if method == "token_sort_ratio":
            return float(fuzz.token_sort_ratio(text_a, text_b))
        if method == "token_set_ratio":
            return float(fuzz.token_set_ratio(text_a, text_b))
        return float(fuzz.ratio(text_a, text_b))

    # Fallback using difflib ratio scaled to percentage
    return SequenceMatcher(None, text_a, text_b).ratio() * 100.0


def _apply_similarity_operation(
    frame: pd.DataFrame,
    op: FeatureMathOperation,
    config: FeatureMathConfig,
) -> Tuple[List[str], str]:
    input_columns = [column for column in op.input_columns if column]
    secondary_columns = [column for column in op.secondary_columns if column]
    configured_secondary = len(op.input_columns) > 1 or bool(op.secondary_columns)

    if not input_columns:
        if secondary_columns:
            input_columns.append(secondary_columns.pop(0))
        else:
            raise FeatureMathOperationError("Similarity operations require at least one input column.")

    primary = input_columns[0]

    candidate_pool: List[str] = []
    if len(input_columns) > 1:
        candidate_pool.extend(input_columns[1:])
    candidate_pool.extend(secondary_columns)

    secondary: Optional[str] = None
    for candidate in candidate_pool:
        if candidate:
            secondary = candidate
            break

    if secondary is None:
        if configured_secondary:
            raise FeatureMathOperationError("Similarity operations require two input columns.")
        secondary = primary

    required_columns = [primary]
    if secondary and secondary != primary:
        required_columns.append(secondary)

    _ensure_columns_present(frame, required_columns)

    method = op.method or "token_sort_ratio"

    series_a = frame[primary]
    series_b = frame[secondary]

    scores = series_a.combine(
        series_b,
        lambda a, b: _compute_similarity_score(a, b, method),
    )
    if op.normalize:
        scores = scores / 100.0

    scores = _apply_rounding(scores, op.round_digits)
    if op.fillna is not None:
        scores = scores.fillna(op.fillna)

    fallback_name = f"similarity_{primary}_{secondary}"
    output_name = _resolve_output_name(frame, op, fallback=fallback_name, allow_overwrite=config.allow_overwrite)
    frame[output_name] = scores
    return [output_name], f"Computed {method} similarity -> {output_name}"


def _determine_season(month: int) -> Optional[str]:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return None


def _resolve_time_of_day(hour: Optional[int]) -> Optional[str]:
    if hour is None:
        return None
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _coerce_datetime_series(raw_series: pd.Series, timezone: Optional[str]) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(raw_series):
        series = raw_series.copy()
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
            series = pd.to_datetime(raw_series, errors="coerce", utc=False, cache=True)

    if not timezone:
        return series

    try:
        if series.dt.tz is None:
            series = series.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
        else:
            series = series.dt.tz_convert(timezone)
    except Exception:  # pragma: no cover - timezone issues are logged for diagnostics
        logger.debug("Unable to adjust timezone %s for datetime column", timezone, exc_info=True)
    return series


def _resolve_datetime_output_name(
    frame: pd.DataFrame,
    op: FeatureMathOperation,
    column: str,
    feature: str,
    config: FeatureMathConfig,
) -> str:
    fallback_prefix = op.output_prefix or f"{column}_"
    base_name = f"{fallback_prefix}{feature}"

    should_overwrite = op.allow_overwrite if op.allow_overwrite is not None else config.allow_overwrite
    candidate = base_name
    if not should_overwrite and candidate in frame.columns:
        alt_candidate = (
            f"{(op.output_prefix or '')}{column}_{feature}"
            if op.output_prefix
            else f"{column}_{feature}"
        )
        candidate = alt_candidate
        suffix = 2
        while candidate in frame.columns:
            candidate = f"{alt_candidate}_{suffix}"
            suffix += 1

    return _resolve_output_name(frame, op, fallback=candidate, allow_overwrite=config.allow_overwrite)


def _compute_datetime_feature(series: pd.Series, feature: str) -> Optional[pd.Series]:
    accessor = series.dt

    if feature == "year":
        return accessor.year
    if feature == "quarter":
        return accessor.quarter
    if feature == "month":
        return accessor.month
    if feature == "month_name":
        return accessor.month_name()
    if feature == "week":
        try:
            return accessor.isocalendar().week
        except AttributeError:  # pragma: no cover - for older pandas versions
            return accessor.week
    if feature == "day":
        return accessor.day
    if feature == "day_name":
        return accessor.day_name()
    if feature == "weekday":
        return accessor.dayofweek
    if feature == "is_weekend":
        return accessor.dayofweek >= 5
    if feature == "hour":
        return accessor.hour
    if feature == "minute":
        return accessor.minute
    if feature == "second":
        return accessor.second
    if feature == "season":
        return accessor.month.apply(lambda m: _determine_season(m) if pd.notna(m) else None)
    if feature == "time_of_day":
        return accessor.hour.apply(lambda h: _resolve_time_of_day(h) if pd.notna(h) else None)

    return None


@dataclass
class _DatetimeOperationContext:
    op: FeatureMathOperation
    config: FeatureMathConfig
    features: List[str]
    timezone: Optional[str]
    fillna: Optional[Any]


def _prepare_datetime_operation_context(
    frame: pd.DataFrame,
    op: FeatureMathOperation,
    config: FeatureMathConfig,
) -> _DatetimeOperationContext:
    if not op.input_columns:
        raise FeatureMathOperationError("Datetime extraction requires at least one input column.")

    _ensure_columns_present(frame, op.input_columns)

    features = op.datetime_features or ["year", "month", "day"]
    timezone = op.timezone or config.default_timezone

    return _DatetimeOperationContext(
        op=op,
        config=config,
        features=features,
        timezone=timezone,
        fillna=op.fillna,
    )


def _apply_datetime_features_for_column(
    frame: pd.DataFrame,
    column: str,
    series: pd.Series,
    context: _DatetimeOperationContext,
) -> List[str]:
    created: List[str] = []

    for feature in context.features:
        output_name = _resolve_datetime_output_name(frame, context.op, column, feature, context.config)
        values = _compute_datetime_feature(series, feature)
        if values is None:  # pragma: no cover - unexpected feature guard
            continue
        if context.fillna is not None:
            values = values.fillna(context.fillna)
        frame[output_name] = values
        created.append(output_name)

    return created


def _apply_datetime_operation(
    frame: pd.DataFrame,
    op: FeatureMathOperation,
    config: FeatureMathConfig,
) -> Tuple[List[str], str]:
    context = _prepare_datetime_operation_context(frame, op, config)

    created_columns: List[str] = []

    for column in op.input_columns:
        series = _coerce_datetime_series(frame[column], context.timezone)
        created_columns.extend(
            _apply_datetime_features_for_column(frame, column, series, context)
        )

    message = f"Extracted datetime features ({', '.join(context.features)})"
    return created_columns, message


def _execute_operation(
    frame: pd.DataFrame,
    op: FeatureMathOperation,
    config: FeatureMathConfig,
) -> Tuple[List[str], str]:
    if op.operation_type == "arithmetic":
        return _apply_arithmetic_operation(frame, op, config)
    if op.operation_type == "ratio":
        return _apply_ratio_operation(frame, op, config)
    if op.operation_type == "stat":
        return _apply_stat_operation(frame, op, config)
    if op.operation_type == "similarity":
        return _apply_similarity_operation(frame, op, config)
    if op.operation_type == "datetime_extract":
        return _apply_datetime_operation(frame, op, config)
    raise FeatureMathOperationError(f"Unsupported operation type '{op.operation_type}'", fatal=True)


def apply_feature_math(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,  # noqa: ARG001 - pipeline_id reserved for future use
) -> Tuple[pd.DataFrame, str, FeatureMathNodeSignal]:
    """Apply feature math operations to the provided frame."""

    node_id = str(node.get("id") or "")
    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_feature_math_config(config_payload)

    signal = FeatureMathNodeSignal(
        node_id=node_id or None,
        generated_at=utcnow(),
    )

    if not config.operations:
        summary = "Feature math: no operations configured"
        signal.total_operations = 0
        return frame, summary, signal

    working_frame = frame.copy()
    results: List[FeatureMathOperationResult] = []
    success_count = 0
    skipped_count = 0
    failed_count = 0
    warnings: List[str] = []

    for operation in config.operations:
        try:
            created_columns, message = _execute_operation(working_frame, operation, config)
        except FeatureMathOperationError as exc:
            if config.error_handling == "fail" or exc.fatal:
                status: FeatureMathStatus = "failed"
            else:
                status = "skipped"
            placeholder_output: Optional[str] = None

            if status == "failed":
                failed_count += 1
            else:
                skipped_count += 1
                placeholder_output = _create_placeholder_column(working_frame, operation, config)

            warning_message = f"Operation {operation.operation_id}: {exc}"
            if placeholder_output:
                warning_message = (
                    f"{warning_message} | Placeholder column '{placeholder_output}' filled with default value."
                )
            warnings.append(warning_message)
            log_message = f"Feature math operation skipped: {warning_message}"
            if status == "failed":
                logger.warning(log_message)
            else:
                logger.debug(log_message)
            results.append(
                FeatureMathOperationResult(
                    operation_id=operation.operation_id,
                    operation_type=operation.operation_type,
                    method=operation.method,
                    output_columns=[placeholder_output] if placeholder_output else [],
                    status=status,
                    message=(
                        str(exc)
                        if not placeholder_output
                        else f"{exc} | Placeholder column '{placeholder_output}' filled with default value."
                    ),
                )
            )
            if status == "failed":
                raise
            continue
        except Exception as exc:  # pragma: no cover - defensive
            failed_count += 1
            warning_message = f"Operation {operation.operation_id} failed: {exc}"
            warnings.append(warning_message)
            logger.exception("Unhandled feature math error: %s", warning_message)
            results.append(
                FeatureMathOperationResult(
                    operation_id=operation.operation_id,
                    operation_type=operation.operation_type,
                    method=operation.method,
                    output_columns=[],
                    status="failed",
                    message=str(exc),
                )
            )
            if config.error_handling == "fail":
                raise
            continue

        success_count += 1
        results.append(
            FeatureMathOperationResult(
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                method=operation.method,
                output_columns=list(created_columns),
                status="applied",
                message=message,
            )
        )

    summary = (
        f"Feature math: {success_count} applied"
        + (f", {skipped_count} skipped" if skipped_count else "")
        + (f", {failed_count} failed" if failed_count else "")
    )

    signal.total_operations = len(config.operations)
    signal.applied_operations = success_count
    signal.skipped_operations = skipped_count
    signal.failed_operations = failed_count
    signal.operations = results
    signal.warnings = warnings

    return working_frame, summary, signal
