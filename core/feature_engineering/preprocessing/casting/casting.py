"""Column casting helpers for preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ...shared.utils import _coerce_boolean_value, _coerce_config_boolean
from core.feature_engineering.schemas import CastColumnAttemptSignal, CastColumnTypesNodeSignal

COLUMN_CAST_ALIASES: Dict[str, Tuple[str, str]] = {
    "float": ("float64", "float"),
    "float32": ("float32", "float"),
    "float64": ("float64", "float"),
    "double": ("float64", "float"),
    "numeric": ("float64", "float"),
    "int": ("int64", "int"),
    "int32": ("int32", "int"),
    "int64": ("int64", "int"),
    "integer": ("int64", "int"),
    "string": ("string", "string"),
    "str": ("string", "string"),
    "text": ("string", "string"),
    "category": ("category", "category"),
    "categorical": ("category", "category"),
    "bool": ("boolean", "boolean"),
    "boolean": ("boolean", "boolean"),
    "datetime": ("datetime64[ns]", "datetime"),
    "datetime64": ("datetime64[ns]", "datetime"),
    "datetime64[ns]": ("datetime64[ns]", "datetime"),
    "date": ("datetime64[ns]", "datetime"),
}


@dataclass
class CastConfiguration:
    node_id: Optional[str]
    configured_columns: List[str]
    column_overrides: Dict[str, str]
    candidate_columns: List[str]
    default_target_dtype: str
    coerce_on_error: bool


@dataclass(frozen=True)
class CastInstruction:
    column: str
    resolved_dtype: str
    dtype_family: str
    requested_dtype: str


def _parse_cast_config(node: Dict[str, Any]) -> CastConfiguration:
    node_id = node.get("id") if isinstance(node, dict) else None
    data = node.get("data") or {}
    config = data.get("config") or {}

    raw_columns = config.get("columns")
    if isinstance(raw_columns, list):
        configured_columns = [str(column).strip() for column in raw_columns if str(column).strip()]
    elif isinstance(raw_columns, str):
        configured_columns = [segment.strip() for segment in raw_columns.split(",") if segment.strip()]
    else:
        configured_columns = []

    raw_overrides = config.get("column_overrides")
    column_overrides: Dict[str, str] = {}
    if isinstance(raw_overrides, dict):
        for key, value in raw_overrides.items():
            name = str(key).strip()
            if not name:
                continue
            dtype_value = value.strip() if isinstance(value, str) else str(value).strip()
            if not dtype_value:
                continue
            column_overrides[name] = dtype_value

    seen_columns: Set[str] = set()
    candidate_columns: List[str] = []
    for column in configured_columns:
        if column and column not in seen_columns:
            candidate_columns.append(column)
            seen_columns.add(column)
    for column in column_overrides.keys():
        if column and column not in seen_columns:
            candidate_columns.append(column)
            seen_columns.add(column)

    target_dtype_raw = config.get("target_dtype") or config.get("dtype") or ""
    default_target_dtype = str(target_dtype_raw).strip()
    coerce_on_error = _coerce_config_boolean(config.get("coerce_on_error"), default=True)

    return CastConfiguration(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
        column_overrides=dict(column_overrides),
        candidate_columns=list(candidate_columns),
        default_target_dtype=default_target_dtype,
        coerce_on_error=coerce_on_error,
    )


def _build_cast_signal(config: CastConfiguration) -> CastColumnTypesNodeSignal:
    return CastColumnTypesNodeSignal(
        node_id=config.node_id,
        configured_columns=list(config.configured_columns),
        column_overrides=dict(config.column_overrides),
        candidate_columns=list(config.candidate_columns),
        coerce_on_error=config.coerce_on_error,
    )


def _collect_cast_instructions(
    frame: pd.DataFrame,
    config: CastConfiguration,
) -> Tuple[List[CastInstruction], List[str], List[str]]:
    instructions: List[CastInstruction] = []
    missing_columns: List[str] = []
    skipped_missing_dtype: List[str] = []

    for column in config.candidate_columns:
        if column not in frame.columns:
            missing_columns.append(column)
            continue

        dtype_candidate = config.column_overrides.get(column, config.default_target_dtype)
        dtype_value = str(dtype_candidate).strip() if dtype_candidate is not None else ""
        if not dtype_value:
            skipped_missing_dtype.append(column)
            continue

        normalized_key = dtype_value.lower()
        resolved_dtype, dtype_family = COLUMN_CAST_ALIASES.get(normalized_key, (dtype_value, "custom"))
        instructions.append(
            CastInstruction(
                column=column,
                resolved_dtype=resolved_dtype,
                dtype_family=dtype_family,
                requested_dtype=dtype_value,
            )
        )

    return instructions, missing_columns, skipped_missing_dtype


def _cast_series_to_dtype(
    series: pd.Series,
    dtype_family: str,
    target_dtype: str,
    coerce_on_error: bool,
) -> pd.Series:
    if dtype_family == "float":
        numeric = pd.to_numeric(series, errors="coerce" if coerce_on_error else "raise")
        return numeric.astype(target_dtype)

    if dtype_family == "int":
        numeric = pd.to_numeric(series, errors="coerce" if coerce_on_error else "raise")
        if coerce_on_error:
            valid_mask = numeric.notna()
            fractional_mask = valid_mask & ~np.isclose(numeric, np.round(numeric))
            if fractional_mask.any():
                numeric.loc[fractional_mask] = np.nan
            nullable_target = target_dtype
            if nullable_target.lower().startswith("int") and not nullable_target.startswith("Int"):
                nullable_target = f"I{nullable_target[1:]}"
            try:
                return numeric.astype(nullable_target)
            except TypeError:
                return numeric.astype("Int64")
        fractional_mask = numeric.notna() & ~np.isclose(numeric, np.round(numeric))
        if fractional_mask.any():
            raise ValueError("Non-integer values encountered during cast")
        return numeric.astype(target_dtype)

    if dtype_family == "boolean":
        try:
            return series.astype("boolean")
        except (TypeError, ValueError):
            if not coerce_on_error:
                raise
            coerced_values = [
                pd.NA if (result := _coerce_boolean_value(value)) is None else result
                for value in series
            ]
            return pd.Series(coerced_values, index=series.index, dtype="boolean")

    if dtype_family == "datetime":
        converted = pd.to_datetime(series, errors="coerce" if coerce_on_error else "raise")
        if target_dtype != "datetime64[ns]":
            try:
                return converted.astype(target_dtype)
            except Exception:  # pragma: no cover - defensive
                return converted
        return converted

    if dtype_family == "string":
        return series.astype("string")

    if dtype_family == "category":
        return series.astype("category")

    return series.astype(target_dtype)


@dataclass
class CastExecutionResult:
    frame: pd.DataFrame
    changed_columns: List[str]
    errors: Dict[str, str]
    coerced_values: int
    attempts: List[CastColumnAttemptSignal]


@dataclass
class _CastPreparationOutcome:
    frame: pd.DataFrame
    instructions: List[CastInstruction]
    missing_columns: List[str]
    skipped_missing_dtype: List[str]
    summary: Optional[str]


def _execute_cast_instructions(
    frame: pd.DataFrame,
    instructions: List[CastInstruction],
    coerce_on_error: bool,
) -> CastExecutionResult:
    working_frame = frame.copy()
    changed_columns: List[str] = []
    errors: Dict[str, str] = {}
    coerced_values = 0
    attempt_records: List[CastColumnAttemptSignal] = []

    for instruction in instructions:
        original_series = working_frame[instruction.column].copy()
        before_missing = int(original_series.isna().sum())

        attempt = CastColumnAttemptSignal(
            column=str(instruction.column),
            original_dtype=str(original_series.dtype),
            requested_dtype=str(instruction.requested_dtype) if instruction.requested_dtype is not None else None,
            resolved_dtype=instruction.resolved_dtype,
            dtype_family=instruction.dtype_family,
        )

        try:
            converted_series = _cast_series_to_dtype(
                original_series,
                instruction.dtype_family,
                instruction.resolved_dtype,
                coerce_on_error,
            )
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            errors[instruction.column] = error_message
            attempt.error = error_message
            attempt_records.append(attempt)
            continue

        after_missing = int(converted_series.isna().sum())
        if after_missing > before_missing:
            coerced_delta = after_missing - before_missing
            coerced_values += coerced_delta
            attempt.values_coerced_to_missing = coerced_delta

        working_frame[instruction.column] = converted_series

        changed = (
            str(converted_series.dtype) != str(original_series.dtype)
            or not converted_series.equals(original_series)
        )
        if changed:
            changed_columns.append(instruction.column)
        attempt.changed_dtype = changed
        attempt_records.append(attempt)

    return CastExecutionResult(
        frame=working_frame,
        changed_columns=changed_columns,
        errors=errors,
        coerced_values=coerced_values,
        attempts=attempt_records,
    )


def _summarize_cast_results(
    instructions: List[CastInstruction],
    result: CastExecutionResult,
    missing_columns: List[str],
    skipped_missing_dtype: List[str],
    coerce_on_error: bool,
) -> str:
    attempted = len(instructions)
    unique_targets = sorted({instruction.resolved_dtype for instruction in instructions})

    if len(unique_targets) == 1:
        summary_parts = [f"Cast column types: attempted {attempted} column(s) to {unique_targets[0]}"]
    else:
        targets_preview = ", ".join(unique_targets[:3])
        if len(unique_targets) > 3:
            targets_preview = f"{targets_preview}, ..."
        summary_parts = [
            f"Cast column types: attempted {attempted} column(s) across {len(unique_targets)} dtype(s): {targets_preview}"
        ]

    if result.changed_columns:
        summary_parts.append(f"updated {len(result.changed_columns)} column(s)")
    else:
        summary_parts.append("no dtype changes detected")

    if coerce_on_error:
        summary_parts.append("coerce enabled")
        if result.coerced_values:
            summary_parts.append(f"coerced {result.coerced_values} value(s) to missing")

    if result.errors:
        preview = ", ".join(list(result.errors.keys())[:3])
        if len(result.errors) > 3:
            preview = f"{preview}, ..."
        summary_parts.append(f"{len(result.errors)} column(s) skipped ({preview})")

    if skipped_missing_dtype:
        preview = ", ".join(skipped_missing_dtype[:3])
        if len(skipped_missing_dtype) > 3:
            preview = f"{preview}, ..."
        summary_parts.append(f"{len(skipped_missing_dtype)} column(s) missing dtype ({preview})")

    if missing_columns:
        preview = ", ".join(missing_columns[:3])
        if len(missing_columns) > 3:
            preview = f"{preview}, ..."
        summary_parts.append(f"{len(missing_columns)} column(s) not found ({preview})")

    return "; ".join(summary_parts)


def _prepare_cast_execution(
    frame: pd.DataFrame,
    config: CastConfiguration,
) -> _CastPreparationOutcome:
    if frame.empty:
        return _CastPreparationOutcome(frame, [], [], [], "Cast column types: no data available")

    if not config.candidate_columns:
        return _CastPreparationOutcome(frame, [], [], [], "Cast column types: no matching columns")

    instructions, missing_columns, skipped_missing_dtype = _collect_cast_instructions(frame, config)

    if instructions:
        return _CastPreparationOutcome(frame, instructions, missing_columns, skipped_missing_dtype, None)

    summary = "Cast column types: target dtype not configured"
    if missing_columns and not config.configured_columns:
        summary = "Cast column types: no matching columns"
    return _CastPreparationOutcome(frame, [], missing_columns, skipped_missing_dtype, summary)


def _update_signal_with_result(
    signal: CastColumnTypesNodeSignal,
    preparation: _CastPreparationOutcome,
    result: Optional[CastExecutionResult] = None,
) -> None:
    signal.missing_columns = sorted(set(preparation.missing_columns))
    signal.skipped_missing_dtype = sorted(set(preparation.skipped_missing_dtype))

    if result is None:
        return

    signal.errors = result.errors
    signal.coerced_values = result.coerced_values
    signal.attempted_columns = result.attempts
    signal.applied_columns = [str(column) for column in result.changed_columns]


def _apply_cast_column_types(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, CastColumnTypesNodeSignal]:
    config = _parse_cast_config(node)
    signal = _build_cast_signal(config)

    preparation = _prepare_cast_execution(frame, config)

    if preparation.summary is not None:
        _update_signal_with_result(signal, preparation, result=None)
        return preparation.frame, preparation.summary, signal

    result = _execute_cast_instructions(frame, preparation.instructions, config.coerce_on_error)
    summary = _summarize_cast_results(
        preparation.instructions,
        result,
        preparation.missing_columns,
        preparation.skipped_missing_dtype,
        config.coerce_on_error,
    )

    _update_signal_with_result(signal, preparation, result)

    return result.frame, summary, signal
