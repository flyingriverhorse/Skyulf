"""Column casting helpers for feature engineering nodes."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .utils import _coerce_boolean_value, _coerce_config_boolean
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
            except Exception:
                return converted
        return converted

    if dtype_family == "string":
        return series.astype("string")

    if dtype_family == "category":
        return series.astype("category")

    return series.astype(target_dtype)


def _apply_cast_column_types(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, CastColumnTypesNodeSignal]:
    if frame.empty:
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

        candidate_columns: List[str] = []
        seen_columns: Set[str] = set()
        for column in configured_columns:
            if column and column not in seen_columns:
                candidate_columns.append(column)
                seen_columns.add(column)
        for column in column_overrides.keys():
            if column and column not in seen_columns:
                candidate_columns.append(column)
                seen_columns.add(column)

        coerce_on_error = _coerce_config_boolean(config.get("coerce_on_error"), default=True)

        signal = CastColumnTypesNodeSignal(
            node_id=str(node_id) if node_id is not None else None,
            configured_columns=list(configured_columns),
            column_overrides=dict(column_overrides),
            candidate_columns=list(candidate_columns),
            coerce_on_error=coerce_on_error,
        )
        return frame, "Cast columns: no data available", signal

    data = node.get("data") or {}
    config = data.get("config") or {}

    node_id = node.get("id") if isinstance(node, dict) else None

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

    target_dtype_raw = config.get("target_dtype") or config.get("dtype") or ""
    default_target_dtype = str(target_dtype_raw).strip()

    candidate_columns: List[str] = []
    seen_columns: Set[str] = set()
    for column in configured_columns:
        if column and column not in seen_columns:
            candidate_columns.append(column)
            seen_columns.add(column)
    for column in column_overrides.keys():
        if column and column not in seen_columns:
            candidate_columns.append(column)
            seen_columns.add(column)

    coerce_on_error = _coerce_config_boolean(config.get("coerce_on_error"), default=True)

    signal = CastColumnTypesNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        configured_columns=list(configured_columns),
        column_overrides=dict(column_overrides),
        candidate_columns=list(candidate_columns),
        coerce_on_error=coerce_on_error,
    )

    if not candidate_columns:
        return frame, "Cast columns: no matching columns", signal

    missing_columns: List[str] = []
    skipped_missing_dtype: List[str] = []
    instructions: List[Tuple[str, str, str, str]] = []

    for column in candidate_columns:
        if column not in frame.columns:
            missing_columns.append(column)
            continue

        dtype_candidate = column_overrides.get(column, default_target_dtype)
        dtype_value = str(dtype_candidate).strip() if dtype_candidate is not None else ""
        if not dtype_value:
            skipped_missing_dtype.append(column)
            continue

        normalized_key = dtype_value.lower()
        normalized_dtype, dtype_family = COLUMN_CAST_ALIASES.get(normalized_key, (dtype_value, "custom"))
        instructions.append((column, normalized_dtype, dtype_family, dtype_value))

    if not instructions:
        if missing_columns and not configured_columns:
            signal.missing_columns = sorted(set(missing_columns))
            return frame, "Cast columns: no matching columns", signal
        signal.skipped_missing_dtype = sorted(set(skipped_missing_dtype))
        return frame, "Cast columns: target dtype not configured", signal

    working_frame = frame.copy()
    changed_columns: List[str] = []
    errors: Dict[str, str] = {}
    coerced_values = 0
    attempt_records: List[CastColumnAttemptSignal] = []

    for column, normalized_dtype, dtype_family, requested_dtype in instructions:
        original_series = working_frame[column].copy()
        before_missing = int(original_series.isna().sum())

        attempt = CastColumnAttemptSignal(
            column=str(column),
            original_dtype=str(original_series.dtype),
            requested_dtype=str(requested_dtype) if requested_dtype is not None else None,
            resolved_dtype=normalized_dtype,
            dtype_family=dtype_family,
        )

        try:
            converted_series = _cast_series_to_dtype(original_series, dtype_family, normalized_dtype, coerce_on_error)
        except Exception as exc:
            errors[column] = str(exc)
            attempt.error = str(exc)
            attempt_records.append(attempt)
            continue

        after_missing = int(converted_series.isna().sum())
        if after_missing > before_missing:
            coerced_values += after_missing - before_missing
            attempt.values_coerced_to_missing = after_missing - before_missing

        working_frame[column] = converted_series

        changed = str(converted_series.dtype) != str(original_series.dtype) or not converted_series.equals(original_series)
        if changed:
            changed_columns.append(column)
        attempt.changed_dtype = changed
        attempt_records.append(attempt)

    attempted = len(instructions)
    unique_targets = sorted({instruction[1] for instruction in instructions})
    if len(unique_targets) == 1:
        summary_parts = [f"Cast columns: attempted {attempted} column(s) to {unique_targets[0]}"]
    else:
        targets_preview = ", ".join(unique_targets[:3])
        if len(unique_targets) > 3:
            targets_preview += ", ..."
        summary_parts = [
            f"Cast columns: attempted {attempted} column(s) across {len(unique_targets)} dtype(s): {targets_preview}"
        ]

    if changed_columns:
        summary_parts.append(f"updated {len(changed_columns)} column(s)")
    else:
        summary_parts.append("no dtype changes detected")

    if coerce_on_error:
        summary_parts.append("coerce enabled")
        if coerced_values:
            summary_parts.append(f"coerced {coerced_values} value(s) to missing")

    if errors:
        preview = ", ".join(list(errors.keys())[:3])
        if len(errors) > 3:
            preview = f"{preview}, ..."
        summary_parts.append(f"{len(errors)} column(s) skipped ({preview})")

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

    signal.missing_columns = sorted(set(missing_columns))
    signal.skipped_missing_dtype = sorted(set(skipped_missing_dtype))
    signal.errors = errors
    signal.coerced_values = coerced_values
    signal.attempted_columns = attempt_records
    signal.applied_columns = [str(column) for column in changed_columns]
    signal.candidate_columns = list(candidate_columns)

    return working_frame, "; ".join(summary_parts), signal


__all__ = [
    "COLUMN_CAST_ALIASES",
    "_apply_cast_column_types",
    "_cast_series_to_dtype",
]
