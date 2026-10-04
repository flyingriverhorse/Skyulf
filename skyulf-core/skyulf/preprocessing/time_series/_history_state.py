"""Bounded JSON history encoding and strict observation ordering checks."""

import hashlib
import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...utils import unpack_pipeline_input


def history_columns(params: dict[str, Any]) -> list[str]:
    """Project only the observed features and their entity/time keys."""
    return list(
        dict.fromkeys([*(params.get("group_by") or []), params["sort_by"], *params["columns"]])
    )


def native_frame(data: Any) -> Any:
    """Unpack local frames without collecting a distributed engine."""
    frame, _, _ = unpack_pipeline_input(data)
    if hasattr(frame, "to_native"):
        frame = frame.to_native()
    if not isinstance(frame, (pd.DataFrame, pl.DataFrame)):
        raise NotImplementedError("Temporal history supports only bounded pandas/Polars frames.")
    return frame


def records(frame: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Read required columns while retaining timestamp precision and scalar types."""
    columns = history_columns(params)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Temporal history columns are missing: {sorted(missing)}.")
    projected = frame[columns]
    if isinstance(projected, pl.DataFrame):
        # pandas preserves nanosecond timestamps which Polars to_dicts truncates.
        projected = projected.to_pandas()
    _validate_clock(projected[params["sort_by"]])
    return projected.to_dict(orient="records")


def _validate_clock(clock: pd.Series) -> None:
    """Reject ambiguous string ordering; require a numeric sequence or typed timestamps."""
    if pd.api.types.is_bool_dtype(clock.dtype):
        raise ValueError("Temporal history clock cannot be boolean.")
    if pd.api.types.is_numeric_dtype(clock.dtype):
        if not np.isfinite(clock.to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError("Temporal history clock must contain finite, nonmissing values.")
        return
    if not pd.api.types.is_datetime64_any_dtype(clock.dtype):
        raise ValueError("Temporal history clock must be numeric or datetime; parse strings first.")


def _encode(value: Any) -> Any:
    """Encode supported observed scalars without pickle or executable objects."""
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return {"timestamp": pd.Timestamp(value).isoformat()}
    if isinstance(value, date):
        return {"date": value.isoformat()}
    if type(value) not in (str, int, float, bool):
        raise ValueError(
            "Temporal history requires scalar numeric, text, date or timestamp values."
        )
    return value


def decode_rows(rows: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the bounded public state before interpreting its closed scalar format."""
    check_budget(rows, params)
    columns = set(history_columns(params))
    decoded = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != columns:
            raise ValueError("Temporal history row schema does not match the fitted step.")
        decoded.append({key: _decode(value) for key, value in row.items()})
    return decoded


def _decode(value: Any) -> Any:
    """Decode only explicit timestamp/date tags and plain JSON scalar values."""
    if isinstance(value, dict):
        if set(value) == {"timestamp"}:
            return pd.Timestamp(value["timestamp"])
        if set(value) == {"date"}:
            return date.fromisoformat(value["date"])
        raise ValueError("Invalid temporal history scalar tag.")
    if value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("Invalid temporal history scalar.")
    return value


def check_budget(rows: Any, params: dict[str, Any]) -> None:
    """Reject oversized state rather than silently losing an entity's context."""
    if not isinstance(rows, list) or len(rows) > params["history_max_rows"]:
        raise ValueError("Temporal history exceeds history_max_rows or has invalid rows.")
    if len(json.dumps(rows, allow_nan=False).encode()) > params["history_max_bytes"]:
        raise ValueError("Temporal history exceeds history_max_bytes.")


def entity(row: dict[str, Any], params: dict[str, Any]) -> tuple[Any, ...]:
    """Use the complete configured entity key; ungrouped data forms one sequence."""
    return tuple(row[key] for key in params.get("group_by") or [])


def ordered_groups(rows: list[dict[str, Any]], params: dict[str, Any]) -> dict[Any, list]:
    """Reject missing or ambiguous temporal keys before sorting each entity."""
    groups: dict[Any, list] = {}
    keys = [*(params.get("group_by") or []), params["sort_by"]]
    seen = set()
    for row in rows:
        identity = tuple(row[key] for key in keys)
        if any(pd.isna(value) for value in identity) or identity in seen:
            raise ValueError("Temporal history needs nonmissing, unique entity/time keys.")
        seen.add(identity)
        groups.setdefault(entity(row, params), []).append(row)
    return {
        key: sorted(values, key=lambda row: row[params["sort_by"]])
        for key, values in groups.items()
    }


def bounded_tail(rows: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Retain only required prior observations per entity, including inactive entities."""
    groups = ordered_groups(rows, params)
    count = params["history_rows"]
    tail = [row for values in groups.values() for row in values[-count:]]
    encoded = [{key: _encode(value) for key, value in row.items()} for row in tail]
    check_budget(encoded, params)
    return encoded


def fit_history(data: Any, config: dict[str, Any], params: dict[str, Any], count: int) -> dict:
    """Attach immutable bounded training context only when explicitly requested."""
    mode = config.get("history_mode", "batch")
    if mode == "batch":
        return params
    if mode != "carry":
        raise ValueError("history_mode must be batch or carry.")
    _validate_config(config, count)
    _, target, _ = unpack_pipeline_input(data)
    target_name = config.get("target_column") or getattr(target, "name", None)
    if target_name in config["columns"]:
        raise ValueError("carry history requires observed features, not the prediction target.")
    params = params | {
        "history_mode": mode,
        "history_rows": max(1, count),
        "history_max_rows": config.get("history_max_rows", 10000),
        "history_max_bytes": config.get("history_max_bytes", 1048576),
    }
    params["history_seed"] = bounded_tail(records(native_frame(data), params), params)
    params["history_id"] = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    return params


def _validate_config(config: dict[str, Any], count: int) -> None:
    """Require an explicit causal order and bounded, row-preserving behavior."""
    if not isinstance(config.get("sort_by"), str) or not config["sort_by"]:
        raise ValueError("carry history requires an explicit sort_by time column.")
    if not config.get("columns") or count < 0:
        raise ValueError("carry history requires feature columns and a valid window.")
    if config.get("drop_na"):
        raise ValueError("carry history does not support drop_na; impute missing lag features.")
    for key, default in (("history_max_rows", 10000), ("history_max_bytes", 1048576)):
        value = config.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer.")
