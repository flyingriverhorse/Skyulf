"""Exact conversion of explicit text constants using each fitted column's scalar type."""

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


def coerce_constant_text(value: str, scalar_type: type) -> Any:
    """Convert explicit text without rounding integer values or treating nonempty text as true."""
    if scalar_type is bool:
        words = {"true": True, "false": False}
        if value.strip().lower() not in words:
            raise ValueError("Boolean constants must be true or false.")
        return words[value.strip().lower()]
    if scalar_type is int:
        number = Decimal(value)
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError("Integer constants must be finite integral values.")
        return int(number)
    if scalar_type is float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Numeric constants must be finite.")
        return number
    return value


def constant_matches(value: Any, fitted: Any) -> bool:
    """Bind a configured scalar to its exact learned constant, including supported text casts."""
    if isinstance(value, str) and type(fitted) in (int, float, bool):
        try:
            return coerce_constant_text(value, type(fitted)) == fitted
        except (ValueError, InvalidOperation, OverflowError):
            return False
    return value == fitted


def _coerce_column(value: str, dtype: Any) -> Any:
    """Respect integer bounds and float range while preserving literal text for text columns."""
    name = str(getattr(dtype, "numpy_dtype", dtype)).lower()
    integer = re.fullmatch(r"(u?)int(8|16|32|64|128)", name)
    if integer:
        result = coerce_constant_text(value, int)
        bits = int(integer.group(2))
        low = 0 if integer.group(1) else -(2 ** (bits - 1))
        high = 2**bits - 1 if integer.group(1) else 2 ** (bits - 1) - 1
        if not low <= result <= high:
            raise ValueError(f"Constant is outside {dtype} bounds.")
        return result
    if name in {"float16", "float32", "float64"}:
        result = coerce_constant_text(value, float)
        if abs(result) > np.finfo(name).max:
            raise ValueError(f"Constant is outside {dtype} bounds.")
        return result
    if name in {"bool", "boolean"}:
        return coerce_constant_text(value, bool)
    if pd.api.types.is_datetime64_any_dtype(dtype) or isinstance(dtype, pl.Datetime):
        raise ValueError("Text constants for datetime columns are unsupported.")
    return value


def constant_text_values(frame: Any, columns: list[str], value: str) -> dict[str, Any]:
    """Validate and resolve every selected column before returning any fitted fill values."""
    values = {}
    for column in columns:
        try:
            values[column] = _coerce_column(value, frame[column].dtype)
        except (ValueError, TypeError, InvalidOperation, OverflowError) as error:
            raise ValueError(
                f"SimpleImputer fill_value {value!r} is invalid for column {column!r}: {error}"
            ) from error
    return values
