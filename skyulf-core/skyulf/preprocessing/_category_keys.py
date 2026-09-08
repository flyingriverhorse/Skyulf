"""Stable scalar keys shared by fitted categorical preprocessing nodes."""

import math
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

_PREFIX = "__skyulf_category__:"
_NUMBER_PREFIX = f"{_PREFIX}n:"


def category_key(value: Any) -> str:
    """Normalize numeric and missing scalars without conflating literal strings."""
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return "nan"
    if isinstance(value, bool | np.bool_):
        return f"{_PREFIX}b:{value}"
    if isinstance(value, Integral):
        return f"{_NUMBER_PREFIX}{int(value)}"
    if isinstance(value, Real):
        number = float(value)
        text = str(int(number)) if math.isfinite(number) and number.is_integer() else str(number)
        return f"{_NUMBER_PREFIX}{text}"
    text = str(value)
    return f"{_PREFIX}s:{text}" if text == "nan" or text.startswith(_PREFIX) else text


def category_keys_pandas(series: pd.Series) -> pd.Series:
    """Build keys before any dtype coercion can change scalar identity."""
    return pd.Series(
        [category_key(value) for value in series],
        index=series.index,
        name=series.name,
        dtype=object,
    )


def category_key_expr(column: str) -> pl.Expr:
    """Build the same scalar keys natively around a Polars column."""
    return pl.col(column).map_elements(category_key, return_dtype=pl.String, skip_nulls=False)


def uses_category_keys(params: dict[str, Any]) -> bool:
    """Keep old artifacts on their original string-key replay contract."""
    if "category_key_version" not in params:
        return False
    version = params["category_key_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError(f"Unsupported category key version {version!r}; refit the encoder.")
    return True


def category_order_keys(categories: list[str], observed: Any) -> list[str]:
    """Retain numeric ordering entered as text for an otherwise numeric feature."""
    values = set(observed)
    numeric = any(value.startswith(_NUMBER_PREFIX) for value in values) and all(
        value == "nan" or value.startswith(_NUMBER_PREFIX) for value in values
    )
    if not numeric:
        return [category_key(value) for value in categories]
    result = []
    for value in categories:
        try:
            number: int | float = int(value)
        except ValueError:
            try:
                number = float(value)
            except ValueError:
                result.append(category_key(value))
                continue
        result.append(category_key(number))
    return result
