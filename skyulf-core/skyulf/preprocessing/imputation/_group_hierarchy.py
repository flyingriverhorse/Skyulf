"""Versioned ordered group levels for the opt-in multi-key imputer contract."""

from typing import Any, cast

import numpy as np
import pandas as pd

from .._artifacts import GroupImputerArtifact
from .._category_keys import category_key
from .._fitted_validation import _columns, _fields, _scalar


def group_statistic(values: pd.Series, strategy: str) -> Any:
    """Return a training statistic without losing exact scalar modes."""
    values = values.dropna()
    if values.empty:
        return None
    if strategy == "mean":
        return float(values.mean())
    if strategy == "median":
        return float(values.median())
    value = values.mode().iloc[0]
    return value.item() if isinstance(value, np.generic) else value


def group_columns(value: Any) -> list[str]:
    """Require ordered, distinct, nonempty key names for the new list-based contract."""
    if (
        type(value) is not list
        or not value
        or any(type(key) is not str or not key for key in value)
    ):
        raise ValueError("GroupImputer group_by must be a nonempty ordered list of column names.")
    if len(set(value)) != len(value):
        raise ValueError("GroupImputer group_by columns must be unique.")
    return list(cast(list[str], value))


def group_fallback(value: Any) -> str:
    """Make the order of fallback levels explicit rather than guessing from key count."""
    if value not in ("global", "hierarchical"):
        raise ValueError("GroupImputer fallback must be global or hierarchical.")
    return value


def _prefixes(keys: list[str], fallback: str) -> list[list[str]]:
    """Keep the full key first, followed by successively coarser prefixes when requested."""
    widths = range(len(keys), 0, -1) if fallback == "hierarchical" else [len(keys)]
    return [keys[:width] for width in widths]


def _key_rows(frame: Any, keys: list[str]) -> list[tuple[str, ...]]:
    """Read group scalars column-wise so mixed rows cannot coerce large integers to floats."""
    return [
        tuple(category_key(value) for value in row)
        for row in zip(*(frame[key].to_list() for key in keys), strict=True)
    ]


def _fit_level(frame: pd.DataFrame, keys: list[str], columns: list[str], strategy: str) -> dict:
    """Fit each level directly from observed rows rather than averaging child statistics."""
    positions: dict[tuple[str, ...], list[int]] = {}
    for position, key in enumerate(_key_rows(frame, keys)):
        if "nan" not in key:
            positions.setdefault(key, []).append(position)
    groups = {}
    for column in columns:
        pairs = []
        for key, rows in sorted(positions.items()):
            value = group_statistic(frame[column].iloc[np.asarray(rows, dtype=np.intp)], strategy)
            if value is not None:
                pairs.append([list(key), value])
        groups[column] = pairs
    return {"group_by": keys, "group_values": groups}


def fit_group_hierarchy(
    frame: pd.DataFrame, config: dict, columns: list[str], strategy: str
) -> GroupImputerArtifact:
    """Persist scalar lookup tables for the full key and each configured fallback prefix."""
    keys = group_columns(config["group_by"])
    fallback = group_fallback(config.get("fallback", "global"))
    frame = frame.copy()
    if strategy in ("mean", "median"):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column])
    return {
        "type": "group_imputer",
        "group_key_version": 1,
        "group_by": keys,
        "strategy": strategy,
        "columns": columns,
        "fallback": fallback,
        "levels": [
            _fit_level(frame, prefix, columns, strategy) for prefix in _prefixes(keys, fallback)
        ],
        "fill_values": {column: group_statistic(frame[column], strategy) for column in columns},
    }


def hierarchy_fills(frame: Any, state: dict, column: str) -> list[Any]:
    """Choose fixed training values from most specific to global without request aggregation."""
    fills = [state["fill_values"][column]] * len(frame)
    for level in reversed(state["levels"]):
        lookup = {tuple(key): value for key, value in level["group_values"][column]}
        for position, key in enumerate(_key_rows(frame, level["group_by"])):
            if key in lookup:
                fills[position] = lookup[key]
    return fills


def _validate_value(value: Any, numeric: bool) -> None:
    """Require finite portable scalars while preserving exact modal integer values."""
    _scalar(value)
    if numeric and type(value) not in (int, float, type(None)):
        raise ValueError("Group statistics must be numeric or missing.")


def _validate_pairs(pairs: Any, width: int, numeric: bool) -> None:
    """Bind every canonical tuple to the level width with no duplicate or missing keys."""
    if type(pairs) is not list:
        raise ValueError("Group replacements must be a list.")
    seen = set()
    for pair in pairs:
        if type(pair) is not list or len(pair) != 2:
            raise ValueError("Group replacements must be key/value pairs.")
        key, value = pair
        token = _tuple_key(key, width)
        if token in seen or value is None:
            raise ValueError("Group tuple keys must be unique and have observed replacements.")
        seen.add(token)
        _validate_value(value, numeric)


def _tuple_key(key: Any, width: int) -> tuple[str, ...]:
    """Require a portable, nonmissing canonical tuple with the configured level width."""
    if type(key) is not list or len(key) != width:
        raise ValueError("Group tuple key width must match its level.")
    if any(type(item) is not str or item == "nan" for item in key):
        raise ValueError("Group tuple keys must contain nonmissing canonical values.")
    return tuple(cast(list[str], key))


def validate_group_hierarchy(state: dict) -> dict:
    """Inspect exact level order and portable values before the existing batch admission path."""
    _fields(
        state,
        {
            "type",
            "group_key_version",
            "group_by",
            "strategy",
            "columns",
            "fallback",
            "levels",
            "fill_values",
        },
    )
    if type(state["group_key_version"]) is not int or state["group_key_version"] != 1:
        raise ValueError("Unsupported GroupImputer group key version.")
    if state["type"] != "group_imputer" or state["strategy"] not in (
        "mean",
        "median",
        "most_frequent",
    ):
        raise ValueError("Unsupported group imputer state.")
    keys = group_columns(state["group_by"])
    columns = _columns(state["columns"])
    if set(keys).intersection(columns):
        raise ValueError("Group columns cannot be filled by themselves.")
    prefixes = _prefixes(keys, group_fallback(state["fallback"]))
    numeric = state["strategy"] in ("mean", "median")
    _fields(state["fill_values"], set(columns))
    for value in state["fill_values"].values():
        _validate_value(value, numeric)
    _validate_levels(state["levels"], prefixes, columns, numeric)
    return state


def _validate_levels(
    levels: Any, prefixes: list[list[str]], columns: list[str], numeric: bool
) -> None:
    """Validate the exact fallback chain and every column's replacement table."""
    if type(levels) is not list or len(levels) != len(prefixes):
        raise ValueError("Group levels must match the configured fallback order.")
    for item, prefix in zip(levels, prefixes, strict=True):
        _fields(item, {"group_by", "group_values"})
        level = cast(dict, item)
        if level["group_by"] != prefix:
            raise ValueError("Group levels must be ordered prefixes of group_by.")
        _fields(level["group_values"], set(columns))
        for pairs in level["group_values"].values():
            _validate_pairs(pairs, len(prefix), numeric)
