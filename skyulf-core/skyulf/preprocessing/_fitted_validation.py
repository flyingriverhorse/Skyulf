"""Shared structural checks for node-owned fitted-state inspection."""

import math
from typing import Any

from ..core.portable_pipeline import _config
from ..core.portable_state import _normalize


def _fields(value: Any, names: set[str]) -> None:
    """Reject unknown or missing state fields rather than ignoring new behavior."""
    if type(value) is not dict or set(value) != names:
        raise ValueError("Unexpected Python-batch state fields.")


def _columns(value: Any) -> list[str]:
    """Require unambiguous ordered feature names."""
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError("Fitted columns must be a list of strings.")
    if len(set(value)) != len(value):
        raise ValueError("Fitted columns must be unique.")
    return value


def _scalar(value: Any) -> None:
    """Allow only finite learned scalars without callbacks or nested objects."""
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("Learned values must be scalar.")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Learned values must be finite.")


def portable_config(node: str, raw: dict, state: dict) -> dict:
    """Normalize fitted defaults while rejecting changed explicit column selection."""
    columns = raw.get("columns")
    if (
        isinstance(columns, list)
        and not raw.get("_auto_columns")
        and columns != state.get("columns", [])
    ):
        raise ValueError("Configured columns disagree with fitted columns.")
    resolved = _config(node, raw, state)
    if node == "SimpleImputer" and resolved["strategy"] == "constant":
        fill = resolved["fill_value"]
        if fill is not None and any(
            value != fill for value in state.get("fill_values", {}).values()
        ):
            raise ValueError("Configured constant disagrees with fitted fill values.")
    return resolved


def fitted_columns(raw: dict, state: dict) -> dict:
    """Resolve learned column selection without changing explicit caller choices."""
    params = _normalize(raw)
    params.pop("target_column", None)
    auto = params.pop("_auto_columns", False)
    columns = params.get("columns")
    if columns is not None and not auto and columns != state["columns"]:
        raise ValueError("Configured columns disagree with fitted columns.")
    params["columns"] = state["columns"]
    return params
