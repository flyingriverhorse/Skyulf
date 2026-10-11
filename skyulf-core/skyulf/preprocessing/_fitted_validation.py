"""Shared structural checks for node-owned fitted-state inspection."""

import math
from typing import Any

import numpy as np
import pandas as pd

from ..core.portable_pipeline import _config
from ..core.portable_state import _normalize


def local_state_fields(raw: Any, kind: str, fields: set[str], *, allow_empty: bool = False) -> bool:
    """Inspect a local artifact's shape without granting portable execution.

    Owners decide whether an empty dictionary is a real fitted no-op. They
    validate their own values after this shared field and discriminator check.
    The original state is neither normalized nor mutated.
    """
    if type(raw) is not dict:
        raise ValueError("Local fitted state must be a dictionary.")
    if not raw and allow_empty:
        return False
    if set(raw) != fields or raw.get("type") != kind:
        raise ValueError("Unexpected local fitted state fields or type.")
    return True


def local_boolean(value: Any, name: str) -> None:
    """Accept saved Python/NumPy booleans without coercing integer flags."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Fitted {name} must be a boolean.")


def local_scalar(value: Any, name: str) -> None:
    """Inspect Python/NumPy scalar rules without converting null or non-finite values."""
    if value is pd.NA:
        return
    numpy_scalar = isinstance(value, np.generic) and value.dtype.kind in "biufU"
    if type(value) not in (str, int, float, bool, type(None)) and not numpy_scalar:
        raise ValueError(f"{name} must contain scalar values.")


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
