"""Explicit, lossless nullable primitive transport across MLflow signatures."""

import re
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

TRANSPORT_KEY = "skyulf_input_transport"
_DTYPES = {"Int32", "Int64", "boolean", "Boolean"}
_INTEGER = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")


def transport_spec(columns: Iterable[tuple[str, str]]) -> dict[str, Any] | None:
    """Describe only dtypes whose nullable storage MLflow cannot validate natively."""
    selected = {name: dtype for name, dtype in columns if dtype in _DTYPES}
    return {"codec": "nullable_primitives_v1", "columns": selected} if selected else None


def validated_transport(
    recorded: dict[str, Any] | None, columns: Iterable[tuple[str, str]]
) -> dict[str, Any] | None:
    """Bind a saved codec to the artifact schema while preserving legacy packages."""
    if recorded is not None and recorded != transport_spec(columns):
        raise ValueError("MLflow nullable input transport differs from the fitted artifact schema.")
    return deepcopy(recorded)


def prepare_pyfunc_input(frame: pd.DataFrame, loaded_model: Any) -> pd.DataFrame:
    """Prepare native nullable inputs for a loaded Skyulf local or model-set pyfunc.

    Pass this returned frame to ``loaded_model.predict``. Nullable Int32/Int64
    and Boolean columns use canonical strings and None across MLflow's public
    signature validation; the adapter restores fitted dtypes before feature
    engineering. Floating-point integers and already encoded strings are
    rejected, since their original integer precision cannot be established.
    Other columns, row order and index are retained without mutating ``frame``.
    Spark callers must cast these columns to strings before ``spark_udf``.
    """
    python_model = loaded_model.unwrap_python_model()
    accessor = getattr(python_model, "input_transport", None)
    if not callable(accessor):
        raise TypeError("prepare_pyfunc_input requires a Skyulf local or model-set pyfunc.")
    spec = accessor()
    metadata = loaded_model.metadata.metadata or {}
    if metadata.get(TRANSPORT_KEY) != spec:
        raise ValueError("MLflow nullable input transport metadata differs from the saved model.")
    return encode_frame(frame, spec)


def _copy_frame(frame: pd.DataFrame, spec: dict[str, Any] | None) -> pd.DataFrame:
    """Reject ambiguous columns before copying caller-owned data."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("MLflow nullable input transport requires a pandas DataFrame.")
    if not frame.columns.is_unique:
        raise ValueError("MLflow nullable input transport requires unique column names.")
    missing = set(spec["columns"]) - set(frame.columns) if spec else set()
    if missing:
        raise ValueError(
            f"MLflow nullable input transport is missing required columns: {sorted(missing)}."
        )
    return frame.copy()


def _is_missing(value: Any) -> bool:
    """Recognize scalar nulls without evaluating array truthiness or coercing values."""
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, (float, np.floating)) and bool(np.isnan(value)))
    )


def _bounded_integer(value: int, dtype: str, name: str) -> int:
    """Check exact signed bounds before creating a nullable integer array."""
    bits = 32 if dtype == "Int32" else 64
    if not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1):
        raise ValueError(f"Column {name!r} integer is outside {dtype} bounds.")
    return value


def _encode_value(value: Any, dtype: str, name: str) -> str | None:
    """Encode native scalars without numeric casts that could lose integer bits."""
    if _is_missing(value):
        return None
    if dtype in {"boolean", "Boolean"}:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"Column {name!r} requires native boolean values or nulls.")
        return "true" if value else "false"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"Column {name!r} requires exact native integer values or nulls.")
    return str(_bounded_integer(int(value), dtype, name))


def _decode_value(value: Any, dtype: str, name: str) -> int | bool | None:
    """Parse only canonical wire strings, with exact ranges and null preservation."""
    if _is_missing(value):
        return None
    if not isinstance(value, str):
        raise TypeError(f"Column {name!r} requires canonical string transport or nulls.")
    if dtype in {"boolean", "Boolean"}:
        if value not in {"true", "false"}:
            raise ValueError(f"Column {name!r} requires canonical boolean strings true/false.")
        return value == "true"
    if len(value) > 20 or not _INTEGER.fullmatch(value):
        raise ValueError(f"Column {name!r} requires canonical integer strings.")
    return _bounded_integer(int(value), dtype, name)


def encode_frame(frame: pd.DataFrame, spec: dict[str, Any] | None) -> pd.DataFrame:
    """Encode declared native nullable columns while retaining all other input data."""
    result = _copy_frame(frame, spec)
    for name, dtype in (spec["columns"] if spec else {}).items():
        result[name] = pd.Series(
            [_encode_value(value, dtype, name) for value in frame[name]],
            index=frame.index,
            dtype=object,
        )
    return result


def decode_frame(frame: pd.DataFrame, spec: dict[str, Any] | None) -> pd.DataFrame:
    """Restore nullable pandas storage before engine conversion and fitted transforms."""
    if spec is None:
        return frame
    result = _copy_frame(frame, spec)
    for name, dtype in spec["columns"].items():
        values = [_decode_value(value, dtype, name) for value in frame[name]]
        result[name] = pd.Series(
            values, index=frame.index, dtype="boolean" if dtype == "Boolean" else dtype
        )
    return result


def restore_nullable_dtypes(
    frame: pd.DataFrame, columns: Iterable[tuple[str, str]]
) -> pd.DataFrame:
    """Restore extension dtypes only from their exact matching NumPy storage type."""
    storage = {
        "Int32": "int32",
        "Int64": "int64",
        "Float32": "float32",
        "Float64": "float64",
        "boolean": "bool",
    }
    replacements = {
        name: dtype
        for name, dtype in columns
        if dtype in storage and name in frame and str(frame[name].dtype) == storage[dtype]
    }
    return frame.astype(replacements) if replacements else frame
