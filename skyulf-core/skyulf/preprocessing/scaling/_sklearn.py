"""Shared matrix restoration and fitted checks for the two native sklearn scalers."""

from typing import Any

import numpy as np
import polars as pl

from ...utils import detect_numeric_columns
from .._fitted_validation import _columns, local_state_fields
from .._helpers import promote_configured_columns_to_float64, resolve_columns_then_to_numpy
from .._schema import SkyulfSchema


def scaler_fit_values(X: Any, config: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Select numeric features before converting their values to the promised float64 dtype."""
    values, columns = resolve_columns_then_to_numpy(X, config, detect_numeric_columns)
    _columns(columns)
    return np.asarray(values, dtype=np.float64), columns


def scaler_output_schema(input_schema: SkyulfSchema, config: dict[str, Any]) -> SkyulfSchema | None:
    """Promote explicit columns; automatic discovery needs data to exclude binary/constants."""
    if config.get("columns") is None:
        return None
    return promote_configured_columns_to_float64(input_schema, config)


def validate_scaler_object(raw: dict, kind: str, expected: type) -> Any:
    """Require the original fitted estimator and feature width without running its methods."""
    if not local_state_fields(
        raw, kind, {"type", "columns", "transformer_object"}, allow_empty=True
    ):
        return None
    columns = _columns(raw["columns"])
    transformer = raw["transformer_object"]
    if type(transformer) is not expected or not columns:
        raise ValueError(f"Expected a fitted {expected.__name__} with selected columns.")
    if getattr(transformer, "n_features_in_", None) != len(columns):
        raise ValueError("Fitted scaler width disagrees with selected columns.")
    if any(callable(value) for value in vars(transformer).values()):
        raise ValueError("Overridden scaler methods are unsupported.")
    return transformer


def _scaled_values(X: Any, params: dict[str, Any]) -> np.ndarray:
    """Transform exactly the saved feature matrix, including a typed empty partition."""
    columns = params["columns"]
    missing = [col for col in columns if col not in X.columns]
    if missing:
        raise ValueError(f"Fitted scaler requires columns: {missing}.")
    values, _ = scaler_fit_values(X, {"columns": columns})
    if not len(values):
        return np.empty((0, len(columns)), dtype=np.float64)
    # Serving may enable sklearn's pandas output; restoration below needs a matrix.
    return np.asarray(params["transformer_object"].transform(values))


def apply_scaler_pandas(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Restore transformed columns without changing the pandas index or unselected values."""
    if not params:
        return X, y
    values = _scaled_values(X, params)
    output = X.copy()
    for index, column in enumerate(params["columns"]):
        output[column] = values[:, index]
    return output, y


def apply_scaler_polars(X: Any, y: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """Replace only fitted Polars columns, retaining their original positions."""
    if not params:
        return X, y
    values = _scaled_values(X, params)
    return X.with_columns(
        [pl.Series(column, values[:, index]) for index, column in enumerate(params["columns"])]
    ), y
