"""Versioned row-local arithmetic over explicitly named model prediction columns."""

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .project_scoring import _identifier, _validate_columns

_FIELDS = {"name", "version", "operation", "params", "columns", "required_components"}
_OPERATIONS = {"weighted_sum", "weighted_mean"}


def _weights(rule: dict) -> list[float]:
    """Require finite native numeric weights with explicit weighted-mean semantics."""
    raw = rule["params"]["weights"]
    inputs = rule["params"]["inputs"]
    if type(raw) is not list or len(raw) != len(inputs):
        raise ValueError("Composition weights must match the ordered inputs.")
    weights = _finite_weights(raw)
    total = _weight_sum(weights)
    if rule["operation"] == "weighted_mean":
        if min(weights) < 0 or total <= 0:
            raise ValueError("Weighted mean requires nonnegative weights with a positive sum.")
        return [weight / total for weight in weights]
    return weights


def _finite_weights(raw: list) -> list[float]:
    """Convert native numeric scalars only after rejecting booleans and coercible strings."""
    if any(type(value) not in (int, float) for value in raw):
        raise ValueError("Composition weights must be finite numeric scalars.")
    try:
        weights = [float(value) for value in raw]
    except (OverflowError, ValueError) as exc:
        raise ValueError("Composition weights must be finite numeric scalars.") from exc
    if not all(math.isfinite(value) for value in weights):
        raise ValueError("Composition weights must be finite numeric scalars.")
    return weights


def _weight_sum(weights: list[float]) -> float:
    """Require a finite denominator even when every individual weight is finite."""
    try:
        return math.fsum(weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Composition weight sum must be finite.") from exc


def _inputs(rule: dict) -> list[str]:
    """Reject implicit parameters, repeated inputs and arbitrary expression strings."""
    params = rule["params"]
    if type(params) is not dict or set(params) != {"inputs", "weights"}:
        raise ValueError("Composition params require exactly inputs and weights.")
    inputs = params["inputs"]
    if type(inputs) is not list or not inputs or any(type(name) is not str for name in inputs):
        raise ValueError("Composition inputs require named prediction columns.")
    if len(set(inputs)) != len(inputs):
        raise ValueError("Composition inputs must be unique.")
    return inputs


def validate_operation(rule: dict, components: Any) -> None:
    """Bind one deterministic rule to numeric direct outputs and an exact output schema."""
    if (
        set(rule) != _FIELDS
        or type(rule["operation"]) is not str
        or rule["operation"] not in _OPERATIONS
    ):
        raise ValueError("Unknown declarative composition operation or rule fields.")
    if not _identifier(rule["name"]) or rule["version"] != "1":
        raise ValueError("Declarative composition requires a simple name and version '1'.")
    _validate_columns(rule["columns"])
    if len(rule["columns"]) != 1 or rule["columns"][0]["dtype"] != "float64":
        raise ValueError("Declarative composition requires exactly one float64 output column.")
    inputs = _inputs(rule)
    _weights(rule)
    _input_dependencies(inputs, _prediction_columns(components), rule["required_components"])


def _prediction_columns(components: Any) -> dict[str, tuple[str, str]]:
    """Expose explicit prediction/probability fields without publishing raw component inputs."""
    return {
        f"{component.branch}__{column.name}": (component.branch, column.dtype)
        for component in components
        for column in component.output_schema
        if column.name == "prediction" or re.fullmatch(r"probability_\d+", column.name)
    }


def _input_dependencies(inputs: list[str], available: dict, required: list[str]) -> None:
    """Forbid raw features, string labels, undeclared branches and unnecessary exclusions."""
    if any(name not in available for name in inputs):
        raise ValueError("Composition inputs must name direct component prediction columns.")
    if any(available[name][1] not in {"float64", "int64"} for name in inputs):
        raise ValueError("Composition inputs require numeric prediction columns.")
    if {available[name][0] for name in inputs} != set(required):
        raise ValueError("Composition required_components must exactly match input dependencies.")


def apply_operation(predictions: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Evaluate fixed-order per-row arithmetic without source loading or batch statistics."""
    inputs = _inputs(rule)
    weights = _weights(rule)
    values = np.zeros(len(predictions), dtype="float64")
    try:
        with np.errstate(over="raise", invalid="raise"):
            for name, weight in zip(inputs, weights, strict=True):
                values += predictions[name].to_numpy(dtype="float64") * weight
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("Composition produced non-finite numeric outputs.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Composition produced non-finite numeric outputs.")
    return pd.DataFrame({rule["columns"][0]["name"]: values}, index=predictions.index)
