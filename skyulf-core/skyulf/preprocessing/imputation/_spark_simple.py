"""Native Spark SimpleImputer branches with one bounded fit aggregate."""

import importlib
import math
from typing import Any

from ...core.capabilities import require_capability
from ...core.portable_state import validate_state
from .._spark import _column
from .._spark_numeric import (
    NUMERIC_DTYPES,
    excluded,
    missing,
    select_columns,
    selection_statistics,
    validate_names,
)


def _constant(value: Any, dtype: str) -> Any:
    """Require unambiguous scalar constants, preserving integer precision."""
    if dtype in NUMERIC_DTYPES:
        value = 0 if value is None else value
        if type(value) not in (int, float):
            raise TypeError("Numeric Spark columns require a numeric fill_value.")
        if type(value) is int and not -(2**63) <= value < 2**63:
            raise ValueError("Spark integer fill_value must fit a signed 64-bit integer.")
        return value
    if dtype == "string":
        if value is None:
            raise ValueError("String constant imputation requires an explicit fill_value.")
        if type(value) is str:
            return value
    if dtype == "boolean" and type(value) is bool:
        return value
    raise TypeError("Unsupported Spark imputer dtype or incompatible fill_value.")


def _aggregate_expressions(
    frame: Any, cols: list[str], schema: dict, strategy: str, automatic: bool, functions: Any
) -> list[Any]:
    """Combine missing counts, means and optional selection statistics in one query."""
    expressions = []
    for index, name in enumerate(cols):
        absent = missing(frame, name, schema[name], functions)
        expressions.append(functions.sum(functions.when(absent, 1).otherwise(0)).alias(f"n{index}"))
        if strategy == "mean":
            valid = functions.when(~absent, _column(frame, name))
            expressions.append(functions.avg(valid).alias(f"v{index}"))
            if automatic:
                expressions.extend(selection_statistics(valid, index, functions))
    return expressions


def fit_spark_imputer(frame: Any, target: Any, config: dict) -> dict:
    """Learn native mean/constant state, returning only one aggregate row to Python."""
    strategy = config.get("strategy", "mean")
    require_capability("SimpleImputer", "fit", "spark", config={**config, "strategy": strategy})
    cols, schema, automatic = select_columns(frame, config, numeric_only=strategy == "mean")
    if not cols:
        return {}
    constants = {}
    for name in cols:
        if strategy == "mean" and schema[name] not in NUMERIC_DTYPES:
            raise TypeError("Spark mean imputation requires numeric columns.")
        if strategy == "constant":
            constants[name] = _constant(config.get("fill_value"), schema[name])
    functions = importlib.import_module("pyspark.sql.functions")
    stats = frame.agg(
        *_aggregate_expressions(frame, cols, schema, strategy, automatic, functions)
    ).first()
    return _artifact(cols, strategy, automatic, stats, constants)


def _artifact(cols: list[str], strategy: str, automatic: bool, stats: Any, constants: dict) -> dict:
    """Build the existing artifact vocabulary without inventing all-null means."""
    fills = {}
    counts = {}
    for index, name in enumerate(cols):
        value = constants[name] if strategy == "constant" else stats[f"v{index}"]
        if strategy == "mean" and value is not None and not math.isfinite(value):
            raise ValueError("Spark mean imputation requires finite observed values.")
        if strategy == "mean" and automatic and excluded(stats, index):
            continue
        fills[name] = value
        counts[name] = int(stats[f"n{index}"] or 0)
    if not fills:
        return {}
    return {
        "type": "simple_imputer",
        "strategy": strategy,
        "columns": list(fills),
        "fill_values": fills,
        "missing_counts": counts,
        "total_missing": sum(counts.values()),
    }


def apply_spark_imputer(frame: Any, target: Any, params: dict) -> tuple[Any, Any]:
    """Fill with native Spark expressions; do not refit or execute a data action."""
    if not params:
        return frame, target
    require_capability("SimpleImputer", "apply", "spark", config=params)
    state = validate_state("SimpleImputer", params)
    validate_names(frame, state["columns"])
    functions = importlib.import_module("pyspark.sql.functions")
    schema = {field.name: field.dataType.typeName() for field in frame.schema.fields}
    expressions = {name: _column(frame, name) for name in frame.columns}
    for name in state["columns"]:
        value = state["fill_values"][name]
        if value is None or (
            state["strategy"] == "mean" and isinstance(value, float) and math.isnan(value)
        ):
            continue
        replacement = functions.lit(value)
        if name in schema:
            _constant(value, schema[name])
            if state["strategy"] == "mean" and schema[name] not in NUMERIC_DTYPES:
                raise TypeError("Spark mean imputation requires numeric columns.")
            absent = missing(frame, name, schema[name], functions)
            replacement = functions.when(absent, replacement).otherwise(_column(frame, name))
        expressions[name] = replacement.alias(name)
    return frame.select(*expressions.values()), target
