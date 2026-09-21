"""Native Spark SimpleImputer branches with one bounded fit aggregate."""

import importlib
import math
from typing import Any

from ...core.capabilities import require_capability
from ...core.portable_state import validate_state
from .._spark import _column, _resolved_names

_NUMERIC = {"byte", "short", "integer", "long", "float", "double"}
_FLOATING = {"float", "double"}


def _missing(frame: Any, name: str, dtype: str, functions: Any) -> Any:
    """Treat both nulls and floating NaNs as missing using native expressions."""
    column = _column(frame, name)
    return column.isNull() | functions.isnan(column) if dtype in _FLOATING else column.isNull()


def _columns(frame: Any, config: dict, strategy: str) -> tuple[list[str], dict[str, str], bool]:
    """Resolve exact names and schema-eligible candidates without a data action."""
    names = _resolved_names(frame)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate Spark column names.")
    schema = {field.name: field.dataType.typeName() for field in frame.schema.fields}
    automatic = config.get("_auto_columns", config.get("columns") is None)
    cols = config.get("columns")
    if cols is None:
        cols = [name for name in frame.columns if name != config.get("target_column")]
    if not isinstance(cols, list) or any(not isinstance(name, str) for name in cols):
        raise ValueError("columns must be a list of names.")
    if len(set(cols)) != len(cols) or set(cols).difference(schema):
        raise ValueError("Spark imputer columns are missing or duplicated.")
    if automatic and strategy == "mean":
        cols = [name for name in cols if schema[name] in _NUMERIC]
    return cols, schema, automatic


def _constant(value: Any, dtype: str) -> Any:
    """Require unambiguous scalar constants, preserving integer precision."""
    if dtype in _NUMERIC:
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
        missing = _missing(frame, name, schema[name], functions)
        expressions.append(
            functions.sum(functions.when(missing, 1).otherwise(0)).alias(f"n{index}")
        )
        if strategy == "mean":
            valid = functions.when(~missing, _column(frame, name))
            expressions.append(functions.avg(valid).alias(f"v{index}"))
            if automatic:
                expressions.extend(_selection_statistics(valid, index, functions))
    return expressions


def _selection_statistics(valid: Any, index: int, functions: Any) -> list[Any]:
    """Mirror local binary/constant exclusion without collecting distinct values."""
    numeric = valid.cast("double")
    binary = (functions.abs(numeric) <= 1e-8) | (functions.abs(numeric - 1) <= 1.001e-5)
    return [
        functions.min(valid).alias(f"lo{index}"),
        functions.max(valid).alias(f"hi{index}"),
        functions.count_distinct(valid).alias(f"distinct{index}"),
        functions.max(functions.when(~binary, 1).otherwise(0)).alias(f"other{index}"),
    ]


def fit_spark_imputer(frame: Any, target: Any, config: dict) -> dict:
    """Learn native mean/constant state, returning only one aggregate row to Python."""
    strategy = config.get("strategy", "mean")
    require_capability("SimpleImputer", "fit", "spark", config={**config, "strategy": strategy})
    cols, schema, automatic = _columns(frame, config, strategy)
    if not cols:
        return {}
    constants = {}
    for name in cols:
        if strategy == "mean" and schema[name] not in _NUMERIC:
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
        if (
            strategy == "mean"
            and automatic
            and (
                stats[f"lo{index}"] == stats[f"hi{index}"]
                or (stats[f"distinct{index}"] <= 2 and not stats[f"other{index}"])
            )
        ):
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
    names = _resolved_names(frame)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate Spark column names.")
    combined = list(dict.fromkeys([*frame.columns, *state["columns"]]))
    if frame.sparkSession.conf.get("spark.sql.caseSensitive") != "true":
        combined = [name.lower() for name in combined]
    if len(set(combined)) != len(combined):
        raise ValueError("Fitted columns collide under Spark column name resolution.")
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
            if state["strategy"] == "mean" and schema[name] not in _NUMERIC:
                raise TypeError("Spark mean imputation requires numeric columns.")
            missing = _missing(frame, name, schema[name], functions)
            replacement = functions.when(missing, replacement).otherwise(_column(frame, name))
        expressions[name] = replacement.alias(name)
    return frame.select(*expressions.values()), target
