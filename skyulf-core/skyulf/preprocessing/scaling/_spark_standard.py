"""Native Spark StandardScaler with population variance and bounded learned state."""

import importlib
import math
import sys
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


def fit_spark_standard(frame: Any, target: Any, config: dict) -> dict:
    """Learn shifted statistics from bounded aggregates without materializing samples."""
    flags = {key: config.get(key, True) for key in ("with_mean", "with_std")}
    require_capability("StandardScaler", "fit", "spark", config=flags)
    cols, schema, automatic = select_columns(frame, config, numeric_only=True)
    if not cols:
        return {}
    for name in cols:
        _validate_dtype(schema[name])
    functions = importlib.import_module("pyspark.sql.functions")
    expressions = [functions.count(functions.lit(1)).alias("rows")]
    for index, name in enumerate(cols):
        valid = functions.when(~missing(frame, name, schema[name], functions), _column(frame, name))
        numeric = valid.cast("double")
        expressions.extend(
            [
                functions.min(numeric).alias(f"min{index}"),
                functions.max(numeric).alias(f"max{index}"),
                functions.max(
                    functions.when(functions.abs(numeric) == float("inf"), 1).otherwise(0)
                ).alias(f"inf{index}"),
            ]
        )
        if automatic:
            expressions.extend(selection_statistics(valid, index, functions))
    references = frame.agg(*expressions).first()
    selected = [i for i in range(len(cols)) if not automatic or not excluded(references, i)]
    if not selected:
        return {}
    if not references["rows"]:
        raise ValueError("Cannot fit StandardScaler on empty training data.")
    if any(references[f"inf{index}"] for index in selected):
        raise ValueError("Spark StandardScaler requires finite training values; infinity found.")
    expressions = []
    if flags["with_mean"] or flags["with_std"]:
        for index in selected:
            name = cols[index]
            valid = functions.when(
                ~missing(frame, name, schema[name], functions), _column(frame, name)
            )
            shifted = valid.cast("double") - _reference(references, index)
            expressions.extend(_statistics(shifted, index, flags, functions))
    stats = frame.agg(*expressions).first() if expressions else None
    return _artifact(cols, selected, references, stats, flags)


def _validate_dtype(dtype: str) -> None:
    """Allow primitive numbers and explicitly selected booleans without string parsing."""
    if dtype not in NUMERIC_DTYPES | {"boolean"}:
        raise TypeError("Spark StandardScaler requires numeric columns or explicit booleans.")


def _reference(stats: Any, index: int) -> float:
    """Shift same-sign offsets toward zero without erasing mixed-sign small values."""
    lower, upper = stats[f"min{index}"], stats[f"max{index}"]
    if lower is None or lower <= 0 <= upper:
        return 0.0
    return lower if lower > 0 else upper


def _statistics(shifted: Any, index: int, flags: dict, functions: Any) -> list[Any]:
    """Aggregate deviations so repeated large offsets do not accumulate rounding error."""
    result = [functions.avg(shifted).alias(f"mean{index}")]
    if flags["with_std"]:
        result.extend(
            [
                functions.var_pop(shifted).alias(f"var{index}"),
                functions.count(shifted).alias(f"n{index}"),
            ]
        )
    return result


def _artifact(
    cols: list[str], selected: list[int], references: Any, stats: Any, flags: dict
) -> dict:
    """Keep local flag-dependent None statistics and NaN all-missing statistics."""
    means = [] if flags["with_mean"] or flags["with_std"] else None
    variances = [] if flags["with_std"] else None
    scales = [] if flags["with_std"] else None
    for index in selected:
        mean = (
            _statistic(stats[f"mean{index}"]) + _reference(references, index)
            if means is not None
            else None
        )
        if means is not None:
            means.append(mean)
        if variances is not None and scales is not None:
            assert mean is not None
            variance = _statistic(stats[f"var{index}"])
            variances.append(variance)
            scales.append(_scale(variance, mean, stats[f"n{index}"]))
    return {
        "type": "standard_scaler",
        "columns": [cols[index] for index in selected],
        "mean": means,
        "var": variances,
        "scale": scales,
        **flags,
    }


def _statistic(value: Any) -> float:
    """Represent missing statistics as NaN and reject overflowing aggregates."""
    if value is None:
        return float("nan")
    if not math.isfinite(value):
        raise ValueError("Spark StandardScaler aggregate is non-finite; rescale training values.")
    return float(value)


def _scale(variance: float, mean: float, count: int) -> float:
    """Match sklearn's float64 constant-feature error bound using observed counts."""
    epsilon = sys.float_info.epsilon
    error = count * mean * epsilon
    upper_bound = count * epsilon * variance + error * error
    return 1.0 if variance <= upper_bound else math.sqrt(variance)


def apply_spark_standard(frame: Any, target: Any, params: dict) -> tuple[Any, Any]:
    """Apply learned statistics with literal native expressions and no data actions."""
    if not params:
        return frame, target
    state = validate_state("StandardScaler", params)
    require_capability("StandardScaler", "apply", "spark", config=state)
    validate_names(frame, state["columns"])
    if not (state["with_mean"] or state["with_std"]):
        return frame, target
    schema = {field.name: field.dataType.typeName() for field in frame.schema.fields}
    expressions = {name: _column(frame, name) for name in frame.columns}
    for index, name in enumerate(state["columns"]):
        if name not in schema:
            continue
        _validate_dtype(schema[name])
        expr = _column(frame, name).cast("double")
        if state["with_mean"]:
            expr = expr - state["mean"][index]
        if state["with_std"]:
            scale = state["scale"][index]
            expr = expr / (scale if scale != 0 else 1.0)
        expressions[name] = expr.alias(name)
    return frame.select(*expressions.values()), target
