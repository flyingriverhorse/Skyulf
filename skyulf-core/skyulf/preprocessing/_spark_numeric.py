"""Shared native Spark column selection and bounded numeric selection statistics."""

from typing import Any

from ._spark import _case_sensitive, _column, _resolved_names

NUMERIC_DTYPES = {"byte", "short", "integer", "long", "float", "double"}


def validate_names(frame: Any, extra: tuple | list = ()) -> None:
    """Reject ambiguous input and learned names under the session's identifier rules."""
    names = _resolved_names(frame)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate Spark column names.")
    combined = list(dict.fromkeys([*frame.columns, *extra]))
    if not _case_sensitive(frame):
        combined = [name.lower() for name in combined]
    if len(set(combined)) != len(combined):
        raise ValueError("Fitted columns collide under Spark column name resolution.")


def select_columns(
    frame: Any, config: dict, *, numeric_only: bool
) -> tuple[list[str], dict[str, str], bool]:
    """Resolve exact names and automatic schema candidates without a data action."""
    validate_names(frame)
    schema = {field.name: field.dataType.typeName() for field in frame.schema.fields}
    automatic = config.get("_auto_columns", config.get("columns") is None)
    cols = config.get("columns")
    if cols is None:
        cols = [name for name in frame.columns if name != config.get("target_column")]
    if not isinstance(cols, list) or any(not isinstance(name, str) for name in cols):
        raise ValueError("columns must be a list of names.")
    if len(set(cols)) != len(cols) or set(cols).difference(schema):
        raise ValueError("Spark feature columns are missing or duplicated.")
    if automatic and numeric_only:
        cols = [name for name in cols if schema[name] in NUMERIC_DTYPES]
    return cols, schema, automatic


def missing(frame: Any, name: str, dtype: str, functions: Any) -> Any:
    """Treat nulls and floating NaNs as missing using native expressions."""
    column = _column(frame, name)
    return (
        column.isNull() | functions.isnan(column)
        if dtype in {"float", "double"}
        else column.isNull()
    )


def selection_statistics(valid: Any, index: int, functions: Any) -> list[Any]:
    """Mirror local binary/constant exclusion without returning distinct values."""
    numeric = valid.cast("double")
    binary = (functions.abs(numeric) <= 1e-8) | (functions.abs(numeric - 1) <= 1.001e-5)
    return [
        functions.min(valid).alias(f"lo{index}"),
        functions.max(valid).alias(f"hi{index}"),
        functions.count_distinct(valid).alias(f"distinct{index}"),
        functions.max(functions.when(~binary, 1).otherwise(0)).alias(f"other{index}"),
    ]


def excluded(stats: Any, index: int) -> bool:
    """Identify empty, constant or at-most-two-value binary automatic candidates."""
    return stats[f"lo{index}"] == stats[f"hi{index}"] or (
        stats[f"distinct{index}"] <= 2 and not stats[f"other{index}"]
    )
