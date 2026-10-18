"""Per-column missing-value markers for monitored features, such as a -200 sensor reading.

The policy lives under ``missing_markers`` in the ``monitoring_drift_thresholds`` JSON. It is
split off at the monitoring boundary because Core drift settings reject unknown keys, so a
direct ``DriftCalculator`` caller can never pass markers that it would then ignore. Report
builders turn marked values into nulls in both the reference and the current population
before quality and drift run. Markers compare in the column's own dtype, so a large
integer never matches a neighbour through a float conversion. Prepared reference tables keep raw rows, so a policy change
never reuses populations normalized under another policy.
"""

import math
import struct
from typing import Any

MARKERS_KEY = "missing_markers"
MAX_MARKERS_PER_COLUMN = 100
Marker = int | float
# Exact value range per integer dtype; engines name their integer types with these keys.
INTEGER_BOUNDS = {
    "Int8": (-(2**7), 2**7 - 1),
    "Int16": (-(2**15), 2**15 - 1),
    "Int32": (-(2**31), 2**31 - 1),
    "Int64": (-(2**63), 2**63 - 1),
    "UInt8": (0, 2**8 - 1),
    "UInt16": (0, 2**16 - 1),
    "UInt32": (0, 2**32 - 1),
    "UInt64": (0, 2**64 - 1),
}
# Floating-point kinds; a marker must have an exact value at the column's precision.
FLOAT_KINDS = ("Float32", "Float64")


def split_missing_markers(
    thresholds: dict | None,
) -> tuple[dict | None, dict[str, tuple[Marker, ...]]]:
    """Return Core drift settings without markers, and the validated marker policy.

    Args:
        thresholds: The parsed ``monitoring_drift_thresholds`` object, or None.

    Returns:
        The settings for Core drift validation and the markers per column, each a sorted tuple
        of distinct JSON numbers kept as int or float. Without markers, the settings are
        returned unchanged.

    Raises:
        ValueError: If the marker policy is not a nonempty object of finite-number lists.
    """
    if type(thresholds) is not dict or MARKERS_KEY not in thresholds:
        return thresholds, {}
    settings = {name: value for name, value in thresholds.items() if name != MARKERS_KEY}
    return settings, _validated_markers(thresholds[MARKERS_KEY])


def _validated_markers(value: Any) -> dict[str, tuple[Marker, ...]]:
    """Reject ambiguous marker policy before any data access."""
    if type(value) is not dict or not value:
        raise ValueError("missing_markers must be a nonempty object of column marker lists.")
    return {column: _column_markers(column, values) for column, values in value.items()}


def _column_markers(column: Any, values: Any) -> tuple[Marker, ...]:
    """Keep one column's distinct markers sorted, without changing their number type."""
    if type(column) is not str or not column:
        raise ValueError("missing_markers column names must be nonempty strings.")
    if type(values) is not list or not 1 <= len(values) <= MAX_MARKERS_PER_COLUMN:
        raise ValueError(
            f"missing_markers for {column} must list 1 to {MAX_MARKERS_PER_COLUMN} values."
        )
    numbers: list[Marker] = [item for item in values if _valid_number(item)]
    if len(numbers) != len(values):
        raise ValueError(f"missing_markers for {column} must be finite numbers.")
    return tuple(sorted(set(numbers)))


def _valid_number(item: Any) -> bool:
    """Accept finite floats and 64-bit integers; avoid float conversion of huge integers."""
    if type(item) is int:
        return -(2**63) <= item <= 2**64 - 1
    return type(item) is float and math.isfinite(item)


def typed_markers(column: str, values: tuple[Marker, ...], kind: str | None) -> list[Marker]:
    """Convert markers to the column's dtype exactly, or fail instead of matching a neighbour.

    Args:
        column: The feature column, for error messages.
        values: Validated markers for that column.
        kind: A ``FLOAT_KINDS`` or ``INTEGER_BOUNDS`` key, or None for any other dtype.

    Returns:
        The markers as Python values of the column's kind.

    Raises:
        ValueError: If the dtype is not integer or floating point, or a marker has no exact
            value in it.
    """
    if kind in FLOAT_KINDS:
        return [_exact_float(column, value, kind) for value in values]
    if kind not in INTEGER_BOUNDS:
        raise numeric_marker_error(column)
    low, high = INTEGER_BOUNDS[kind]
    if any(value != int(value) or not low <= int(value) <= high for value in values):
        raise ValueError(f"missing_markers for {column} must be whole numbers within {kind}.")
    return [int(value) for value in values]


def _exact_float(column: str, value: Marker, kind: str) -> float:
    """Reject a marker that the column's float precision would round to a neighbour.

    A Float32 column therefore accepts -200 or -999.5 but not 0.1 or 16777217.
    """
    converted = float(value)
    exact = type(value) is not int or int(converted) == value
    if exact and kind == "Float32":
        try:
            exact = struct.unpack("f", struct.pack("f", converted))[0] == converted
        except OverflowError:
            exact = False
    if not exact:
        raise ValueError(f"missing_markers for {column} has no exact {kind} value.")
    return converted


def require_marked_features(
    markers: dict[str, tuple[Marker, ...]], feature_columns: tuple[str, ...]
) -> None:
    """Fail on a marker for a column that monitoring does not measure, such as a typo."""
    unknown = sorted(set(markers) - set(feature_columns))
    if unknown:
        raise ValueError(
            f"missing_markers name columns that are not monitored features: {unknown}."
        )


def numeric_marker_error(column: str) -> ValueError:
    """Explain that markers need an integer or floating-point column."""
    return ValueError(
        f"missing_markers apply only to integer or floating-point feature columns: {column}."
    )
