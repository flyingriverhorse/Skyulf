"""Resolve pipeline column rules once from training data, never from inference requests."""

import re
from typing import Any

import pandas as pd

from ..core.schema import SkyulfSchema
from ..data.dataset import SplitDataset
from ..registry import NodeRegistry
from ..utils import unpack_pipeline_input
from ._helpers import (
    auto_detect_datetime_columns,
    auto_detect_numeric_columns,
    auto_detect_text_columns,
)

_SELECTORS = frozenset({"columns_regex", "dtype", "exclude_columns"})
_DTYPES = ("numeric", "categorical", "datetime", "boolean")


def uses_column_selection(config: dict[str, Any]) -> bool:
    """Identify explicit pipeline rules without interpreting their values."""
    return bool(_SELECTORS.intersection(config))


def _names(value: Any, field: str) -> list[str]:
    """Validate a configured name sequence without turning a string into characters."""
    if not isinstance(value, list) or any(not isinstance(name, str) for name in value):
        raise ValueError(f"{field} must be a list of column names.")
    return list(dict.fromkeys(value))


def _pattern(value: Any) -> re.Pattern[str] | None:
    """Compile an explicit regular expression and report its configuration field."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("columns_regex must be a regular expression string.")
    try:
        return re.compile(value)
    except re.error as exc:
        raise ValueError(f"Invalid columns_regex: {exc}") from exc


def _schema_kind(label: str, kind: str) -> bool:
    """Classify only dtype labels whose category is known before seeing values."""
    label = label.lower()
    prefixes = {
        "numeric": ("int", "uint", "float", "decimal"),
        "categorical": ("string", "str", "utf8", "category", "categorical", "enum"),
        "datetime": ("date", "timestamp"),
        "boolean": ("bool",),
    }
    if label in ("unknown", "object"):
        raise ValueError("Column selection dtype needs runtime data for an unknown/object schema.")
    return label.startswith(prefixes[kind])


def _dtype_columns(frame: Any, kind: Any, candidates: list[str]) -> set[str]:
    """Use native dtype discovery at fit and conservative metadata during preview."""
    if kind is None:
        return set(candidates)
    if not isinstance(kind, str) or kind not in _DTYPES:
        raise ValueError(f"Column selection dtype must be one of {', '.join(_DTYPES)}.")
    if isinstance(frame, SkyulfSchema):
        return {
            name for name in candidates if _schema_kind(frame.dtypes.get(name, "unknown"), kind)
        }
    detectors = {
        "numeric": _numeric_columns,
        "categorical": auto_detect_text_columns,
        "datetime": auto_detect_datetime_columns,
    }
    if kind in detectors:
        return set(detectors[kind](frame))
    schema = SkyulfSchema.from_dataframe(frame)
    return {name for name in candidates if schema.dtypes.get(name, "").lower().startswith("bool")}


def _numeric_columns(frame: Any) -> list[str]:
    """Select real numeric features while retaining pandas Decimal support."""
    columns = auto_detect_numeric_columns(frame)
    if isinstance(frame, pd.DataFrame):
        columns = [
            name
            for name in columns
            if not pd.api.types.is_timedelta64_dtype(frame[name])
            and not pd.api.types.is_complex_dtype(frame[name])
        ]
    return columns


def _selection_frame(data: Any) -> Any:
    """Use only the training features when a pipeline carries split or paired data."""
    frame = data.train if isinstance(data, SplitDataset) else data
    frame = frame if isinstance(frame, SkyulfSchema) else unpack_pipeline_input(frame)[0]
    return frame.to_native() if hasattr(frame, "to_native") else frame


def _candidate_columns(frame: Any, config: dict[str, Any]) -> list[str]:
    """Preserve explicit order while protecting implicit target selection."""
    explicit = config.get("columns")
    columns = _names(
        list(frame.columns) if explicit is None or explicit == [] else explicit, "columns"
    )
    candidates = [name for name in columns if name in frame.columns]
    if not config.get("columns"):
        candidates = [name for name in candidates if name != config.get("target_column")]
    return candidates


def _filter_columns(frame: Any, config: dict[str, Any]) -> list[str]:
    """Intersect configured name and dtype filters without consulting scoring data."""
    candidates = _candidate_columns(frame, config)
    pattern = _pattern(config.get("columns_regex"))
    if pattern is not None:
        candidates = [name for name in candidates if pattern.search(name)]
    excluded = set(_names(config.get("exclude_columns", []), "exclude_columns"))
    candidates = [name for name in candidates if name not in excluded]
    eligible = _dtype_columns(frame, config.get("dtype"), candidates)
    return [name for name in candidates if name in eligible]


def resolve_column_selection(data: Any, config: dict[str, Any], transformer: str) -> dict[str, Any]:
    """Replace pipeline selectors with concrete columns selected only from training data.

    Nonempty explicit columns constrain the candidates; otherwise candidates are
    input columns except the target. Regex and dtype are intersections, followed
    by exclusions. No matches raise rather than triggering a node's auto-detection.
    Existing configurations without these options are returned unchanged. Direct
    Calculator users can call this helper before fitting; Appliers never call it.
    """
    if not uses_column_selection(config):
        return config
    metadata = NodeRegistry.get_all_metadata().get(transformer, {})
    if "columns" not in metadata.get("params", {}):
        raise ValueError(f"{transformer} does not accept pipeline column selection rules.")
    selected = _filter_columns(_selection_frame(data), config)
    if not selected:
        raise ValueError(f"{transformer} column selection matched no columns.")
    return with_resolved_columns(config, selected)


def with_resolved_columns(config: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """Replace only selectors and column names, retaining every other recipe parameter."""
    selected = _names(columns, "columns")
    if not selected:
        raise ValueError("Resolved column selection must not be empty.")
    return {
        **{key: value for key, value in config.items() if key not in _SELECTORS},
        "columns": selected,
    }
