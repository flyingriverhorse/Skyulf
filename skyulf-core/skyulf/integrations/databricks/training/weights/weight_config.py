"""Capture trusted training-only weight declarations without changing model parameters."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .....inference.project_code import project_source_digest
from ...scoring.shared.prediction_output import IDENTIFIER_PATTERN

WEIGHT_FIELDS = {
    "weight_column",
    "reserved_weight_columns",
    "weights_python_source",
    "weights_python_sha256",
}
_DELTA_METADATA = {"_change_type", "_commit_version", "_commit_timestamp"}


def _weight_column(value: Any) -> str | None:
    """Require an optional simple source identifier, excluding Delta metadata."""
    if value is None:
        return None
    if type(value) is not str or not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError("weight_column must be None or a simple nonempty column identifier.")
    if value.casefold() in _DELTA_METADATA:
        raise ValueError("weight_column cannot use reserved Delta change metadata names.")
    return value


def capture_model_weights(source: str, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze a model-file declaration without executing or reopening its source."""
    if "weight_column" not in settings:
        return {}
    column = _weight_column(settings["weight_column"])
    return {
        "weight_column": column,
        "reserved_weight_columns": [] if column is None else [column],
        "weights_python_source": source,
        "weights_python_sha256": project_source_digest(source),
    }


def capture_branch_weights(source: str, entries: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Reserve the union of branch weights while preserving each branch's opt-out."""
    captured = {
        name: capture_model_weights(source, entry.get("workflow", {}))
        for name, entry in entries.items()
        if isinstance(entry, dict) and isinstance(entry.get("workflow", {}), dict)
    }
    reserved = list(
        {
            column.casefold(): column
            for settings in captured.values()
            for column in settings.get("reserved_weight_columns", [])
        }.values()
    )
    return {
        name: {**deepcopy(captured.get(name, {})), "reserved_weight_columns": reserved.copy()}
        if reserved
        else deepcopy(captured.get(name, {}))
        for name in entries
    }


def validate_weight_roles(config: Mapping[str, Any]) -> None:
    """Protect declared weights using workflow or LocalTrainingSpec column names."""
    active = _weight_column(config.get("weight_column"))
    reserved = config.get("reserved_weight_columns", [])
    if not isinstance(reserved, (list, tuple)):
        raise ValueError("reserved_weight_columns must be a list of weight column identifiers.")
    columns = [_weight_column(column) for column in reserved]
    if None in columns:
        raise ValueError("reserved_weight_columns cannot contain None.")
    weights = {column.casefold() for column in columns if column is not None}
    if active is not None:
        weights.add(active.casefold())
    if not weights:
        return
    roles = _source_roles(config)
    if weights.intersection(roles):
        raise ValueError(
            "Weight columns must be distinct from input, target, key, group and time roles."
        )


def _source_roles(config: Mapping[str, Any]) -> set[str]:
    """Collect explicit roles without replacing ordinary workflow validation."""
    columns = list(config.get("input_columns", [])) + list(config.get("record_key_columns", []))
    columns.extend(
        config.get(field)
        for field in (
            "target_column",
            "event_column",
            "result_available_at_column",
            "cv_group_column",
            "group_column",
        )
    )
    return {column.casefold() for column in columns if isinstance(column, str)}
