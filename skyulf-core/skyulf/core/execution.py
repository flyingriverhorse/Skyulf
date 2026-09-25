"""Immutable execution declarations, independent of optional runtime packages."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_ENGINES = ("pandas", "polars", "spark")


def _require_choice(name: str, value: object, choices: tuple[str, ...]) -> None:
    """Reject unknown names without coercing values into supported choices."""
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {choices}; got {value!r}.")


def _require_positive_integer(name: str, value: object) -> None:
    """Reject booleans and implicit numeric coercion for resource limits."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_record_key_columns(keys: tuple[str, ...]) -> None:
    """Require stable, distinct names without accepting a mutable container."""
    if not isinstance(keys, tuple) or not keys:
        raise ValueError("record_key_columns must be a nonempty tuple of column names.")
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("record_key_columns must contain nonempty column names.")
    if len(set(keys)) != len(keys):
        raise ValueError("record_key_columns must contain distinct column names.")


@dataclass(frozen=True)
class FrameSpec:
    """Declare observation identity and an optional target in a single frame.

    This validates names only. Checking key values for nulls or duplicates
    belongs to the runtime that receives the actual dataframe.
    """

    record_key_columns: tuple[str, ...]
    target: str | None = None

    def __post_init__(self) -> None:
        """Validate identity names without reading or modifying any data."""
        _validate_record_key_columns(self.record_key_columns)
        if self.target is not None:
            if not isinstance(self.target, str) or not self.target.strip():
                raise ValueError("target must be a nonempty column name or None.")
            if self.target in self.record_key_columns:
                raise ValueError("target must not also be a record_key_columns column.")


@dataclass(frozen=True)
class ExecutionOptions:
    """Validate an explicit engine and resource budgets without activating it.

    These declarations do not install a runtime or enable node capabilities.
    A Python batch row limit is not a guarantee about its memory footprint.
    """

    engine: str
    state_max_bytes: int = 8 * 1024 * 1024
    python_batch_rows: int = 4096
    model_max_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        """Reject invalid selections and limits before runtime submission."""
        _require_choice("engine", self.engine, _ENGINES)
        for field in ("state_max_bytes", "python_batch_rows", "model_max_bytes"):
            _require_positive_integer(field, getattr(self, field))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ExecutionOptions":
        """Parse options strictly; unknown fields raise TypeError.

        Unlike legacy node configuration, execution configuration never ignores
        misspelled fields. The caller's mapping is not modified.
        """
        return cls(**dict(config))
