"""Validated batch identities, independent of Spark or platform connections."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def table_name(value: str) -> str:
    """Validate and quote a simple one-, two- or three-part catalog identifier."""
    if type(value) is not str or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}", value
    ):
        raise ValueError("Table name must contain one to three simple identifiers.")
    return ".".join(f"`{part}`" for part in value.split("."))


def column_name(value: str) -> str:
    """Restrict control columns to simple names outside reserved batch metadata."""
    if type(value) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Control column must be a simple identifier.")
    if value.lower().startswith("__skyulf_"):
        raise ValueError("Control column collides with reserved __skyulf_ metadata.")
    return f"`{value}`"


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchSpec:
    """Pin period, source, model and expected target version for one logical run.

    Aware dates represent instants and are converted to UTC without reinterpreting
    wall-clock values. Construct local calendar boundaries with ``ZoneInfo``.
    ``source_version`` is a Delta integer version, never an arbitrary label.
    The output table must already exist with the intended prediction schema.
    """

    period_start: datetime
    period_end: datetime
    as_of: datetime
    row_keys: tuple[str, ...]
    output_table: str
    model_name: str
    model_version: str
    source_version: int
    code_version: str
    run_id: str
    model_digest: str
    expected_target_version: int
    period_column: str = "event_time"
    business_timezone: str = "UTC"
    mode: str = "native_features"
    publish_mode: str = "replace_period"
    allow_empty: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguity before any Spark action or table modification."""
        for name in ("period_start", "period_end", "as_of"):
            value = getattr(self, name)
            if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware.")
            # Reject nonexistent local times; fold explicitly resolves ambiguous times.
            if value.astimezone(UTC).astimezone(value.tzinfo).replace(tzinfo=None) != value.replace(
                tzinfo=None
            ):
                raise ValueError(f"{name} is not a valid local instant.")
        if self.period_start_utc >= self.period_end_utc:
            raise ValueError("period_start must precede period_end.")
        table_name(self.output_table)
        column_name(self.period_column)
        if type(self.row_keys) is not tuple or not self.row_keys:
            raise ValueError("row_keys must be a nonempty tuple.")
        for key in self.row_keys:
            column_name(key)
        names = [key.lower() for key in self.row_keys]
        if len(set(names)) != len(names) or self.period_column.lower() in names:
            raise ValueError("row_keys and period_column must be distinct.")
        for name in ("model_name", "code_version", "run_id"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip() or len(value) > 512:
                raise ValueError(f"{name} must be a nonempty string of at most 512 characters.")
        if type(self.model_version) is not str or not re.fullmatch(
            r"[1-9][0-9]*", self.model_version
        ):
            raise ValueError("model_version must be a concrete positive version, not an alias.")
        for name in ("source_version", "expected_target_version"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if type(self.model_digest) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", self.model_digest
        ):
            raise ValueError("model_digest must be a SHA-256 bundle digest.")
        if self.mode not in ("native_features", "python_pipeline", "local_pipeline"):
            raise ValueError("mode must be native_features, python_pipeline or local_pipeline.")
        if self.publish_mode != "replace_period":
            raise ValueError("Only replace_period publication is supported.")
        if type(self.allow_empty) is not bool:
            raise TypeError("allow_empty must be a bool.")
        try:
            ZoneInfo(self.business_timezone)
        except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("business_timezone must name an IANA timezone.") from exc

    @property
    def period_start_utc(self) -> datetime:
        """Return the inclusive start instant in UTC."""
        return self.period_start.astimezone(UTC)

    @property
    def period_end_utc(self) -> datetime:
        """Return the exclusive end instant in UTC."""
        return self.period_end.astimezone(UTC)

    @property
    def as_of_utc(self) -> datetime:
        """Return the source availability cutoff in UTC."""
        return self.as_of.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Report a verified Delta commit without retaining distributed data."""

    spec: BatchSpec
    input_count: int
    output_count: int
    commit_version: int
    manifest: dict[str, Any]
    replayed: bool = False
