"""Explicit, dependency-free contracts for optional Unity Catalog lookups."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from ..shared._contracts import column_name, table_name


def _columns(value: tuple[str, ...], field: str, *, allow_empty: bool = False) -> None:
    """Require immutable, distinct column names instead of wildcard selection."""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{field} must be a tuple of explicit column names.")
    for name in value:
        column_name(name)
    if len({name.casefold() for name in value}) != len(value):
        raise ValueError(f"{field} must contain distinct column names.")


def _uc_name(value: str) -> None:
    """Require a catalog-qualified name for this Unity Catalog-only adapter."""
    table_name(value)
    if len(value.split(".")) != 3:
        raise ValueError("Unity Catalog names must contain catalog.schema.name.")


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureLookupSpec:
    """Select named features and optionally retrieve their historical values.

    ``lookup_key`` contains source columns in feature-table primary-key order,
    excluding the table's TIMESERIES key. The SDK validates table metadata and
    key type compatibility. ``timestamp_type`` pins the source time column's
    Spark type across training and scoring. Time-series tables must already
    have a TIMESERIES primary key; merely storing a timestamp is insufficient.
    """

    table_name: str
    lookup_key: tuple[str, ...]
    feature_names: tuple[str, ...]
    timestamp_lookup_key: str | None = None
    timestamp_type: Literal["timestamp", "date"] = "timestamp"
    lookback_window: timedelta | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous output names and invalid temporal lookup settings."""
        _uc_name(self.table_name)
        _columns(self.lookup_key, "lookup_key")
        _columns(self.feature_names, "feature_names")
        if self.timestamp_lookup_key is not None:
            column_name(self.timestamp_lookup_key)
        _columns((*self.lookup_key, *self.feature_names, *self.timestamp_columns), "lookup")
        if self.timestamp_type not in ("timestamp", "date"):
            raise ValueError("timestamp_type must be timestamp or date.")
        self._validate_lookback()

    @property
    def timestamp_columns(self) -> tuple[str, ...]:
        """Return the optional source time column without a nullable element."""
        return () if self.timestamp_lookup_key is None else (self.timestamp_lookup_key,)

    def _validate_lookback(self) -> None:
        """Keep temporal bounds nonnegative and tied to an as-of lookup."""
        if self.lookback_window is None:
            return
        if not isinstance(self.lookback_window, timedelta):
            raise TypeError("lookback_window must be datetime.timedelta.")
        if self.timestamp_lookup_key is None or self.lookback_window < timedelta(0):
            raise ValueError("lookback_window requires a timestamp lookup and must be nonnegative.")


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureTrainingSpec:
    """Describe lookup lineage and columns excluded from model inputs.

    Feature names are explicit; selecting all columns with ``None`` is not
    supported. Labels cannot participate in any lookup, and overlapping feature
    outputs require upstream renaming into distinct feature tables/columns.
    """

    lookups: tuple[FeatureLookupSpec, ...]
    label: str | None
    exclude_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate output identity and reject direct label leakage."""
        if not isinstance(self.lookups, tuple) or not self.lookups:
            raise ValueError("lookups must be a nonempty tuple of FeatureLookupSpec.")
        if any(not isinstance(item, FeatureLookupSpec) for item in self.lookups):
            raise TypeError("lookups must contain FeatureLookupSpec values.")
        _columns(self.exclude_columns, "exclude_columns", allow_empty=True)
        _columns(self.feature_names, "feature_names")
        self._validate_label()
        self._validate_lookup_columns()
        self._validate_table_windows()

    def _validate_table_windows(self) -> None:
        """Reject windows the SDK's single per-table lookback field cannot represent."""
        windows: dict[str, timedelta | None] = {}
        for lookup in self.lookups:
            table = lookup.table_name.casefold()
            if table in windows and windows[table] != lookup.lookback_window:
                raise ValueError("Lookups of the same table must use one lookback_window.")
            windows[table] = lookup.lookback_window

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Keep feature output order stable across the configured lookups."""
        return tuple(name for lookup in self.lookups for name in lookup.feature_names)

    def _validate_label(self) -> None:
        """Prevent the target from being fetched, used as a key, or excluded."""
        if self.label is None:
            return
        column_name(self.label)
        inputs = set(self.exclude_columns) | set(self.feature_names)
        for lookup in self.lookups:
            inputs.update((*lookup.lookup_key, *lookup.timestamp_columns))
        if self.label.casefold() in {name.casefold() for name in inputs}:
            raise ValueError(
                "label cannot be a feature, lookup key, timestamp, or excluded column."
            )

    def _validate_lookup_columns(self) -> None:
        """Prevent lookup outputs from shadowing another lookup's input keys."""
        features = {name.casefold() for name in self.feature_names}
        temporal_types: dict[str, str] = {}
        for lookup in self.lookups:
            inputs = (*lookup.lookup_key, *lookup.timestamp_columns)
            if features.intersection(name.casefold() for name in inputs):
                raise ValueError("feature names cannot overlap lookup input columns.")
            for name in lookup.timestamp_columns:
                previous = temporal_types.setdefault(name.casefold(), lookup.timestamp_type)
                if previous != lookup.timestamp_type:
                    raise ValueError("Shared timestamp lookup columns must use the same type.")
