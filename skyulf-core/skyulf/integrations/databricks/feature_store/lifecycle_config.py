"""Serialize explicit feature lookups and immutable training-table bindings."""

import json
import math
from copy import deepcopy
from datetime import timedelta
from typing import Any

from .config import FeatureLookupSpec, FeatureTrainingSpec, _uc_name


def _mapping(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    """Reject unknown fields before an SDK argument can be silently omitted."""
    if type(value) is not dict or set(value) - fields:
        raise ValueError(f"{label} must contain only {sorted(fields)}.")
    return value


def _names(value: Any, label: str) -> tuple[str, ...]:
    """Require a real JSON list; the dataclass validates names and uniqueness."""
    if type(value) is not list:
        raise ValueError(f"{label} must be a list of column names.")
    return tuple(value)


def _table_name(value: Any) -> str:
    """Validate required table strings before passing them to the typed contract."""
    if not isinstance(value, str):
        raise ValueError("Feature table_name must be a string.")
    _uc_name(value)
    return value


def serialize_feature_spec(spec: FeatureTrainingSpec) -> dict[str, Any]:
    """Return a JSON-compatible contract without serializing an SDK object."""
    if not isinstance(spec, FeatureTrainingSpec):
        raise TypeError("spec must be FeatureTrainingSpec.")
    lookups = [
        {
            "table_name": lookup.table_name,
            "lookup_key": list(lookup.lookup_key),
            "feature_names": list(lookup.feature_names),
            "timestamp_lookup_key": lookup.timestamp_lookup_key,
            "timestamp_type": lookup.timestamp_type,
            "lookback_seconds": (
                None if lookup.lookback_window is None else lookup.lookback_window.total_seconds()
            ),
        }
        for lookup in spec.lookups
    ]
    return {"lookups": lookups, "label": spec.label, "exclude_columns": list(spec.exclude_columns)}


def _lookup(value: Any) -> FeatureLookupSpec:
    """Validate one bounded temporal request before constructing its typed contract."""
    item = _mapping(
        value,
        {
            "table_name",
            "lookup_key",
            "feature_names",
            "timestamp_lookup_key",
            "timestamp_type",
            "lookback_seconds",
        },
        "feature lookup",
    )
    seconds = item.get("lookback_seconds")
    if seconds is not None and (
        type(seconds) not in (int, float)
        or not 0 <= seconds <= 3_153_600_000
        or not math.isfinite(seconds)
    ):
        raise ValueError("lookback_seconds must be finite, nonnegative and at most 100 years.")
    return FeatureLookupSpec(
        table_name=_table_name(item.get("table_name")),
        lookup_key=_names(item.get("lookup_key"), "lookup_key"),
        feature_names=_names(item.get("feature_names"), "feature_names"),
        timestamp_lookup_key=item.get("timestamp_lookup_key"),
        timestamp_type=item.get("timestamp_type", "timestamp"),
        lookback_window=None if seconds is None else timedelta(seconds=seconds),
    )


def deserialize_feature_spec(value: Any) -> FeatureTrainingSpec:
    """Restore only declared feature lineage, never arbitrary Python types."""
    item = _mapping(value, {"lookups", "label", "exclude_columns"}, "lookup_spec")
    lookups = item.get("lookups")
    if type(lookups) is not list or not 1 <= len(lookups) <= 20:
        raise ValueError("lookups must contain 1 to 20 explicit entries.")
    return FeatureTrainingSpec(
        lookups=tuple(_lookup(lookup) for lookup in lookups),
        label=item.get("label"),
        exclude_columns=_names(item.get("exclude_columns", []), "exclude_columns"),
    )


def parse_lookup_config(
    value: Any,
    *,
    label: str | None,
    input_columns: tuple[str, ...],
    exclude_columns: tuple[str, ...] = (),
) -> FeatureTrainingSpec | None:
    """Derive metadata exclusions while keeping workflow lookup settings single-owned."""
    if value is None:
        return None
    config = _mapping(value, {"lookups"}, "feature_lookup")
    spec = deserialize_feature_spec(
        {
            **config,
            "label": label,
            "exclude_columns": list(exclude_columns),
        }
    )
    if not set(spec.feature_names).issubset(input_columns):
        raise ValueError("Feature lookup outputs must be declared in input_columns.")
    return spec


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate fields before decoding a persisted binding."""
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate feature binding key: {name}.")
        result[name] = value
    return result


def _table_snapshot(value: Any) -> dict[str, Any]:
    """Require one concrete Delta identity and nonnegative history version."""
    row = _mapping(value, {"table_name", "table_id", "version"}, "feature table")
    _table_name(row.get("table_name"))
    if type(row.get("version")) is not int or row["version"] < 0:
        raise ValueError("Feature table version must be a nonnegative integer.")
    identity = row.get("table_id")
    if type(identity) is not str or not identity.strip() or len(identity) > 256:
        raise ValueError("Feature table_id must be a nonempty Delta identity.")
    return deepcopy(row)


def _table_snapshots(value: Any, spec: FeatureTrainingSpec) -> list[dict[str, Any]]:
    """Bind every lookup table to exactly one validated snapshot."""
    if type(value) is not list or not value:
        raise ValueError("feature_tables must contain pinned Delta tables.")
    result = [_table_snapshot(table) for table in value]
    names = [row["table_name"] for row in result]
    if len(set(names)) != len(names) or set(names) != {
        lookup.table_name for lookup in spec.lookups
    }:
        raise ValueError("Feature table snapshots must match each lookup table exactly once.")
    return sorted(result, key=lambda item: item["table_name"])


def parse_feature_binding(value: str | dict[str, Any]) -> dict[str, Any]:
    """Validate the complete persisted native feature contract and return a detached copy."""
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 64 * 1024:
            raise ValueError("Feature binding exceeds 64 KiB.")
        try:
            value = json.loads(value, object_pairs_hook=_unique)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("Invalid feature binding JSON.") from exc
    item = _mapping(value, {"version", "lookup_spec", "lookup_evidence"}, "feature binding")
    if type(item.get("version")) is not int or item["version"] != 1:
        raise ValueError("Feature binding version must be 1.")
    spec = deserialize_feature_spec(item.get("lookup_spec"))
    evidence = _mapping(
        item.get("lookup_evidence"), {"policy", "feature_tables"}, "lookup_evidence"
    )
    if evidence.get("policy") != "training_snapshot":
        raise ValueError("Feature lookup requires the training_snapshot policy.")
    return {
        "version": 1,
        "lookup_spec": serialize_feature_spec(spec),
        "lookup_evidence": {
            "policy": "training_snapshot",
            "feature_tables": _table_snapshots(evidence.get("feature_tables"), spec),
        },
    }


def binding_json(value: dict[str, Any]) -> str:
    """Freeze a validated contract for immutable artifact and receipt attributes."""
    result = json.dumps(
        parse_feature_binding(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if len(result.encode("utf-8")) > 64 * 1024:
        raise ValueError("Feature binding exceeds 64 KiB.")
    return result


def workflow_lookup_json(config: dict[str, Any]) -> str | None:
    """Freeze enabled training lookup declarations after target-name resolution."""
    value = config.get("feature_lookup")
    if value is None:
        return None
    if config.get("engine") != "pandas" or config.get("inference_mode") != "spark":
        raise ValueError("feature_lookup requires engine=pandas and inference_mode=spark.")
    spec = parse_lookup_config(
        value, label=config["target_column"], input_columns=tuple(config["input_columns"])
    )
    assert spec is not None
    return json.dumps(serialize_feature_spec(spec), sort_keys=True, separators=(",", ":"))


def bind_lookup_tables(value: Any, bindings: dict[str, str]) -> dict[str, Any]:
    """Expand only declared table identifiers without modifying editable YAML values."""
    config = deepcopy(_mapping(value, {"lookups"}, "feature_lookup"))
    if type(config.get("lookups")) is not list:
        raise ValueError("feature_lookup.lookups must be a list.")
    for lookup in config["lookups"]:
        if type(lookup) is not dict or not isinstance(lookup.get("table_name"), str):
            raise ValueError("feature_lookup table_name must be a string.")
        for key, replacement in bindings.items():
            lookup["table_name"] = lookup["table_name"].replace("{" + key + "}", replacement)
        _uc_name(lookup["table_name"])
    return config
