"""Validate feature data domains independently of model and training settings."""

import re
from dataclasses import dataclass
from typing import Any

from ..shared._contracts import column_name, table_name


@dataclass(frozen=True)
class FeatureGroup:
    """One persisted feature domain and its frozen observation lookup rule."""

    name: str
    source_table: str
    output_table: str
    transform: str
    columns: tuple[str, ...]
    lookup: str = "exact"
    allow_missing: bool = False


@dataclass(frozen=True)
class FeaturePlan:
    """Build named feature domains before joining them to unique observations."""

    base_table: str
    output_table: str
    keys: tuple[str, ...]
    timestamp: str
    groups: tuple[FeatureGroup, ...]

    @property
    def record_keys(self) -> tuple[str, ...]:
        """Return the entity and time grain shared by all feature tables."""
        return (*self.keys, self.timestamp)


def _fields(value: Any, allowed: set[str], required: set[str], label: str) -> dict:
    """Reject unknown or incomplete declarations before any data access."""
    if type(value) is not dict:
        raise ValueError(f"{label} must be a mapping.")
    if set(value) - allowed:
        unknown = sorted(str(key) for key in set(value) - allowed)
        raise ValueError(f"Unknown {label} fields: {unknown}.")
    if required - set(value):
        raise ValueError(f"Missing {label} fields: {sorted(required - set(value))}.")
    return value


def _columns(value: Any, label: str) -> tuple[str, ...]:
    """Require a nonempty, unique list of simple, case-insensitive column names."""
    if type(value) is not list or not value:
        raise ValueError(f"{label} must be a nonempty list.")
    for name in value:
        column_name(name)
    if len({name.lower() for name in value}) != len(value):
        raise ValueError(f"{label} must contain distinct columns.")
    return tuple(value)


def _feature_table(value: Any) -> None:
    """Require canonical catalog.schema.table identity for source/write isolation."""
    table_name(value)
    if len(value.split(".")) != 3:
        raise ValueError("Feature table names must contain catalog.schema.table.")


def _group(name: str, value: Any) -> FeatureGroup:
    """Normalize one Spark transform without executing project code."""
    if type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_]{0,49}", name):
        raise ValueError("Feature group names need 1-50 lowercase identifier characters.")
    required = {"source_table", "output_table", "transform", "columns"}
    data = _fields(value, required | {"lookup", "allow_missing"}, required, f"group {name}")
    for field in ("source_table", "output_table"):
        _feature_table(data[field])
    transform = data["transform"]
    if type(transform) is not str or not re.fullmatch(
        r"src/features/groups/[A-Za-z_]\w*\.py:[A-Za-z_]\w*", transform
    ):
        raise ValueError("Feature transform must be src/features/groups/module.py:function.")
    lookup = data.get("lookup", "exact")
    if type(lookup) is not str or lookup not in {"exact", "asof"}:
        raise ValueError("Feature lookup must be exact or asof.")
    missing = data.get("allow_missing", False)
    if type(missing) is not bool:
        raise ValueError("Feature allow_missing must be boolean.")
    return FeatureGroup(
        name,
        data["source_table"],
        data["output_table"],
        transform,
        _columns(data["columns"], f"{name}.columns"),
        lookup,
        missing,
    )


def _validate_domains(plan: FeaturePlan) -> None:
    """Keep writes away from inputs and reject overlapping feature ownership."""
    inputs = {plan.base_table.lower(), *(g.source_table.lower() for g in plan.groups)}
    outputs = [plan.output_table.lower(), *(g.output_table.lower() for g in plan.groups)]
    if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
        raise ValueError(
            "Feature output tables must be distinct and separate from all input tables."
        )
    seen = {name.lower() for name in plan.record_keys}
    for group in plan.groups:
        names = {name.lower() for name in group.columns}
        if names.intersection(seen):
            raise ValueError(f"Feature columns overlap another group or a key: {group.name}.")
        seen.update(names)


def _config_header(value: Any) -> dict:
    """Validate the version and require no dormant settings for disabled features."""
    required = {"version", "groups"}
    allowed = required | {"base_table", "output_table", "keys", "timestamp"}
    data = _fields(value, allowed, required, "features")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Feature config version must be 1.")
    groups = data["groups"]
    if type(groups) is not dict or len(groups) > 20:
        raise ValueError("Feature groups must be a mapping of at most 20 domains.")
    if not groups:
        if set(data) != required:
            raise ValueError("Disabled features must contain only version and empty groups.")
        return data
    _fields(data, allowed, allowed, "features")
    return data


def parse_feature_config(value: Any) -> FeaturePlan | None:
    """Return no plan for ready-table projects; otherwise require a complete graph."""
    data = _config_header(value)
    if not data["groups"]:
        return None
    for name in ("base_table", "output_table"):
        _feature_table(data[name])
    keys = _columns(data["keys"], "keys")
    column_name(data["timestamp"])
    if data["timestamp"].lower() in {name.lower() for name in keys}:
        raise ValueError("Feature timestamp must be separate from entity keys.")
    plan = FeaturePlan(
        data["base_table"],
        data["output_table"],
        keys,
        data["timestamp"],
        tuple(_group(name, group) for name, group in data["groups"].items()),
    )
    _validate_domains(plan)
    return plan


def _validate_plan_containers(plan: FeaturePlan) -> None:
    """Require immutable column and domain lists before serializing a direct plan."""
    if type(plan.keys) is not tuple:
        raise ValueError("Feature plan keys must be an immutable tuple.")
    if type(plan.groups) is not tuple or not plan.groups:
        raise ValueError("Feature plan groups must be a nonempty immutable tuple.")
    if any(type(group.columns) is not tuple for group in plan.groups):
        raise ValueError("Feature group columns must be an immutable tuple.")


def validate_feature_plan(plan: FeaturePlan) -> None:
    """Apply the same configuration checks to programmatically constructed plans."""
    from dataclasses import asdict  # noqa: PLC0415

    _validate_plan_containers(plan)
    data = asdict(plan)
    data["keys"] = list(plan.keys)
    data["groups"] = {
        group.name: {key: value for key, value in asdict(group).items() if key != "name"}
        for group in plan.groups
    }
    if len(data["groups"]) != len(plan.groups):
        raise ValueError("Feature group names must be distinct.")
    for group in data["groups"].values():
        group["columns"] = list(group["columns"])
    parse_feature_config({"version": 1, **data})
