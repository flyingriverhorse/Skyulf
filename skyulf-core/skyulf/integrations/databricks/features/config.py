"""Validate feature data domains independently of model and training settings."""

import re
from dataclasses import dataclass, replace
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
    # Optional src/features/<module>.py:<function> run on the pinned base before joins.
    base_transform: str | None = None
    # materialize (default) | fail: sources without Delta history (Lakeflow tables, views)
    # are copied into a pinned snapshot, or rejected.
    source_snapshot: str = "materialize"
    source_snapshot_schema: str | None = None

    @property
    def snapshot_schema(self) -> str:
        """Return where materialized source snapshots go; default is the output schema."""
        return self.source_snapshot_schema or self.output_table.rsplit(".", 1)[0]

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


# The same target bindings as training.yml; the feature job passes them as task parameters.
TARGET_PLACEHOLDERS = (
    "catalog",
    "input_schema",
    "output_schema",
    "metadata_schema",
    "resource_suffix",
)
_PLACEHOLDER = re.compile(r"\{(" + "|".join(TARGET_PLACEHOLDERS) + r")\}")


def _unbound_sample(value: Any) -> Any:
    """Stand in a valid identifier for each placeholder so names validate before deployment."""
    if type(value) is not str:
        return value
    return _PLACEHOLDER.sub(lambda match: "" if match[1] == "resource_suffix" else "x", value)


def _feature_table(value: Any) -> None:
    """Require canonical catalog.schema.table identity for source/write isolation."""
    sample = _unbound_sample(value)
    table_name(sample)
    if len(sample.split(".")) != 3:
        raise ValueError("Feature table names must contain catalog.schema.table.")


def bind_feature_plan(plan: FeaturePlan, bindings: dict[str, str]) -> FeaturePlan:
    """Resolve target placeholders in every table name before any read or write."""

    def bind(value: str) -> str:
        """Replace each known placeholder with the active target's value."""

        def resolve(match: re.Match) -> str:
            """Fail clearly when an older feature graph passes no binding."""
            if match[1] not in bindings:
                raise ValueError(
                    f"Feature table placeholder {{{match[1]}}} has no value; "
                    "run src/tools/refresh_feature_graph.py and redeploy."
                )
            return bindings[match[1]]

        return _PLACEHOLDER.sub(resolve, value)

    groups = tuple(
        replace(group, source_table=bind(group.source_table), output_table=bind(group.output_table))
        for group in plan.groups
    )
    schema = plan.source_snapshot_schema
    bound = replace(
        plan,
        base_table=bind(plan.base_table),
        output_table=bind(plan.output_table),
        groups=groups,
        source_snapshot_schema=None if schema is None else bind(schema),
    )
    if bound != plan:
        validate_feature_plan(bound)  # bound names still need three parts and distinct outputs
    return bound


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


def _snapshot_settings(data: dict) -> tuple[str, str | None]:
    """Validate the snapshot policy and optional catalog.schema for snapshot tables."""
    policy = data.get("source_snapshot", "materialize")
    if policy not in {"fail", "materialize"}:
        raise ValueError("source_snapshot must be fail or materialize.")
    schema = data.get("source_snapshot_schema")
    if schema is not None:
        if type(schema) is not str or len(schema.split(".")) != 2:
            raise ValueError("source_snapshot_schema must be catalog.schema.")
        table_name(f"{_unbound_sample(schema)}.probe")
    return policy, schema


def _base_transform(value: Any) -> str | None:
    """Accept an optional project function that derives labels and keys from the base."""
    if value is None:
        return None
    if type(value) is not str or not re.fullmatch(
        r"src/features/groups/[A-Za-z_]\w*\.py:[A-Za-z_]\w*", value
    ):
        # groups/ holds upstream Spark code and is kept out of model packages.
        raise ValueError("base_transform must be src/features/groups/<module>.py:<function>.")
    return value


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
    optional = {"base_transform", "source_snapshot", "source_snapshot_schema"}
    allowed = required | {"base_table", "output_table", "keys", "timestamp"} | optional
    data = _fields(value, allowed, required, "features")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("Feature config version must be 1.")
    groups = data["groups"]
    if type(groups) is not dict or len(groups) > 20:
        raise ValueError("Feature groups must be a mapping of at most 20 domains.")
    if not groups and data.get("base_transform") is None:
        if set(data) != required:
            raise ValueError("Disabled features must contain only version and empty groups.")
        return data
    _fields(data, allowed, allowed - optional, "features")
    return data


def parse_feature_config(value: Any) -> FeaturePlan | None:
    """Return no plan for ready-table projects; otherwise require a complete graph."""
    data = _config_header(value)
    if not data["groups"] and data.get("base_transform") is None:
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
        _base_transform(data.get("base_transform")),
        *_snapshot_settings(data),
    )
    _validate_domains(plan)
    return plan


def _validate_plan_containers(plan: FeaturePlan) -> None:
    """Require immutable column and domain lists before serializing a direct plan."""
    if type(plan.keys) is not tuple:
        raise ValueError("Feature plan keys must be an immutable tuple.")
    if type(plan.groups) is not tuple or not (plan.groups or plan.base_transform):
        raise ValueError("Feature plan needs groups or a base_transform, as an immutable tuple.")
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
