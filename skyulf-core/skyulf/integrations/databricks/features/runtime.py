"""Pin feature inputs and code before separately repairable Spark tasks execute."""

import hashlib
import json
from pathlib import Path
from typing import Any

from ..projects._project_files import read_source
from .config import FeatureGroup, FeaturePlan, validate_feature_plan
from .delta import current_version, publish_feature_table, read_version
from .graph import plan_digest
from .joins import join_feature_groups


def selected_groups(plan: FeaturePlan, selection: str) -> set[str]:
    """Resolve a comma-separated subset; an empty selection explicitly reuses all."""
    names = {group.name for group in plan.groups}
    if selection == "*":
        return names
    selected = {name.strip() for name in selection.split(",") if name.strip()}
    if selected - names:
        raise ValueError(f"Unknown selected feature groups: {sorted(selected - names)}.")
    return selected


def transform_source(project: Path, group: FeatureGroup) -> tuple[str, str]:
    """Read a bounded project-contained Spark transform without importing it."""
    relative = group.transform.split(":", 1)[0]
    path = project / relative
    if not path.resolve().is_relative_to(project.resolve()):
        raise ValueError("Feature transform must remain inside the project.")
    if any(
        part.is_symlink() or part.is_junction()
        for part in [path, *path.parents]
        if part != project and project in part.parents
    ):
        raise ValueError("Feature transform must remain inside the project without links.")
    source = read_source(path)
    return source, hashlib.sha256(source.encode()).hexdigest()


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    """Bind group receipts to one immutable input/code snapshot."""
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def initialize_features(
    spark: Any,
    project: Path,
    plan: FeaturePlan,
    selection: str,
) -> dict[str, Any]:
    """Pin all source versions and reused outputs once for the complete job run."""
    validate_feature_plan(plan)
    selected = selected_groups(plan, selection)
    sources = {plan.base_table, *(g.source_table for g in plan.groups if g.name in selected)}
    reused = {
        g.name: current_version(spark, g.output_table)
        for g in plan.groups
        if g.name not in selected
    }
    return {
        "plan_sha256": plan_digest(plan),
        "selected": sorted(selected),
        "source_versions": {table: current_version(spark, table) for table in sorted(sources)},
        "reused_versions": reused,
        "transform_sha256": {g.name: transform_source(project, g)[1] for g in plan.groups},
    }


def _require_snapshot(plan: FeaturePlan, snapshot: dict[str, Any]) -> None:
    """Reject changed graph settings and invalid selection before reading records."""
    validate_feature_plan(plan)
    if snapshot.get("plan_sha256") != plan_digest(plan):
        raise ValueError("Feature configuration changed during this run; start a new run.")
    names = {group.name for group in plan.groups}
    selected = snapshot.get("selected")
    if type(selected) is not list or any(type(name) is not str for name in selected):
        raise ValueError("Feature snapshot selected groups must be a list of names.")
    if len(set(selected)) != len(selected) or set(selected) - names:
        raise ValueError("Feature snapshot has invalid selected groups.")


def build_feature_group(
    spark: Any,
    project: Path,
    plan: FeaturePlan,
    snapshot: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Run one trusted Spark transform or reuse its explicitly pinned Delta version."""
    _require_snapshot(plan, snapshot)
    group = next((g for g in plan.groups if g.name == name), None)
    if group is None:
        raise ValueError(f"Unknown feature group: {name}.")
    source, digest = transform_source(project, group)
    if snapshot["transform_sha256"][name] != digest:
        raise ValueError(f"Feature transform changed during this run: {name}.")
    if name not in snapshot["selected"]:
        receipt = {
            "table": group.output_table,
            "version": snapshot["reused_versions"][name],
            "reused": True,
        }
    else:
        namespace: dict[str, Any] = {"__name__": "skyulf_feature_transform"}
        exec(compile(source, group.transform.split(":")[0], "exec"), namespace)  # noqa: S102
        transform = namespace.get(group.transform.split(":")[1])
        if not callable(transform):
            raise ValueError(f"Feature transform function is missing: {group.transform}.")
        frame = read_version(
            spark, group.source_table, snapshot["source_versions"][group.source_table]
        )
        result = transform(frame)
        expected = {*plan.record_keys, *group.columns}
        if set(result.columns) != expected:
            raise ValueError(
                f"Feature transform must return exactly keys, time and columns: {name}."
            )
        receipt = publish_feature_table(
            spark, result, group.output_table, plan.keys, plan.timestamp
        )
        receipt["reused"] = False
    return {**receipt, "snapshot_sha256": snapshot_digest(snapshot)}


def merge_feature_groups(
    spark: Any,
    plan: FeaturePlan,
    snapshot: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Join the exact group commits from this run and publish its checked observations."""
    _require_snapshot(plan, snapshot)
    if set(receipts) != {g.name for g in plan.groups}:
        raise ValueError("Feature merge requires one receipt from every configured group.")
    frames = {}
    for group in plan.groups:
        receipt = receipts[group.name]
        if receipt.get("table") != group.output_table or receipt.get(
            "snapshot_sha256"
        ) != snapshot_digest(snapshot):
            raise ValueError(f"Feature receipt does not belong to this run: {group.name}.")
        frames[group.name] = read_version(spark, group.output_table, receipt["version"])
    base = read_version(spark, plan.base_table, snapshot["source_versions"][plan.base_table])
    merged = join_feature_groups(base, frames, plan)
    return publish_feature_table(spark, merged, plan.output_table, plan.keys, plan.timestamp)
