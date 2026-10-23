"""Pin feature inputs and code before separately repairable Spark tasks execute."""

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from ..projects._project_files import read_source
from .config import FeatureGroup, FeaturePlan, validate_feature_plan
from .delta import (
    current_version,
    publication_state,
    publish_feature_table,
    read_version,
    require_publication_head,
    validate_output_head,
)
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
    return project_function_source(project, group.transform)


def project_function_source(project: Path, reference: str) -> tuple[str, str]:
    """Read ``path.py:function`` source inside the project and return it with its digest."""
    relative = reference.split(":", 1)[0]
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


def _initialize_publication(spark: Any, plan: FeaturePlan) -> dict[str, Any]:
    """Advance the generation even when only transform code changed."""
    outputs = {
        table: publication_state(spark, table)
        for table in sorted({plan.output_table, *(g.output_table for g in plan.groups)})
    }
    generation = (
        max(
            (
                state["intent"]["generation"]
                for state in outputs.values()
                if state and state["intent"]
            ),
            default=0,
        )
        + 1
    )
    return {
        "generation": generation,
        "snapshot_id": uuid.uuid4().hex,
        "outputs": {
            table: {key: state[key] if state else None for key in ("table_id", "version")}
            for table, state in outputs.items()
        },
    }


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
        "publication": _initialize_publication(spark, plan),
        **(
            {"base_transform_sha256": project_function_source(project, plan.base_transform)[1]}
            if plan.base_transform
            else {}
        ),
    }


def _publication_intent(plan: FeaturePlan, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Refuse legacy or malformed mutation snapshots before touching any output."""
    publication = snapshot.get("publication")
    if type(publication) is not dict or set(publication) != {
        "generation",
        "snapshot_id",
        "outputs",
    }:
        raise ValueError("Feature publication requires a fresh initialized snapshot.")
    identity, outputs = publication["snapshot_id"], publication["outputs"]
    if type(identity) is not str or re.fullmatch(r"[0-9a-f]{32}", identity) is None:
        raise ValueError("Invalid feature publication snapshot identity.")
    tables = {plan.output_table, *(group.output_table for group in plan.groups)}
    if type(outputs) is not dict or set(outputs) != tables:
        raise ValueError("Feature publication output heads do not match the plan.")
    for head in outputs.values():
        validate_output_head(head)
    return {
        "generation": publication["generation"],
        "snapshot_sha256": snapshot_digest(snapshot),
    }


def _check_publication(
    spark: Any, plan: FeaturePlan, snapshot: dict[str, Any], table: str
) -> dict[str, Any]:
    """Fence every output against newer partial runs before either publication phase."""
    intent = _publication_intent(plan, snapshot)
    outputs = snapshot["publication"]["outputs"]
    for output, expected in outputs.items():
        require_publication_head(publication_state(spark, output), expected, intent, output)
    return {**intent, "expected": outputs[table]}


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


def _project_function(source: str, reference: str) -> Any:
    """Load one function from pinned, project-contained source."""
    namespace: dict[str, Any] = {"__name__": "skyulf_feature_transform"}
    # Trusted project code only: contained bounded source and pinned digest checked by callers.
    # This runs with job-owner privileges; it is not a sandbox for untrusted Python.
    # nosemgrep: python.lang.security.audit.exec-detected.exec-detected, python_exec_rule-exec-used
    exec(  # nosec B102 # noqa: S102 # pylint: disable=exec-used
        compile(source, reference.split(":")[0], "exec"), namespace
    )
    function = namespace.get(reference.split(":")[1])
    if not callable(function):
        raise ValueError(f"Feature transform function is missing: {reference}.")
    return function


def _prepared_base(spark: Any, project: Path | None, plan: FeaturePlan, snapshot: dict) -> Any:
    """Read the pinned base and apply the optional base_transform from this run's code."""
    base = read_version(spark, plan.base_table, snapshot["source_versions"][plan.base_table])
    if plan.base_transform is None:
        return base
    if project is None:
        raise ValueError("base_transform requires the project directory.")
    source, digest = project_function_source(project, plan.base_transform)
    if snapshot.get("base_transform_sha256") != digest:
        raise ValueError("base_transform changed during this run; start a new run.")
    return _project_function(source, plan.base_transform)(base)


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
        _check_publication(spark, plan, snapshot, group.output_table)
        transform = _project_function(source, group.transform)
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
            spark,
            result,
            group.output_table,
            plan.keys,
            plan.timestamp,
            publication=_check_publication(spark, plan, snapshot, group.output_table),
        )
        receipt["reused"] = False
    return {**receipt, "snapshot_sha256": snapshot_digest(snapshot)}


def merge_feature_groups(
    spark: Any,
    plan: FeaturePlan,
    snapshot: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
    project: Path | None = None,
) -> dict[str, Any]:
    """Join the exact group commits from this run and publish its checked observations.

    With ``base_transform``, the base passes through that project function first, so
    labels and record keys can be derived from a raw source.
    """
    _require_snapshot(plan, snapshot)
    _check_publication(spark, plan, snapshot, plan.output_table)
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
    base = _prepared_base(spark, project, plan, snapshot)
    merged = join_feature_groups(base, frames, plan)
    return publish_feature_table(
        spark,
        merged,
        plan.output_table,
        plan.keys,
        plan.timestamp,
        publication=_check_publication(spark, plan, snapshot, plan.output_table),
    )
