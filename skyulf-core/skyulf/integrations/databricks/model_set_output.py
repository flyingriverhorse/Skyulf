"""Select stored results and named views over one committed model-set table."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ...inference.model_set import ModelSetArtifact
from ...inference.model_set_scoring import model_set_output_schema
from ._contracts import column_name, table_name

METADATA_COLUMNS = ("model_set_name", "model_set_version", "model_set_digest", "run_id")
_VIEW_PROPERTY = "prediction.projection"
_VIEW_OPTIONS = {"model_views", "model_view_template", "combined_view"}


@dataclass(frozen=True)
class PredictionView:
    """Name one consumer projection without creating an independent data copy."""

    name: str
    columns: tuple[str, ...]


def publication_policy(value: Any = None) -> dict[str, Any]:
    """Reject misspelled settings and leave the legacy all-column default intact."""
    policy = {"mode": "all"} if value is None else value
    if not isinstance(policy, dict) or set(policy) - {"mode"} - _VIEW_OPTIONS:
        raise ValueError("Publication requires mode and optional named view settings.")
    mode = policy.get("mode", "all")
    if not isinstance(mode, str) or mode not in {"all", "combined_only", "separate_views"}:
        raise ValueError("Publication mode must be all, combined_only or separate_views.")
    if mode != "separate_views" and _VIEW_OPTIONS.intersection(policy):
        raise ValueError("View settings require separate_views publication mode.")
    _validate_view_options(policy)
    return {**policy, "mode": mode}


def _validate_view_options(policy: dict[str, Any]) -> None:
    """Keep optional explicit branch selections and name templates unambiguous."""
    views = policy.get("model_views")
    if views is not None:
        if not isinstance(views, dict):
            raise ValueError("model_views must map branch names to view names, or be None.")
        for branch, name in views.items():
            column_name(branch)
            if not isinstance(name, str) or not name:
                raise ValueError("model_views values must be nonempty view names.")
    for key in ("model_view_template", "combined_view"):
        value = policy.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{key} must be a nonempty name or None.")


def _combined_columns(artifact: ModelSetArtifact) -> tuple[str, ...]:
    """Keep declared combined values and their independent exclusion outcomes."""
    rules = artifact.manifest.composition_config["outputs"]
    if not rules:
        raise ValueError("combined_only requires saved combined rules; configure and retrain.")
    columns = list(artifact.manifest.record_key_columns)
    for rule in rules:
        columns.extend(column["name"] for column in rule["columns"])
        columns.extend(
            f"{rule['name']}__{field}" for field in ("scoring_status", "exclusion_reason")
        )
    return tuple(columns) + METADATA_COLUMNS


def publication_columns(artifact: ModelSetArtifact, policy: Any = None) -> tuple[str, ...]:
    """Select persisted columns without changing the calculations in the saved artifact."""
    if publication_policy(policy)["mode"] == "combined_only":
        return _combined_columns(artifact)
    return tuple(column.name for column in model_set_output_schema(artifact)) + METADATA_COLUMNS


def publication_views(
    artifact: ModelSetArtifact, target: str, policy: Any = None
) -> tuple[PredictionView, ...]:
    """Resolve explicit or automatic consumer names and reject destination collisions."""
    table_name(target)
    policy = publication_policy(policy)
    if policy["mode"] != "separate_views":
        return ()
    names = _component_view_names(artifact, target, policy)
    all_columns = publication_columns(artifact)
    shared = artifact.manifest.record_key_columns + METADATA_COLUMNS
    components = {component.branch: component for component in artifact.manifest.components}
    views = [
        PredictionView(
            name,
            tuple(c for c in all_columns if c in shared + _component_columns(components[branch])),
        )
        for branch, name in names.items()
    ]
    if artifact.manifest.composition_config["outputs"]:
        name = policy.get("combined_view") or f"{target}_combined"
        views.append(PredictionView(name, _combined_columns(artifact)))
    _validate_view_destinations(views, target)
    return tuple(views)


def _component_columns(component: Any) -> tuple[str, ...]:
    """Select exact component fields without matching similarly prefixed business outputs."""
    names = tuple(column.name for column in component.output_schema)
    names += ("scoring_status", "exclusion_reason")
    return tuple(f"{component.branch}__{name}" for name in names)


def _component_view_names(artifact: ModelSetArtifact, target: str, policy: dict) -> dict:
    """None selects all branches; an explicit mapping selects only the named branches."""
    branches = {component.branch for component in artifact.manifest.components}
    explicit = policy.get("model_views")
    if explicit is not None:
        if set(explicit) - branches:
            raise ValueError("model_views contains unknown model branches.")
        return explicit
    template = policy.get("model_view_template") or target + "_{branch}"
    return {
        component.branch: template.replace("{branch}", component.branch)
        for component in artifact.manifest.components
    }


def _validate_view_destinations(views: list[PredictionView], target: str) -> None:
    """Require unique validated names separate from the physical publication table."""
    used = {target.casefold()}
    for view in views:
        table_name(view.name)
        if view.name.casefold() in used:
            raise ValueError(
                "Prediction view names must be distinct from each other and the table."
            )
        used.add(view.name.casefold())


def _view_digest(view: PredictionView, target: str) -> str:
    """Identify the immutable table projection owned by this publisher."""
    encoded = json.dumps([target.casefold(), view.columns], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _existing_view(spark: Any, view: PredictionView, target: str) -> bool:
    """Never replace an unrelated table, view or changed projection."""
    if not spark.catalog.tableExists(view.name):
        return False
    if spark.catalog.getTable(view.name).tableType.upper() != "VIEW":
        raise ValueError(f"Prediction view name already belongs to a table: {view.name}.")
    row = spark.sql(f"SHOW TBLPROPERTIES {table_name(view.name)} ('{_VIEW_PROPERTY}')").first()
    if row is None or row["value"] != _view_digest(view, target):
        raise ValueError(f"Existing prediction view has another owner or projection: {view.name}.")
    if tuple(spark.table(view.name).columns) != view.columns:
        raise ValueError(f"Existing prediction view columns changed: {view.name}.")
    return True


def provision_publication_views(spark: Any, target: str, views: tuple[PredictionView, ...]) -> None:
    """Prepare all consumer projections before the sole data commit, including on retries.

    DDL setup can leave views over the old snapshot after failure. It never writes
    predictions and never replaces existing objects. Data advances only in the
    common backing table after every view and model calculation succeeds.
    """
    missing = [view for view in views if not _existing_view(spark, view, target)]
    for view in missing:
        columns = ", ".join(column_name(column) for column in view.columns)
        spark.sql(
            f"CREATE VIEW IF NOT EXISTS {table_name(view.name)} "
            f"TBLPROPERTIES ('{_VIEW_PROPERTY}' = '{_view_digest(view, target)}') "
            f"AS SELECT {columns} FROM {table_name(target)}"
        ).collect()
        _existing_view(spark, view, target)
