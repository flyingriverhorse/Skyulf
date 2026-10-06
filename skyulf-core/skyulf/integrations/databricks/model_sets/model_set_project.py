"""Connect explicit project model sets to frozen packages and serialized Bundle jobs."""

import hashlib
import html
import json
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from skyulf.integrations.mlflow.shared._client import make_registry_client, require_mlflow

from ....inference._manifest import ColumnSpec
from ....inference.model_set import ComponentReference, save_model_set
from ....inference.project_code import load_project_module
from ...mlflow.lifecycle.promotion import ExclusiveAliasWriterAdmission, controlled_champion_version
from ...mlflow.registration.registry import (
    ResolvedModel,
    downloaded_registered_payload,
    load_local_package,
    packaged_artifact_path,
    register_model,
    resolve_model,
)
from ..data.admission import SingleWriterAdmission
from ..jobs.shared.job_output import render_bundle_output, render_scoring_summary
from ..jobs.shared.job_runtime import (
    OPERATOR_FIELDS,
    notebook_output,
    parse_model_version,
    read_notebook_config,
    validate_job_parameters,
)
from ..lifecycle.local_workflow import bind_target_name, resolve_target_config
from ..projects._project_files import project_source, read_source
from ..projects.workflow_config import validate_workflow_config
from ..scoring.incremental.local_incremental import (
    bounded_frame,
    latest_source_version,
    validate_source_change_policy,
)
from ..shared._contracts import input_budget_bytes
from ..training.local_branches import BranchTrainingResult, TrainingBranch, branch_training_payload
from ..training.tuning.local_search import base_model_config
from .model_set_output import publication_policy, publication_views


def load_project_model_set(
    values: dict[str, str], base_config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Resolve optional set destinations without loading training or composition source."""
    path = Path(values["config_path"]).parent.parent / "src/modeling/model_set.py"
    if not path.exists():
        return None
    base = read_notebook_config(values) if base_config is None else base_config
    factory = getattr(load_project_module(read_source(path)), "build_model_set", None)
    if not callable(factory):
        raise ValueError("model_set.py must define build_model_set().")
    settings = factory()
    if settings is None:
        return None
    _validate_set_settings(settings)
    bindings = {
        key: values[key]
        for key in (
            "catalog",
            "input_schema",
            "output_schema",
            "metadata_schema",
            "resource_suffix",
        )
    }
    bound = resolve_target_config({**base, **settings}, bindings)
    _set_policy(bound)
    result = {key: bound[key] for key in settings}
    result["publication"] = _bind_publication(settings.get("publication"), bindings)
    return result


def _validate_set_settings(settings: Any) -> None:
    """Separate editable destinations from rules captured only during training."""
    required = {"model_name", "prediction_table"}
    optional = {
        "composition_config",
        "combined_rules_path",
        "publication",
        "source_change_policy",
        "promotion_policy",
    }
    if not isinstance(settings, dict) or not required.issubset(settings):
        raise ValueError("Model set requires model_name and prediction_table.")
    if set(settings) - required - optional:
        raise ValueError("Unknown model-set setting.")
    if {"composition_config", "combined_rules_path"}.issubset(settings):
        raise ValueError("Choose combined_rules_path or legacy composition_config, not both.")
    validate_source_change_policy(settings.get("source_change_policy", "reject"))


def _bind_publication(value: Any, bindings: dict[str, str]) -> dict[str, Any]:
    """Apply the deployment namespace and suffix to every optional consumer view."""
    policy = publication_policy(value)
    if policy["mode"] != "separate_views":
        return policy

    def bind(name: str) -> str:
        """Keep the branch token while validating the destination namespace."""
        return bind_target_name(
            "prediction_table",
            name.replace("{branch}", "BRANCHTOKEN"),
            bindings,
            bindings.get("resource_suffix", ""),
        ).replace("BRANCHTOKEN", "{branch}")

    for key in ("combined_view", "model_view_template"):
        if policy.get(key) is not None:
            policy[key] = bind(policy[key])
    if policy.get("model_views") is not None:
        policy["model_views"] = {
            branch: bind(name) for branch, name in policy["model_views"].items()
        }
    return policy


def _set_policy(bound: dict[str, Any]) -> None:
    """Allow gated activation and optional handoff to the existing score job."""
    if bound["promotion_policy"] not in {"manual_approval", "automatic"} or bound[
        "score_handoff"
    ] not in {"disabled", "after_alias_change"}:
        raise ValueError(
            "Model sets require manual_approval or automatic and score_handoff "
            "disabled or after_alias_change."
        )


def capture_set_composition(values: dict[str, str]) -> str:
    """Freeze the optional project package only when creating a new set candidate."""
    path = Path(values["config_path"]).parent.parent / "src/composition"
    return project_source(path) if path.exists() else ""


def capture_set_rules(values: dict[str, str], settings: dict) -> tuple[dict, str]:
    """Capture the shared scoring package without consulting it at inference time."""
    if "combined_rules_path" not in settings:
        return {
            **settings,
            "composition_config": settings.get("composition_config", {"outputs": []}),
        }, capture_set_composition(values)
    root = Path(values["config_path"]).parent.parent / "src"
    relative = settings["combined_rules_path"]
    if not isinstance(relative, str) or not relative:
        raise ValueError("combined_rules_path must name a feature package within src.")
    path = (root / "modeling" / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("combined_rules_path must stay within src.")
    source = project_source(path)
    factory = getattr(load_project_module(source), "build_combined_rules", None)
    if not callable(factory):
        raise ValueError("The shared feature package must export build_combined_rules().")
    rules = factory()
    if not isinstance(rules, list):
        raise ValueError("build_combined_rules() must return a list of output rules.")
    return {**settings, "composition_config": {"outputs": rules}}, source


def _record_key_schema(source: Any, keys: tuple[str, ...]) -> tuple[ColumnSpec, ...]:
    """Derive stable key types from the common pinned Spark source schema."""
    supported = {
        "bigint": "int64",
        "int": "int64",
        "smallint": "int64",
        "tinyint": "int64",
        "string": "string",
        "boolean": "bool",
    }
    types = dict(source.dtypes)
    if any(types.get(key) not in supported for key in keys):
        raise ValueError("Model set record keys require integer, string or boolean source columns.")
    return tuple(ColumnSpec(name=key, dtype=supported[types[key]]) for key in keys)


def _completed_parent(
    outcome: BranchTrainingResult,
    branches: tuple[TrainingBranch, ...],
    tracking_uri: str,
    registry_uri: str,
) -> None:
    """Require the exact successful parent receipt before publishing a complete set."""
    if not isinstance(outcome, BranchTrainingResult) or set(outcome.components) != {
        branch.name for branch in branches
    }:
        raise ValueError("A model set requires a complete branch training result.")
    spec = branches[0].spec
    if (outcome.source_table, outcome.source_version) != (spec.table, spec.version):
        raise ValueError("Model-set training result differs from the pinned source.")
    _verify_training_plan(branches, outcome)
    client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
    parent = client.get_run(outcome.parent_run_id)
    if (
        parent.info.status != "FINISHED"
        or parent.data.tags.get("skyulf.training.status") != "complete"
    ):
        raise ValueError("Model set requires a successfully completed parent training run.")
    if parent.data.tags.get("skyulf.training.plan_sha256") != outcome.plan_sha256:
        raise ValueError("Model set parent training plan digest differs.")
    saved_path = client.download_artifacts(outcome.parent_run_id, "branch_training_result.json")
    if json.loads(Path(saved_path).read_text(encoding="utf-8")) != asdict(outcome):
        raise ValueError("Model set requires the exact complete parent result receipt.")


def _verify_training_plan(
    branches: tuple[TrainingBranch, ...], outcome: BranchTrainingResult
) -> None:
    """Bind every source, model and recipe in the packaging request to the saved plan."""
    payload = branch_training_payload(branches)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    if hashlib.sha256(encoded).hexdigest() != outcome.plan_sha256:
        raise ValueError("Model set branches differ from the completed training plan.")
    if any(outcome.components[branch.name].model_name != branch.model_name for branch in branches):
        raise ValueError("Model set component names differ from the completed training plan.")


def _component_directory(resolved: ResolvedModel, tracking_uri: str, registry_uri: str) -> Path:
    """Validate and reuse registered artifact bytes without reserializing fitted pipelines."""
    mlflow = require_mlflow()
    client = make_registry_client(mlflow, tracking_uri, registry_uri)
    if not isinstance(resolved.digest, str):
        raise ValueError("Component needs a resolved digest.")
    with downloaded_registered_payload(
        mlflow, client, resolved, tracking_uri, "local_pipeline"
    ) as (package, model):
        load_local_package(package, model, resolved.digest)
        return packaged_artifact_path(package, model.flavors, "local_pipeline")


def _registered_components(
    outcome: BranchTrainingResult, tracking_uri: str, registry_uri: str
) -> dict[str, tuple[ComponentReference, Path]]:
    """Resolve only concrete candidate versions and require the saved component digest."""
    components = {}
    for branch, result in outcome.components.items():
        resolved = resolve_model(
            result.model_name,
            version=result.model_version,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        if resolved.digest != result.model_digest:
            raise ValueError("Registered component digest differs from training result.")
        reference = ComponentReference(
            name=resolved.name, version=resolved.version, digest=result.model_digest
        )
        components[branch] = (reference, _component_directory(resolved, tracking_uri, registry_uri))
    return components


def _component_tags(
    branches: tuple[TrainingBranch, ...], outcome: BranchTrainingResult
) -> dict[str, str]:
    """Describe each pinned component on the set version without reading mutable aliases."""
    tags = {"model_set_model_count": str(len(branches))}
    for branch in branches:
        result = outcome.components[branch.name]
        prefix = f"model_set_{branch.name}"
        tags[f"{prefix}_version"] = result.model_version
        tags[f"{prefix}_type"] = base_model_config(branch.pipeline)["type"]
        # Validated registry names are ASCII. Keep every UC tag value <= 256 bytes.
        for index, start in enumerate(range(0, len(result.model_name), 256), start=1):
            suffix = "" if index == 1 else f"_{index}"
            tags[f"{prefix}_name{suffix}"] = result.model_name[start : start + 256]
    return tags


def package_training_model_set(
    spark: Any,
    branches: tuple[TrainingBranch, ...],
    outcome: BranchTrainingResult,
    config: dict[str, Any],
    *,
    composition_source: str,
    tracking_uri: str,
    registry_uri: str,
) -> ResolvedModel:
    """Register one frozen candidate from a fully completed multi-target training run."""
    from ...mlflow.models.model_set import log_model_set  # noqa: PLC0415 - optional MLflow boundary

    _completed_parent(outcome, branches, tracking_uri, registry_uri)
    source = spark.read.option("versionAsOf", outcome.source_version).table(outcome.source_table)
    keys = _record_key_schema(source, branches[0].spec.record_key_columns)
    components = _registered_components(outcome, tracking_uri, registry_uri)
    with tempfile.TemporaryDirectory(prefix="skyulf-model-set-") as directory:
        path = Path(directory) / "set"
        saved = save_model_set(
            path,
            components=components,
            record_key_schema=keys,
            composition_source=composition_source,
            composition_config=config["composition_config"],
            quality_evidence={
                "expected_champion_version": config.get("expected_champion_version"),
                "comparisons": {
                    name: result.comparison_sha256 for name, result in outcome.components.items()
                },
            },
        )
        uri = log_model_set(
            path,
            run_id=outcome.parent_run_id,
            artifact_path=f"model_sets/{saved.manifest.set_sha256}",
            tracking_uri=tracking_uri,
        )
        version = register_model(
            uri,
            config["model_name"],
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            tags=_component_tags(branches, outcome),
        )
    return resolve_model(
        config["model_name"],
        version=str(version.version),
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )


def project_endpoints(config: dict[str, Any]) -> dict[str, str]:
    """Use the same default registry services as branch training."""
    return {
        "tracking_uri": config.get("tracking_uri") or "databricks",
        "registry_uri": config.get("registry_uri") or "databricks-uc",
    }


def approval_frame(spark: Any, artifact: Any, config: dict[str, Any]) -> Any:
    """Read a bounded functional probe; Spark checks all keys before sampling.

    Saved holdout quality gates still use their complete training evidence.
    The probe exercises each component; distributed publication scores all rows.
    """
    distributed = config.get("inference_mode", "local") == "spark"
    if distributed:
        from ....inference.model_set_partition_safety import (  # noqa: PLC0415
            require_partition_safe_model_set,
        )

        require_partition_safe_model_set(artifact)
    table = config["score_source_table"]
    source = spark.read.option(
        "versionAsOf", int(latest_source_version(spark, table)["version"])
    ).table(table)
    columns = tuple(column.name for column in artifact.manifest.input_schema)
    keys = artifact.manifest.record_key_columns
    if distributed:
        from ..scoring.batch.spark_scoring import read_distributed_rows  # noqa: PLC0415

        source = read_distributed_rows(source, columns, keys).frame
        source = source.orderBy(*keys).limit(config["max_rows"])
    return bounded_frame(
        source,
        columns,
        keys,
        config["max_rows"],
        input_budget_bytes(config.get("max_input_mb")),
    )


def run_model_set_operator(
    spark: Any,
    values: dict[str, str],
    settings: dict[str, Any],
    options: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Approve or roll back one complete saved set without reading editable recipes."""
    from ...mlflow.lifecycle.model_set_lifecycle import rollback_model_set  # noqa: PLC0415

    action = values["lifecycle_action"]
    config = validate_workflow_config(
        read_notebook_config(values) if config is None else config, action=action
    )
    endpoints = project_endpoints(config)
    admission = ExclusiveAliasWriterAdmission()
    if action == "rollback":
        receipt = options["promotion_receipt"]
        if receipt.model_name != settings["model_name"]:
            raise ValueError("Rollback receipt must belong to the project's model set.")
        result = rollback_model_set(
            receipt,
            expected_current_version=options["expected_champion_version"],
            admission=admission,
            **endpoints,
        )
    elif action in {"approve", "reject"}:
        if options.get("comparison_sha256"):
            raise ValueError(
                "Model-set approval uses saved per-component comparisons; "
                "do not pass a single comparison_sha256."
            )
        resolved = resolve_model(
            settings["model_name"], version=options["candidate_version"], **endpoints
        )
        if action == "reject":
            from ...mlflow.lifecycle.model_set_challenger import reject_model_set  # noqa: PLC0415

            result = reject_model_set(
                resolved,
                reason=options["rejection_reason"],
                expected_champion_version=options["expected_champion_version"],
                admission=admission,
                **endpoints,
            )
            return {
                "action": action,
                "model_set_name": settings["model_name"],
                "receipt": asdict(result),
            }
        from .model_set_release import approve_project_model_set  # noqa: PLC0415

        result = approve_project_model_set(
            spark,
            resolved,
            config,
            expected_champion_version=options["expected_champion_version"],
            policy="manual_approval",
        )
    else:
        raise ValueError("Model sets support train, approve, reject and rollback.")
    return {"action": action, "model_set_name": settings["model_name"], "receipt": asdict(result)}


def render_model_set_result(payload: dict[str, Any]) -> str:
    """Show concrete set identity and explicit operator actions with escaped evidence."""
    if payload.get("recovery_required"):
        return render_bundle_output(payload)
    raw = html.escape(json.dumps(payload, indent=2, default=str, allow_nan=False))
    if "noop" not in payload:
        return "<h2>Model set result</h2><pre>" + raw + "</pre>"
    summary = render_scoring_summary(
        {
            **payload,
            "selected_model_name": payload["model_set_name"],
            "selected_model_version": payload["model_set_version"],
        }
    )
    return (
        "<h2>Model set result</h2>"
        + summary
        + "<details><summary>Technical details (JSON)</summary><pre>"
        + raw
        + "</pre></details>"
    )


def run_model_set_score_notebook(
    spark: Any,
    dbutils: Any,
    *,
    display_html: Callable[[str], Any] | None = None,
    exit_notebook: bool = True,
) -> str:
    """Score one frozen set with a fixed role and one controlled champion lookup."""
    from ..observability.monitoring.monitoring_registration import (  # noqa: PLC0415
        publish_monitoring_request,
        validate_monitoring_settings,
    )
    from ..scoring.incremental.scoring_recovery import run_scoring_step  # noqa: PLC0415

    values = dbutils.widgets.getAll()
    config = validate_workflow_config(read_notebook_config(values), action="score")
    validate_monitoring_settings(values, config)
    payload = run_scoring_step(
        config, dbutils, lambda: score_model_set_payload(spark, config, values)
    )
    publish_monitoring_request(dbutils, payload)
    return notebook_output(
        payload,
        dbutils,
        render=render_model_set_result,
        display_html=display_html,
        exit_notebook=exit_notebook,
    )


def score_model_set_payload(
    spark: Any,
    config: dict[str, Any],
    values: dict[str, str],
    *,
    recovery_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a saved set, reusing the predecessor's concrete pin for CDF recovery."""
    from ...mlflow.models.model_set import load_registered_model_set  # noqa: PLC0415
    from .model_set_batch import run_model_set_batch  # noqa: PLC0415 - optional Spark boundary
    from .monitoring_model_set import (  # noqa: PLC0415
        register_set_monitors,
        validate_component_monitoring,
        validate_set_monitoring,
    )

    parameters = {
        key: value
        for key, value in values.items()
        if key not in OPERATOR_FIELDS | {"lifecycle_action"}
    }
    validate_job_parameters(parameters)
    config = validate_workflow_config(config, action="score")
    settings = load_project_model_set(values, config)
    if settings is None:
        raise ValueError("Multi-target score requires an enabled model set.")
    validate_set_monitoring(values, settings)
    endpoints = project_endpoints(config)
    override = (
        recovery_request["model_version"]
        if recovery_request is not None
        else parameters.get("score_model_version", "")
    )
    version = (
        parse_model_version(override)
        if override
        else controlled_champion_version(settings["model_name"], **endpoints)
    )
    if version is None:
        raise ValueError("Approve a model-set champion before scoring.")
    resolved = resolve_model(settings["model_name"], version=version, **endpoints)
    artifact = load_registered_model_set(resolved, **endpoints)
    validate_component_monitoring(config, values, settings, resolved, artifact)
    result = run_model_set_batch(
        spark,
        resolved,
        artifact,
        source_table=config["score_source_table"],
        prediction_table=settings["prediction_table"],
        admission=SingleWriterAdmission(),
        model_change_mode=config["model_change_mode"],
        max_rows=config["max_rows"],
        max_bytes=input_budget_bytes(config.get("max_input_mb")),
        publication=settings.get("publication"),
        source_change_policy=settings.get("source_change_policy", "reject"),
        inference_mode=config.get("inference_mode", "local"),
        spark_udf_env_manager=config.get("spark_udf_env_manager", "virtualenv"),
        spark_udf_prediction_batch_rows=config.get("spark_udf_prediction_batch_rows", 10_000),
        tracking_uri=endpoints["tracking_uri"],
        registry_uri=endpoints["registry_uri"],
        **({"recovery_request": recovery_request} if recovery_request is not None else {}),
    )
    payload = {
        "model_set_name": resolved.name,
        "model_set_version": resolved.version,
        "model_set_digest": resolved.digest,
        "source_table": config["score_source_table"],
        "prediction_table": settings["prediction_table"],
        "publication": settings.get("publication", {"mode": "all"}),
        "source_change_policy": settings.get("source_change_policy", "reject"),
        "prediction_views": [
            view.name
            for view in publication_views(
                artifact, settings["prediction_table"], settings.get("publication")
            )
        ],
        **asdict(result),
    }
    registration = register_set_monitors(spark, config, values, settings, resolved, artifact)
    if registration is not None:
        payload["monitoring"] = registration
    return payload
