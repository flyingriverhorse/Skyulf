"""Sequential, independent target training over one pinned Delta source.

Plans contain trusted executable project source when custom steps are present.
Replaying a plan starts fresh runs and registrations; it never retries a possibly
completed registration or promises exactly-once publication. No aliases change.
"""

import hashlib
import json
import re
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ...inference.project_code import load_project_module
from ..mlflow.promotion import controlled_champion_version
from ..mlflow.tracking import TrackingConfig, TrackingRun, track_run
from .local_cv import LocalCVSpec
from .local_retraining import (
    LocalCandidateResult,
    LocalTrainingSpec,
    candidate_config,
    train_local_candidate,
    training_spec_payload,
    validate_cv_holdout_policy,
    validate_pre_split_step,
)
from .local_workflow import resolve_training_spec, training_settings, training_spec
from .prediction_output import TABLE_NAME_PATTERN
from .weight_config import validate_weight_roles
from .workflow_config import validate_workflow_config


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingBranch:
    """Bind one target to its immutable source, model and independent evaluation policy."""

    name: str
    spec: LocalTrainingSpec
    pipeline: dict[str, Any]
    model_name: str
    metric: str
    min_improvement: float = 0.0
    engine: Literal["pandas", "polars"] = "pandas"
    champion_version: str | None = None
    quality_threshold: float | None = None
    quality_gates: dict[str, float] | None = None
    risk_category: str | None = None
    cv: LocalCVSpec = LocalCVSpec()
    tracking_uri: str | None = None
    registry_uri: str | None = None


@dataclass(frozen=True, slots=True)
class BranchTrainingResult:
    """Expose only a fully completed set of immutable branch candidates."""

    parent_run_id: str
    source_table: str
    source_version: int
    components: dict[str, LocalCandidateResult]
    plan_sha256: str


def _branch_name(name: Any) -> None:
    """Require a portable identifier usable as an artifact directory name."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name):
        raise ValueError("Branch names must be simple identifiers of at most 128 characters.")
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
    if name.casefold() in reserved:
        raise ValueError("Branch names must be portable path-safe identifiers.")


def _unique(values: list[str], label: str) -> None:
    """Reject case-insensitive collisions in identifiers shared with external systems."""
    if len({value.casefold() for value in values}) != len(values):
        raise ValueError(f"Training branches require distinct {label}.")


def _validate_weight_roles(branches: tuple[TrainingBranch, ...]) -> None:
    """Protect every declared weight against every branch role during frozen replay."""
    reserved = {
        column
        for branch in branches
        for column in (*branch.spec.reserved_weight_columns, branch.spec.weight_column)
        if column is not None
    }
    for branch in branches:
        validate_weight_roles({**asdict(branch.spec), "reserved_weight_columns": tuple(reserved)})
        for index, step in enumerate(branch.spec.pre_split_steps):
            validate_pre_split_step(step, index, branch.spec.target_column, tuple(reserved))


def _validate_shared(branches: tuple[TrainingBranch, ...]) -> None:
    """Require distinct targets and models over exactly one source and key contract."""
    _unique([branch.name for branch in branches], "names")
    _unique([branch.model_name for branch in branches], "registered model names")
    _unique([branch.spec.target_column for branch in branches], "targets")
    _validate_weight_roles(branches)
    first = branches[0].spec
    targets = {branch.spec.target_column.casefold() for branch in branches}
    for branch in branches:
        spec = branch.spec
        if (spec.table, spec.version, spec.record_key_columns) != (
            first.table,
            first.version,
            first.record_key_columns,
        ):
            raise ValueError(
                "Training branches must share the exact table, version and record keys."
            )
        siblings = targets - {spec.target_column.casefold()}
        if any(column.casefold() in siblings for column in spec.source_columns):
            raise ValueError("Branch features and source dependencies must exclude other targets.")


def _validate_branch(branch: TrainingBranch) -> None:
    """Run all single-candidate checks without opening readers or creating runs."""
    _branch_name(branch.name)
    if not isinstance(branch.model_name, str) or not TABLE_NAME_PATTERN.fullmatch(
        branch.model_name
    ):
        raise ValueError("Branch model_name must be a resolved three-part Unity Catalog name.")
    candidate_config(
        branch.spec,
        branch.pipeline,
        engine=branch.engine,
        cv=branch.cv,
        metric=branch.metric,
        min_improvement=branch.min_improvement,
        champion_version=branch.champion_version,
        quality_threshold=branch.quality_threshold,
        quality_gates=branch.quality_gates,
        risk_category=branch.risk_category,
    )
    validate_cv_holdout_policy(branch.spec, branch.cv)
    if not branch.spec.drop_missing_labels:
        raise ValueError("Training branches require drop_missing_labels=True.")
    for field in ("tracking_uri", "registry_uri"):
        value = getattr(branch, field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Branch {field} must be nonempty text or null.")


def _validated_branches(branches: Any) -> tuple[TrainingBranch, ...]:
    """Copy caller-owned recipes and validate the entire collection before side effects."""
    if not isinstance(branches, (list, tuple)) or not branches:
        raise ValueError("Training branches must be a nonempty sequence.")
    if any(not isinstance(branch, TrainingBranch) for branch in branches):
        raise TypeError("Expected TrainingBranch entries.")
    copied = deepcopy(tuple(branches))
    for branch in copied:
        _validate_branch(branch)
    _validate_shared(copied)
    return tuple(sorted(copied, key=lambda branch: branch.name))


def _from_config(name: str, config: dict[str, Any], now: datetime) -> TrainingBranch:
    """Translate validated workflow policy without resolving any remote identities."""
    return TrainingBranch(
        name=name,
        spec=replace(training_spec(training_settings(config, now)), drop_missing_labels=True),
        pipeline=deepcopy(config["pipeline"]),
        model_name=config["model_name"],
        metric=config["metric"],
        min_improvement=config["min_improvement"],
        engine=config["engine"],
        champion_version=config.get("champion_version"),
        quality_threshold=config.get("quality_threshold"),
        quality_gates=config.get("quality_gates"),
        risk_category=config.get("risk_category"),
        cv=LocalCVSpec.from_workflow(config),
        tracking_uri=config["tracking_uri"],
        registry_uri=config["registry_uri"],
    )


def _checked_configs(configs: Any) -> dict[str, dict[str, Any]]:
    """Validate all workflow policies and a common explicit source version offline."""
    if not isinstance(configs, dict) or not configs:
        raise ValueError("Training configs must be a nonempty mapping of branch names.")
    for name in configs:
        _branch_name(name)
    checked = {}
    for name in sorted(configs):
        config = validate_workflow_config(configs[name], action="train")
        if config["promotion_policy"] != "manual_approval" or config["score_handoff"] != "disabled":
            raise ValueError(
                "Training branches require manual_approval and disabled score_handoff."
            )
        checked[name] = config
    _shared_endpoints(checked)
    _pin_explicit_version(checked)
    return checked


def _pin_explicit_version(checked: dict[str, dict[str, Any]]) -> None:
    """Apply one explicit source snapshot to unspecified sibling configurations."""
    versions = {
        config["training_version"]
        for config in checked.values()
        if config.get("training_version") is not None
    }
    if len(versions) > 1:
        raise ValueError("Training branches must share the same explicit source version.")
    if versions:
        for config in checked.values():
            config["training_version"] = next(iter(versions))


def _shared_endpoints(configs: dict[str, dict[str, Any]]) -> None:
    """Prevent champion pins from being resolved against different registry services."""
    for field, default in (("tracking_uri", "databricks"), ("registry_uri", "databricks-uc")):
        values = {_endpoint(config.get(field), default, field) for config in configs.values()}
        if len(values) != 1:
            raise ValueError(f"Training branches must share the same {field}.")
        for config in configs.values():
            config[field] = next(iter(values))


def _endpoint(value: Any, default: str, field: str) -> str:
    """Default only omitted endpoints, rejecting false, blank and malformed overrides."""
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text or null.")
    return value


def _resolve_champion(branch: TrainingBranch, config: dict[str, Any]) -> TrainingBranch:
    """Capture the branch's champion for comparison without mutating registry aliases."""
    champion = controlled_champion_version(
        branch.model_name,
        tracking_uri=config.get("tracking_uri", "databricks"),
        registry_uri=config.get("registry_uri", "databricks-uc"),
    )
    if branch.champion_version is not None and branch.champion_version != champion:
        raise ValueError(
            f"Branch {branch.name} champion_version does not match the current champion."
        )
    return replace(branch, champion_version=champion)


def prepare_training_branches(
    spark: Any,
    configs: dict[str, dict[str, Any]],
    *,
    now: datetime | None = None,
    champion_versions: dict[str, str | None] | None = None,
) -> tuple[TrainingBranch, ...]:
    """Validate every branch offline, then pin one source version and each champion.

    Distinct labels train sequentially with independent eligible populations and
    holdouts. Same-target model competition belongs in a separate workflow.
    """
    instant = datetime.now(UTC) if now is None else now
    if not isinstance(instant, datetime) or instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("Training needs a timezone-aware run instant.")
    instant = instant.astimezone(UTC)
    checked = _checked_configs(configs)
    branches = _validated_branches(
        tuple(_from_config(name, config, instant) for name, config in checked.items())
    )
    branch_training_payload(branches)
    resolved = resolve_training_spec(spark, checked[branches[0].name], instant)
    pinned = tuple(
        replace(branch, spec=replace(branch.spec, version=resolved.version)) for branch in branches
    )
    if champion_versions is not None:
        return _set_champion_pins(pinned, champion_versions)
    return tuple(_resolve_champion(branch, checked[branch.name]) for branch in pinned)


def _set_champion_pins(
    branches: tuple[TrainingBranch, ...], versions: dict
) -> tuple[TrainingBranch, ...]:
    """Use explicitly captured set counterparts without reading component aliases."""
    if set(versions) != {branch.name for branch in branches}:
        raise ValueError("Model set champion pins must identify every training branch.")
    for branch in branches:
        if branch.champion_version is not None and branch.champion_version != versions[branch.name]:
            raise ValueError(
                f"Branch {branch.name} champion_version differs from model-set baseline."
            )
    return _validated_branches(
        tuple(replace(branch, champion_version=versions[branch.name]) for branch in branches)
    )


def _branch_payload(branch: TrainingBranch) -> dict[str, Any]:
    """Serialize every branch decision, including concrete UTC selection boundaries."""
    payload = asdict(branch)
    payload["spec"] = training_spec_payload(branch.spec, branch.engine)
    payload["spec"].pop("engine")
    return payload


def branch_training_payload(branches: tuple[TrainingBranch, ...]) -> dict[str, Any]:
    """Produce a strict JSON plan for a fresh-run replay of trusted training recipes."""
    checked = _validated_branches(branches)
    payload = {
        "format_version": 1,
        "source_table": checked[0].spec.table,
        "source_version": checked[0].spec.version,
        "branches": [_branch_payload(branch) for branch in checked],
    }
    return json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))


def _exact_fields(value: Any, expected: set[str], label: str) -> None:
    """Reject missing and unknown saved fields rather than silently changing a replay."""
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Invalid {label} fields.")


def _restore_branch(payload: Any) -> TrainingBranch:
    """Restore saved project registrations before validating its pinned spec."""
    _exact_fields(payload, {field.name for field in fields(TrainingBranch)}, "training branch")
    values = deepcopy(payload)
    _exact_fields(
        values["spec"], {field.name for field in fields(LocalTrainingSpec)}, "branch spec"
    )
    _exact_fields(values["cv"], {field.name for field in fields(LocalCVSpec)}, "branch cv")
    if not isinstance(values["pipeline"], dict):
        raise ValueError("Branch pipeline must be an object.")
    source = values["pipeline"].get("project_python_source")
    if source is not None:
        module = load_project_module(source)
        for name in ("build_preprocessing", "build_pre_split_steps"):
            factory = getattr(module, name, None)
            if factory is not None:
                factory()
    values["spec"] = LocalTrainingSpec.from_payload(values["spec"])
    values["cv"] = LocalCVSpec(**values["cv"])
    return TrainingBranch(**values)


def restore_training_branches(payload: dict[str, Any]) -> tuple[TrainingBranch, ...]:
    """Restore a trusted saved plan without consulting current Delta versions or aliases."""
    _exact_fields(payload, {"format_version", "source_table", "source_version", "branches"}, "plan")
    if type(payload["format_version"]) is not int or payload["format_version"] != 1:
        raise ValueError("Training plan format_version must be 1.")
    if not isinstance(payload["branches"], list) or not payload["branches"]:
        raise ValueError("Training plan branches must be a nonempty list.")
    branches = _validated_branches(tuple(_restore_branch(value) for value in payload["branches"]))
    spec = branches[0].spec
    if (payload["source_table"], payload["source_version"]) != (spec.table, spec.version):
        raise ValueError("Training plan source differs from its branch specs.")
    if type(payload["source_version"]) is not int:
        raise ValueError("Training plan source_version must be an integer.")
    return branches


def _artifact_paths(root: str | Path, branches: tuple[TrainingBranch, ...]) -> dict[str, Path]:
    """Resolve every artifact destination inside the caller's root before opening a run."""
    directory = Path(root).resolve()
    paths = {branch.name: (directory / branch.name).resolve() for branch in branches}
    if len(set(paths.values())) != len(paths):
        raise ValueError("Branch artifact paths must resolve to distinct directories.")
    if any(not path.is_relative_to(directory) or path == directory for path in paths.values()):
        raise ValueError("Branch artifact paths must remain inside artifact_path.")
    return paths


def _validate_run_endpoints(
    branches: tuple[TrainingBranch, ...], tracking_uri: str, registry_uri: str
) -> None:
    """Keep saved model versions bound to the services that resolved their identities."""
    for field, supplied in (("tracking_uri", tracking_uri), ("registry_uri", registry_uri)):
        if not isinstance(supplied, str) or not supplied.strip():
            raise ValueError(f"{field} must be nonempty text.")
        for branch in branches:
            pinned = getattr(branch, field)
            if pinned is not None and pinned != supplied:
                raise ValueError(f"Branch {branch.name} {field} differs from its saved endpoint.")


def log_progress(
    run: TrackingRun,
    completed: dict[str, LocalCandidateResult],
    *,
    status: str,
    failed_branch: str | None = None,
) -> None:
    """Retain completed immutable candidates even when a later branch fails."""
    run.client.log_dict(
        run.run_id,
        {
            "status": status,
            "failed_branch": failed_branch,
            "completed": {name: asdict(result) for name, result in completed.items()},
        },
        "branch_training_progress.json",
    )
    run.set_tags({"skyulf.training.status": status})


def train_branch(
    spark: Any,
    branch: TrainingBranch,
    *,
    run: TrackingRun,
    digest: str,
    path: Path,
    tracking_uri: str,
    registry_uri: str,
    experiment_name: str,
    evaluation_charts: dict[str, Any] | None = None,
) -> LocalCandidateResult:
    """Delegate fitting, registration and comparison to the existing candidate service."""
    return train_local_candidate(
        spark,
        branch.spec,
        branch.pipeline,
        model_name=branch.model_name,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        experiment_name=experiment_name,
        run_name=branch.name,
        artifact_path=path,
        metric=branch.metric,
        min_improvement=branch.min_improvement,
        engine=branch.engine,
        champion_version=branch.champion_version,
        quality_threshold=branch.quality_threshold,
        quality_gates=branch.quality_gates,
        risk_category=branch.risk_category,
        cv=branch.cv,
        evaluation_charts=evaluation_charts,
        run_tags={
            "mlflow.parentRunId": str(run.run_id),
            "skyulf.training.branch": branch.name,
            "skyulf.training.plan_sha256": digest,
        },
    )


def train_local_branches(
    spark: Any,
    branches: tuple[TrainingBranch, ...],
    *,
    tracking_uri: str,
    registry_uri: str,
    experiment_name: str,
    artifact_path: str | Path,
    run_name: str = "multi_target_training",
) -> BranchTrainingResult:
    """Train validated branches sequentially under a parent that fails with any child.

    Completed candidates remain inspectable after failure. Only a completely
    successful run emits the result artifact. Replay uses a new artifact directory
    and creates new candidate versions; it never resumes uncertain registrations.
    """
    checked = _validated_branches(branches)
    _validate_run_endpoints(checked, tracking_uri, registry_uri)
    plan = branch_training_payload(checked)
    digest = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    paths = _artifact_paths(artifact_path, checked)
    tracking = TrackingConfig(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        failure_policy="raise",
    )
    with track_run(tracking, run_name=run_name) as run:
        if run.run_id is None:
            raise RuntimeError("MLflow did not provide a parent run ID.")
        run.log_config(plan, artifact_file="branch_training_plan.json")
        run.set_tags({"skyulf.training.plan_sha256": digest})
        completed: dict[str, LocalCandidateResult] = {}
        log_progress(run, completed, status="running")
        for branch in checked:
            try:
                completed[branch.name] = train_branch(
                    spark,
                    branch,
                    run=run,
                    digest=digest,
                    path=paths[branch.name],
                    tracking_uri=tracking_uri,
                    registry_uri=registry_uri,
                    experiment_name=experiment_name,
                )
                log_progress(run, completed, status="running")
            except BaseException:
                with suppress(Exception):
                    log_progress(run, completed, status="failed", failed_branch=branch.name)
                raise
        result = BranchTrainingResult(
            run.run_id, checked[0].spec.table, checked[0].spec.version, completed, digest
        )
        run.client.log_dict(run.run_id, asdict(result), "branch_training_result.json")
        log_progress(run, completed, status="complete")
    return result
