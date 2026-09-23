"""Offline configuration and preflight for bounded local-engine jobs.

The SDK does not submit jobs or read Unity Catalog tables. The separate
local_batch adapter uses the declared table snapshot and period.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlsplit

import pandas as pd
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...inference._manifest import ColumnSpec, label_dtype
from ...inference.bundle import InferenceBundle, load_bundle, predict_local
from ...inference.local_pipeline import (
    LocalPipelineArtifact,
    load_local_pipeline,
    predict_local_pipeline,
)
from ..mlflow.registry import (
    RegistryAccessError,
    RegistryDependencyError,
    RegistryError,
    RegistryModelNotFoundError,
    ResolvedModel,
)
from ._contracts import table_name


class InputSource(BaseModel):
    """Select caller-owned rows or describe a pinned UC table read."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    kind: Literal["caller_frame", "uc_table"]
    table: str | None = None
    version: int | None = None
    read_mode: Literal["snapshot", "incremental"] = "snapshot"
    max_rows: int = Field(default=100_000, gt=0)
    max_bytes: int = Field(default=128 * 1024 * 1024, gt=0)


class ModelSelection(BaseModel):
    """Choose one artifact contract and either a path or pinned registry selector."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    kind: Literal["local_pipeline", "portable_bundle"]
    path: str | None = None
    name: str | None = None
    alias: str | None = None
    version: str | None = None
    tracking_uri: str | None = None
    registry_uri: str | None = None

    @field_validator("tracking_uri", "registry_uri")
    @classmethod
    def reject_embedded_credentials(cls, value: str | None) -> str | None:
        """Keep credentials in the supported runtime provider, outside config."""
        if value is not None:
            parsed = urlsplit(value)
            query = {key.lower() for key, _ in parse_qsl(parsed.query)}
            if (
                parsed.username
                or parsed.password
                or parsed.fragment
                or any(
                    marker in key
                    for key in query
                    for marker in ("token", "password", "secret", "credential", "api_key")
                )
            ):
                raise ValueError("Store URI must not embed credentials.")
        return value

    @model_validator(mode="after")
    def validate_selector(self) -> ModelSelection:
        """Require exactly one local path or one registry alias/version selector."""
        if self.path is not None:
            if not self.path.strip() or any(
                value is not None
                for value in (
                    self.name,
                    self.alias,
                    self.version,
                    self.tracking_uri,
                    self.registry_uri,
                )
            ):
                raise ValueError("A local path cannot be combined with registry settings.")
        elif (
            not self.name or not self.name.strip() or (self.alias is None) == (self.version is None)
        ):
            raise ValueError("A registry model needs a name and exactly one alias or version.")
        if self.alias is not None and not self.alias.strip():
            raise ValueError("alias must be non-empty.")
        if self.version is not None and (
            not self.version.isascii() or not self.version.isdigit() or int(self.version) <= 0
        ):
            raise ValueError("version must be a concrete positive integer string.")
        return self


class OutputSink(BaseModel):
    """Select returned predictions or a precreated UC Delta target."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    kind: Literal["return_frame", "uc_delta"]
    table: str | None = None


class LocalWorkflowConfig(BaseModel):
    """Immutable, serializable decisions for one local-engine scoring job."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    runtime: Literal["local", "databricks", "spark"]
    engine: Literal["pandas", "polars"]
    source: InputSource
    model: ModelSelection
    sink: OutputSink


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    """Identify one incompatibility and an actionable repair."""

    code: str
    category: Literal["config", "source", "node", "model", "runtime", "sink"]
    message: str
    fix: str
    phase: Literal["local", "remote"] = "local"


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Keep local evidence separate from optional read-only registry evidence."""

    issues: tuple[PreflightIssue, ...]
    remote_checked: bool
    model_version: str | None = None
    model_digest: str | None = None
    feature_order: tuple[str, ...] = ()
    output_schema: tuple[ColumnSpec, ...] = ()

    @property
    def ready(self) -> bool:
        """Permit prediction only after metadata and every compatibility check pass."""
        return not self.issues and self.model_digest is not None

    @property
    def local_issues(self) -> tuple[PreflightIssue, ...]:
        """Expose checks that did not need a registry or workspace read."""
        return tuple(issue for issue in self.issues if issue.phase == "local")

    @property
    def remote_issues(self) -> tuple[PreflightIssue, ...]:
        """Expose failures from an explicitly requested read-only model lookup."""
        return tuple(issue for issue in self.issues if issue.phase == "remote")

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Expose output names in the saved model's column order."""
        return tuple(column.name for column in self.output_schema)


class PreflightError(ValueError):
    """The selected local workflow cannot safely start."""

    def __init__(self, result: PreflightResult) -> None:
        """Retain structured issues for CLI or job diagnostics."""
        self.result = result
        super().__init__("; ".join(f"{issue.code}: {issue.fix}" for issue in result.issues))


def _config_issues(config: LocalWorkflowConfig) -> list[PreflightIssue]:
    """Reject invalid runtime, source and sink choices before artifact I/O."""
    issues: list[PreflightIssue] = []
    if config.runtime == "spark":
        issues.append(
            PreflightIssue(
                "runtime_unsupported",
                "runtime",
                "Spark is not a local engine runtime.",
                "Choose local or databricks; use the separate Spark batch runner for Spark inference.",
            )
        )
    if config.source.kind == "uc_table":
        if (
            not config.source.table
            or (
                config.source.read_mode == "snapshot"
                and (config.source.version is None or config.source.version < 0)
            )
            or (config.source.read_mode == "incremental" and config.source.version is not None)
        ):
            issues.append(
                PreflightIssue(
                    "source_unbounded",
                    "source",
                    "Snapshot reads need a fixed nonnegative version; incremental reads derive it.",
                    "Set a version for snapshot reads or omit it for incremental reads.",
                )
            )
    elif (
        config.source.table is not None
        or config.source.version is not None
        or config.source.read_mode != "snapshot"
    ):
        issues.append(
            PreflightIssue(
                "source_conflict",
                "source",
                "A caller frame cannot also select a UC table or version.",
                "Remove table and version from caller_frame input.",
            )
        )
    if config.sink.kind == "uc_delta":
        if config.runtime != "databricks":
            issues.append(
                PreflightIssue(
                    "sink_runtime_mismatch",
                    "sink",
                    "UC Delta publication requires the Databricks runtime.",
                    "Choose runtime='databricks' for this sink.",
                )
            )
        if config.source.kind != "uc_table":
            issues.append(
                PreflightIssue(
                    "sink_source_mismatch",
                    "sink",
                    "UC Delta publication requires a pinned UC source.",
                    "Choose an existing UC source table and concrete Delta version.",
                )
            )
        try:
            if config.sink.table is None or len(config.sink.table.split(".")) != 3:
                raise ValueError("A three-part UC target is required.")
            table_name(config.sink.table)
        except ValueError:
            issues.append(
                PreflightIssue(
                    "sink_invalid_target",
                    "sink",
                    "UC Delta publication requires a three-part target table.",
                    "Specify an existing catalog.schema.table target.",
                )
            )
        if config.sink.table and config.sink.table.lower() == (config.source.table or "").lower():
            issues.append(
                PreflightIssue(
                    "sink_source_conflict",
                    "sink",
                    "Source and target tables must be different.",
                    "Choose a separate prediction target table.",
                )
            )
    elif config.sink.table is not None:
        issues.append(
            PreflightIssue(
                "sink_conflict",
                "sink",
                "return_frame does not write a table.",
                "Remove the sink table or select a supported writer later.",
            )
        )
    return issues


def preflight_local(
    config: LocalWorkflowConfig,
    *,
    artifact: LocalPipelineArtifact | InferenceBundle | None = None,
    resolved: ResolvedModel | None = None,
    probe_frame: pd.DataFrame | pl.DataFrame | None = None,
) -> PreflightResult:
    """Check selected contracts without network, pickle loading or job submission.

    Pass an already validated artifact for full local checks. An optional small
    probe exercises real fitted FE/model prediction before job submission.
    Registry selection remains unresolved until explicit preparation.
    """
    if not isinstance(config, LocalWorkflowConfig):
        raise TypeError("config must be a LocalWorkflowConfig.")
    issues = _config_issues(config)
    remote_checked = resolved is not None
    if config.model.name is not None and resolved is None:
        issues.append(
            PreflightIssue(
                "model_unresolved",
                "model",
                "Registry identity and artifact metadata have not been checked.",
                "Call prepare_local_workflow to resolve the alias once and load its pinned version.",
            )
        )
    if resolved is not None and (
        config.model.name != resolved.name
        or (config.model.version is not None and config.model.version != resolved.version)
        or resolved.model_uri != f"models:/{resolved.name}/{resolved.version}"
    ):
        issues.append(
            PreflightIssue(
                "model_reference_mismatch",
                "model",
                "Resolved model differs from the selected concrete identity.",
                "Resolve the configured name and selector again before this job.",
                "remote",
            )
        )
    if artifact is None:
        issues.append(
            PreflightIssue(
                "artifact_unchecked",
                "model",
                "Artifact metadata is not available for preflight.",
                "Load a trusted artifact and rerun preflight.",
            )
        )
        return PreflightResult(tuple(issues), remote_checked)
    if config.model.kind == "local_pipeline" and isinstance(artifact, LocalPipelineArtifact):
        manifest = artifact.manifest
        digest = manifest.pipeline_sha256
        feature_order = manifest.feature_columns
        label = "float64" if manifest.task == "regression" else label_dtype(manifest.classes)
        output = (
            ColumnSpec(name="prediction", dtype=label),
            *(
                ColumnSpec(name=f"probability_{i}", dtype="float64")
                for i in range(len(manifest.classes))
            ),
        )
        if config.engine != manifest.fitted_engine:
            issues.append(
                PreflightIssue(
                    "engine_mismatch",
                    "runtime",
                    "Selected engine differs from the fitted local pipeline.",
                    f"Choose engine='{manifest.fitted_engine}' or refit the pipeline.",
                )
            )
        if manifest.execution_scope != "whole_frame_local":
            issues.append(
                PreflightIssue(
                    "scope_unsupported",
                    "model",
                    "Local package is not eligible for whole-frame local scoring.",
                    "Use a package fitted for whole_frame_local.",
                )
            )
        if artifact.pipeline.preprocessing_steps != artifact.pipeline.feature_engineer.steps_config:
            issues.append(
                PreflightIssue(
                    "node_contract_mismatch",
                    "node",
                    "Pipeline FE configuration differs from its fitted transformer.",
                    "Reload the original fitted artifact instead of changing its steps.",
                )
            )
        estimator = artifact.pipeline.model_estimator
        if estimator is None or estimator.model is None:
            actual_model_class = None
        else:
            model = estimator._unwrap_tuned_model()
            actual_model_class = f"{type(model).__module__}.{type(model).__qualname__}"
        if actual_model_class != manifest.model_class:
            issues.append(
                PreflightIssue(
                    "model_contract_mismatch",
                    "model",
                    "Artifact model class differs from the fitted estimator.",
                    "Reload a complete fitted artifact without changing its metadata.",
                )
            )
    elif config.model.kind == "portable_bundle" and isinstance(artifact, InferenceBundle):
        manifest = artifact.manifest
        digest = artifact.semantic_digest
        feature_order = artifact.feature_order
        output = manifest.output_schema
    else:
        issues.append(
            PreflightIssue(
                "artifact_kind_mismatch",
                "model",
                "Selected artifact kind differs from the loaded package.",
                "Choose the matching local_pipeline or portable_bundle contract.",
            )
        )
        return PreflightResult(tuple(issues), remote_checked)
    if resolved is not None and resolved.digest != digest:
        issues.append(
            PreflightIssue(
                "model_digest_mismatch",
                "model",
                "Resolved registry digest differs from loaded artifact.",
                "Reload the selected concrete version and verify its package.",
                "remote",
            )
        )
    if probe_frame is not None and not issues:
        try:
            _check_frame_budget(probe_frame, config.source)
            if isinstance(artifact, LocalPipelineArtifact):
                predict_local_pipeline(probe_frame, artifact)
            else:
                predict_local(probe_frame, artifact)
        except (TypeError, ValueError, RuntimeError) as exc:
            issues.append(
                PreflightIssue(
                    "prediction_probe_failed",
                    "model",
                    str(exc),
                    "Fix the sample schema or fitted pipeline; replay a representative batch.",
                )
            )
    return PreflightResult(
        tuple(issues),
        remote_checked,
        resolved.version if resolved else None,
        digest,
        feature_order,
        output,
    )


@dataclass(frozen=True, slots=True)
class PreparedLocalWorkflow:
    """A verified local predictor bound to one loaded artifact and config."""

    config: LocalWorkflowConfig
    artifact: LocalPipelineArtifact | InferenceBundle
    preflight: PreflightResult

    def predict(self, frame: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        """Score only a bounded caller-owned batch with the selected contract."""
        _check_frame_budget(frame, self.config.source)
        if isinstance(self.artifact, LocalPipelineArtifact):
            return predict_local_pipeline(frame, self.artifact)
        return predict_local(frame, self.artifact)


def _check_frame_budget(frame: pd.DataFrame | pl.DataFrame, source: InputSource) -> None:
    """Bound driver rows and in-memory frame bytes before any prediction."""
    if not isinstance(frame, pd.DataFrame | pl.DataFrame):
        raise TypeError("Local workflow requires a pandas or Polars DataFrame.")
    if len(frame) > source.max_rows:
        raise ValueError("Input exceeds max_rows.")
    size = (
        int(frame.memory_usage(index=True, deep=True).sum())
        if isinstance(frame, pd.DataFrame)
        else frame.estimated_size()
    )
    if size > source.max_bytes:
        raise ValueError("Input exceeds max_bytes.")


def prepare_local_workflow(
    config: LocalWorkflowConfig, *, probe_frame: pd.DataFrame | pl.DataFrame | None = None
) -> PreparedLocalWorkflow:
    """Load a trusted path or read a registry alias once, then run full preflight.

    Registry access is read-only. Loading a package deserializes trusted pickle;
    call this only for artifacts from a trusted producer. No job is submitted.
    """
    if not isinstance(config, LocalWorkflowConfig):
        raise TypeError("config must be a LocalWorkflowConfig.")
    issues = _config_issues(config)
    if issues:
        raise PreflightError(PreflightResult(tuple(issues), False))
    selection = config.model
    resolved = None
    if selection.path is not None:
        path = Path(selection.path)
        try:
            artifact = (
                load_local_pipeline(path)
                if selection.kind == "local_pipeline"
                else load_bundle(path)
            )
        except (OSError, ValueError) as exc:
            issue = PreflightIssue(
                "artifact_invalid",
                "model",
                str(exc),
                "Select a trusted, complete artifact of the declared kind and compatible runtime.",
            )
            raise PreflightError(PreflightResult((issue,), False)) from exc
    else:
        from ..mlflow.registry import (  # noqa: PLC0415 - optional MLflow client boundary
            load_registered_bundle,
            load_registered_local_pipeline,
            resolve_model,
        )

        if selection.name is None:
            raise ValueError("Registry model name is missing.")
        try:
            resolved = resolve_model(
                selection.name,
                alias=selection.alias,
                version=selection.version,
                tracking_uri=selection.tracking_uri,
                registry_uri=selection.registry_uri,
            )
            loader = (
                load_registered_local_pipeline
                if selection.kind == "local_pipeline"
                else load_registered_bundle
            )
            artifact = loader(
                resolved, tracking_uri=selection.tracking_uri, registry_uri=selection.registry_uri
            )
        except (RegistryError, OSError, ValueError) as exc:
            if isinstance(exc, RegistryAccessError):
                code, fix = (
                    "registry_access_denied",
                    "Grant model read/EXECUTE to the job identity.",
                )
            elif isinstance(exc, RegistryModelNotFoundError):
                code, fix = "registry_model_missing", "Check the model name and alias or version."
            elif isinstance(exc, RegistryDependencyError):
                code, fix = "mlflow_unavailable", "Install the optional MLflow extra."
            else:
                code, fix = "registry_or_artifact_invalid", "Check the trusted package metadata."
            issue = PreflightIssue(code, "model", str(exc), fix, "remote")
            result = PreflightResult(
                (issue,), True, resolved.version if resolved is not None else None
            )
            raise PreflightError(result) from exc
    result = preflight_local(config, artifact=artifact, resolved=resolved, probe_frame=probe_frame)
    if not result.ready:
        raise PreflightError(result)
    return PreparedLocalWorkflow(config, artifact, result)
