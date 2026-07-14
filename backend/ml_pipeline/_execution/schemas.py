"""Execution schemas for the ML Pipeline."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobInfo(BaseModel):
    job_id: str
    pipeline_id: str
    node_id: str
    dataset_id: str | None = None
    dataset_name: str | None = None
    job_type: Literal["training", "tuning", "preview", "basic_training", "advanced_tuning"]
    status: JobStatus
    start_time: datetime | None
    end_time: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    logs: list[str] | None = None

    # Extended fields for Experiments Page
    model_type: str | None = None
    hyperparameters: dict[str, Any] | None = None
    created_at: datetime | None = None
    metrics: dict[str, Any] | None = None
    search_strategy: str | None = None
    target_column: str | None = None
    dropped_columns: list[str] | None = None
    version: int | None = None
    graph: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    promoted_at: datetime | None = None

    # Parallel branch metadata
    branch_index: int | None = None
    parent_pipeline_id: str | None = None


from backend.ml_pipeline.constants import StepType  # noqa: E402


def coerce_step_type(value: Any) -> Any:
    """Coerce to ``StepType`` when possible; otherwise return the raw string."""
    if isinstance(value, StepType):
        return value
    try:
        return StepType(value)
    except ValueError:
        return value


@dataclass
class NodeConfig:
    """Configuration for a single pipeline node."""

    node_id: str
    # `Any` so transformer kinds bypass strict StepType coercion.
    step_type: Any
    params: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)  # IDs of upstream nodes


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""

    pipeline_id: str
    nodes: list[NodeConfig]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeExecutionResult:
    """Result of a single node execution."""

    node_id: str
    status: Literal["success", "failed", "skipped"]
    output_artifact_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    execution_time: float = 0.0  # Seconds
    # Carried through so cross-cutting consumers (slow-nodes admin view,
    # debug logs) can identify the node without joining back to the
    # original PipelineConfig. Optional to keep older callsites valid.
    step_type: str | None = None
    # Free-form per-node metadata surfaced to the canvas. `summary` is a
    # short one-line human string the node card renders post-run
    # (e.g. "7,000 / 1,500 / 1,500" for a split, "acc 0.87" for a
    # classifier). Other keys are reserved for future overlays
    # (duration_ms, memory_delta, drift_score, ...).
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineExecutionResult:
    """Result of the entire pipeline execution."""

    pipeline_id: str
    status: Literal["success", "failed", "partial"]
    node_results: dict[str, NodeExecutionResult] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    merge_warnings: list[dict[str, Any]] = field(default_factory=list)
    # Soft, advisory warnings emitted by individual nodes during their
    # `Calculator.fit` / `Applier.apply` (e.g. TargetEncoder coercing a
    # float multiclass target to int, OneHotEncoder seeing a degenerate
    # category). Each entry is `{"node_id": str, "node_type": str,
    # "level": "warning"|"info", "message": str}`. Captured by the
    # `WarningCaptureHandler` attached during `PipelineEngine.run`.
    node_warnings: list[dict[str, Any]] = field(default_factory=list)
    # Pre-execution schema predictions (C7 Phase B). One entry per node;
    # value is ``{"columns": [...], "dtypes": {...}}`` when the node's
    # Calculator overrides ``infer_output_schema`` and an upstream schema
    # was available, ``None`` otherwise. Populated in ``PipelineEngine.run``
    # before the per-node loop runs.
    predicted_schemas: dict[str, dict[str, Any] | None] = field(default_factory=dict)
