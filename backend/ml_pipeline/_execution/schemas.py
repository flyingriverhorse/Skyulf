"""Execution schemas for the ML Pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
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
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    job_type: Literal["training", "tuning", "preview", "basic_training", "advanced_tuning"]
    status: JobStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    logs: Optional[List[str]] = None

    # Extended fields for Experiments Page
    model_type: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    metrics: Optional[Dict[str, Any]] = None
    search_strategy: Optional[str] = None
    target_column: Optional[str] = None
    dropped_columns: Optional[List[str]] = None
    version: Optional[int] = None
    graph: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    promoted_at: Optional[datetime] = None

    # Parallel branch metadata
    branch_index: Optional[int] = None
    parent_pipeline_id: Optional[str] = None


from backend.ml_pipeline.constants import StepType


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
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)  # IDs of upstream nodes


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""

    pipeline_id: str
    nodes: List[NodeConfig]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeExecutionResult:
    """Result of a single node execution."""

    node_id: str
    status: Literal["success", "failed", "skipped"]
    output_artifact_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0  # Seconds
    # Carried through so cross-cutting consumers (slow-nodes admin view,
    # debug logs) can identify the node without joining back to the
    # original PipelineConfig. Optional to keep older callsites valid.
    step_type: Optional[str] = None
    # Free-form per-node metadata surfaced to the canvas. `summary` is a
    # short one-line human string the node card renders post-run
    # (e.g. "7,000 / 1,500 / 1,500" for a split, "acc 0.87" for a
    # classifier). Other keys are reserved for future overlays
    # (duration_ms, memory_delta, drift_score, ...).
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineExecutionResult:
    """Result of the entire pipeline execution."""

    pipeline_id: str
    status: Literal["success", "failed", "partial"]
    node_results: Dict[str, NodeExecutionResult] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    merge_warnings: List[Dict[str, Any]] = field(default_factory=list)
    # Pre-execution schema predictions (C7 Phase B). One entry per node;
    # value is ``{"columns": [...], "dtypes": {...}}`` when the node's
    # Calculator overrides ``infer_output_schema`` and an upstream schema
    # was available, ``None`` otherwise. Populated in ``PipelineEngine.run``
    # before the per-node loop runs.
    predicted_schemas: Dict[str, Optional[Dict[str, Any]]] = field(default_factory=dict)
