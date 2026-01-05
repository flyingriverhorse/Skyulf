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


from backend.ml_pipeline.constants import StepType

@dataclass
class NodeConfig:
    """Configuration for a single pipeline node."""

    node_id: str
    step_type: StepType
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


@dataclass
class PipelineExecutionResult:
    """Result of the entire pipeline execution."""

    pipeline_id: str
    status: Literal["success", "failed", "partial"]
    node_results: Dict[str, NodeExecutionResult] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
