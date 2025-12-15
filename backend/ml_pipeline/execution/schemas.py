"""Execution schemas for the ML Pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


@dataclass
class NodeConfig:
    """Configuration for a single pipeline node."""

    node_id: str
    step_type: Literal[
        "data_loader", "feature_engineering", "model_training", "model_tuning"
    ]
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
