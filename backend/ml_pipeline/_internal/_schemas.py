"""Pydantic request/response models for the ML-pipeline router.

Extracted from `api.py` (E9). Routers and route handlers stay in
`api.py`; this module only owns the on-the-wire contracts plus the
shared `_advisor.Recommendation` re-export so legacy
`from backend.ml_pipeline.api import RunPipelineResponse` style
imports continue to work via `api.py`'s re-export shim.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from backend.ml_pipeline._internal._advisor import Recommendation
from backend.ml_pipeline.constants import StepType


class RegistryItem(BaseModel):
    """Node metadata shape returned by the /registry endpoint."""

    id: str
    name: str
    category: str
    description: str
    params: Dict[str, Any] = {}
    tags: List[str] = []


class NodeConfigModel(BaseModel):
    node_id: str
    step_type: str
    params: Dict[str, Any] = {}
    inputs: List[str] = []


class PipelineConfigModel(BaseModel):
    pipeline_id: str
    nodes: List[NodeConfigModel]
    metadata: Dict[str, Any] = {}
    target_node_id: Optional[str] = None
    # "basic_training", "advanced_tuning", or "preview".
    job_type: Optional[str] = StepType.BASIC_TRAINING


class RunPipelineResponse(BaseModel):
    message: str
    pipeline_id: str
    job_id: str
    job_ids: List[str] = []  # All jobs when parallel branches are detected


class PreviewResponse(BaseModel):
    pipeline_id: str
    status: str
    node_results: Dict[str, Any]
    # Preview data for the last node (or specific nodes).
    preview_data: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    # True row counts per split key in `preview_data` (rows are capped at 50
    # for transport). For single-frame previews uses the synthetic
    # `_total` key.
    preview_totals: Optional[Dict[str, int]] = None
    # Per-branch preview keyed by branch label (e.g. "Path A · Random Forest")
    # when the pipeline has multiple parallel branches.
    branch_previews: Optional[Dict[str, Any]] = None
    # Per-branch true row counts (mirrors `branch_previews` keys).
    branch_preview_totals: Optional[Dict[str, Dict[str, int]]] = None
    # Per-branch list of node ids that ran in that branch. Used by the
    # frontend to show only the relevant "applied steps" pills per tab.
    branch_node_ids: Optional[Dict[str, List[str]]] = None
    recommendations: List[Recommendation] = []
    # Advisory messages from the engine about merge semantics applied during
    # this preview (e.g. sibling fan-in detected).
    merge_warnings: List[Dict[str, Any]] = []
    # Soft per-node warnings captured by `WarningCaptureHandler` during
    # the run (e.g. TargetEncoder coerced a float multiclass target,
    # OneHotEncoder saw a degenerate category). Each entry is
    # `{"node_id", "node_type", "level", "logger", "message"}`. Surfaced
    # by the frontend as toasts + an in-app notification panel.
    node_warnings: List[Dict[str, Any]] = []


class SavedPipelineModel(BaseModel):
    name: str
    description: Optional[str] = None
    graph: Dict[str, Any]
    # L7: optional metadata captured at save time so the auto-snapshot row
    # carries useful labels even when the client doesn't pass them.
    note: Optional[str] = None
    dataset_name: Optional[str] = None


class PipelineVersionCreateModel(BaseModel):
    """Body for explicit POST /pipeline/versions/{dataset_id}."""

    name: str
    graph: Dict[str, Any]
    note: Optional[str] = None
    dataset_name: Optional[str] = None
    kind: str = "manual"
    pinned: bool = False


class PipelineVersionPatchModel(BaseModel):
    """Body for PATCH /pipeline/versions/{dataset_id}/{version_id}."""

    name: Optional[str] = None
    note: Optional[str] = None
    pinned: Optional[bool] = None


__all__ = [
    "RegistryItem",
    "NodeConfigModel",
    "PipelineConfigModel",
    "RunPipelineResponse",
    "PreviewResponse",
    "SavedPipelineModel",
    "PipelineVersionCreateModel",
    "PipelineVersionPatchModel",
]
