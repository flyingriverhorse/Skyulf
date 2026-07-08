"""Pydantic request/response models for the ML-pipeline router.

Extracted from `api.py` (E9). Routers and route handlers stay in
`api.py`; this module only owns the on-the-wire contracts plus the
shared `_advisor.Recommendation` re-export so legacy
`from backend.ml_pipeline.api import RunPipelineResponse` style
imports continue to work via `api.py`'s re-export shim.
"""

from typing import Any

from pydantic import BaseModel

from backend.ml_pipeline._internal._advisor import Recommendation
from backend.ml_pipeline.constants import StepType


class RegistryItem(BaseModel):
    """Node metadata shape returned by the /registry endpoint."""

    id: str
    name: str
    category: str
    description: str
    params: dict[str, Any] = {}
    tags: list[str] = []


class NodeConfigModel(BaseModel):
    node_id: str
    step_type: str
    params: dict[str, Any] = {}
    inputs: list[str] = []


class PipelineConfigModel(BaseModel):
    pipeline_id: str
    nodes: list[NodeConfigModel]
    metadata: dict[str, Any] = {}
    target_node_id: str | None = None
    # "basic_training", "advanced_tuning", or "preview".
    job_type: str | None = StepType.BASIC_TRAINING


class RunPipelineResponse(BaseModel):
    message: str
    pipeline_id: str
    job_id: str
    job_ids: list[str] = []  # All jobs when parallel branches are detected


class PreviewResponse(BaseModel):
    pipeline_id: str
    status: str
    node_results: dict[str, Any]
    # Preview data for the last node (or specific nodes).
    preview_data: list[dict[str, Any]] | dict[str, Any] | None = None
    # True row counts per split key in `preview_data` (rows are capped at 50
    # for transport). For single-frame previews uses the synthetic
    # `_total` key.
    preview_totals: dict[str, int] | None = None
    # Per-branch preview keyed by branch label (e.g. "Path A · Random Forest")
    # when the pipeline has multiple parallel branches.
    branch_previews: dict[str, Any] | None = None
    # Per-branch true row counts (mirrors `branch_previews` keys).
    branch_preview_totals: dict[str, dict[str, int]] | None = None
    # Per-branch list of node ids that ran in that branch. Used by the
    # frontend to show only the relevant "applied steps" pills per tab.
    branch_node_ids: dict[str, list[str]] | None = None
    recommendations: list[Recommendation] = []
    # Advisory messages from the engine about merge semantics applied during
    # this preview (e.g. sibling fan-in detected).
    merge_warnings: list[dict[str, Any]] = []
    # Soft per-node warnings captured by `WarningCaptureHandler` during
    # the run (e.g. TargetEncoder coerced a float multiclass target,
    # OneHotEncoder saw a degenerate category). Each entry is
    # `{"node_id", "node_type", "level", "logger", "message"}`. Surfaced
    # by the frontend as toasts + an in-app notification panel.
    node_warnings: list[dict[str, Any]] = []


class SavedPipelineModel(BaseModel):
    name: str
    description: str | None = None
    graph: dict[str, Any]
    # L7: optional metadata captured at save time so the auto-snapshot row
    # carries useful labels even when the client doesn't pass them.
    note: str | None = None
    dataset_name: str | None = None


class PipelineVersionCreateModel(BaseModel):
    """Body for explicit POST /pipeline/versions/{dataset_id}."""

    name: str
    graph: dict[str, Any]
    note: str | None = None
    dataset_name: str | None = None
    kind: str = "manual"
    pinned: bool = False


class PipelineVersionPatchModel(BaseModel):
    """Body for PATCH /pipeline/versions/{dataset_id}/{version_id}."""

    name: str | None = None
    note: str | None = None
    pinned: bool | None = None


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
