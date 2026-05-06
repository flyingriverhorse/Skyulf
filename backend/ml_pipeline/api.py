"""ML pipeline HTTP surface — thin aggregator.

This module used to host every `/ml-pipeline/*` route inline (~1500 LOC).
After the modularisation it is just an aggregator: it owns the public
`router` instance and mounts five sub-routers from
`_internal/_routers/*`. Public URLs and import paths are unchanged.

Sub-routers (mount order is irrelevant — paths are unique):

- `pipelines_io`  — `/save`, `/load`, `/versions/...`
- `run_pipeline`  — `POST /run`
- `preview`       — `POST /preview`
- `jobs`          — `/jobs/...`
- `meta`          — `/registry`, `/stats`, `/datasets/...`, `/hyperparameters/...`

Backward-compat re-exports below keep
`from backend.ml_pipeline.api import …` working for tests and any
external callers (Pydantic models, helper functions, the legacy
`get_job_evaluation` handler, `_build_node_registry`, etc.).
"""

from __future__ import annotations

from fastapi import APIRouter

from ._internal._routers.jobs import get_job_evaluation  # noqa: F401  (re-exported)
from ._internal._routers.jobs import router as _jobs_router
from ._internal._routers.meta import _build_node_registry  # noqa: F401  (re-exported)
from ._internal._routers.meta import router as _meta_router
from ._internal._routers.pipelines_io import router as _pipelines_io_router
from ._internal._routers.preview import router as _preview_router
from ._internal._routers.run_pipeline import _get_submit_lock  # noqa: F401  (re-exported)
from ._internal._routers.run_pipeline import router as _run_router

# Schema / helper / advisor re-exports — used by tests and callers that
# imported these directly from `backend.ml_pipeline.api` before the E9 split.
from ._internal import (  # noqa: F401  (re-exported)
    AdvisorEngine,
    AnalysisProfile,
    DataProfiler,
    NodeConfigModel,
    PipelineConfigModel,
    PipelineVersionCreateModel,
    PipelineVersionPatchModel,
    PreviewResponse,
    Recommendation,
    RegistryItem,
    RunPipelineResponse,
    SavedPipelineModel,
)
from ._internal import branch_label as _branch_label  # noqa: F401
from ._internal import prettify_model_type as _prettify_model_type  # noqa: F401

# Single public router; main.py mounts this under `/ml-pipeline`.
router = APIRouter(tags=["ML Pipeline"])

# Mount sub-routers. Each sub-router uses the same tag and no prefix,
# so the resulting path surface is byte-identical to the pre-split version.
router.include_router(_pipelines_io_router)
router.include_router(_run_router)
router.include_router(_preview_router)
router.include_router(_jobs_router)
router.include_router(_meta_router)

__all__ = [
    "router",
    # Schemas
    "PipelineConfigModel",
    "NodeConfigModel",
    "PreviewResponse",
    "RunPipelineResponse",
    "RegistryItem",
    "SavedPipelineModel",
    "PipelineVersionCreateModel",
    "PipelineVersionPatchModel",
    # Advisor
    "Recommendation",
    "AnalysisProfile",
    "DataProfiler",
    "AdvisorEngine",
    # Helpers (legacy underscore-prefixed names)
    "_branch_label",
    "_prettify_model_type",
    "_build_node_registry",
    "_get_submit_lock",
    # Legacy handler reference
    "get_job_evaluation",
]
