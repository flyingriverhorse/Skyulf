"""Internal package for `backend.ml_pipeline`.

Holds the pieces extracted from the historic monolithic `api.py`
(E9 modularisation): advisor classes, request/response schemas, and
small label helpers. Routers will follow in phase 2.

The leading underscore signals these modules are private to
`backend.ml_pipeline` — external code should keep importing from
`backend.ml_pipeline.api`, which re-exports everything below.
"""

from ._advisor import AdvisorEngine, AnalysisProfile, DataProfiler, Recommendation
from ._helpers import branch_label, prettify_model_type
from ._schemas import (
    NodeConfigModel,
    PipelineConfigModel,
    PipelineVersionCreateModel,
    PipelineVersionPatchModel,
    PreviewResponse,
    RegistryItem,
    RunPipelineResponse,
    SavedPipelineModel,
)

__all__ = [
    # _advisor
    "AdvisorEngine",
    "AnalysisProfile",
    "DataProfiler",
    "Recommendation",
    # _helpers
    "branch_label",
    "prettify_model_type",
    # _schemas
    "NodeConfigModel",
    "PipelineConfigModel",
    "PipelineVersionCreateModel",
    "PipelineVersionPatchModel",
    "PreviewResponse",
    "RegistryItem",
    "RunPipelineResponse",
    "SavedPipelineModel",
]
