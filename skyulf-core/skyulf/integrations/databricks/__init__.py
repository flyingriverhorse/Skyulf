"""Optional Databricks batch and Delta integrations."""

from .batch import BatchResult, BatchSpec, run_batch
from .local_batch import (
    LocalScoreResult,
    LocalSourceSpec,
    evaluate_local_holdout,
    fit_local_workflow,
    read_local_source,
    score_local_source,
)
from .local_sdk import (
    InputSource,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    PreflightError,
    PreflightIssue,
    PreflightResult,
    PreparedLocalWorkflow,
    preflight_local,
    prepare_local_workflow,
)

__all__ = [
    "BatchResult",
    "BatchSpec",
    "InputSource",
    "LocalWorkflowConfig",
    "LocalScoreResult",
    "LocalSourceSpec",
    "ModelSelection",
    "OutputSink",
    "PreflightError",
    "PreflightIssue",
    "PreflightResult",
    "PreparedLocalWorkflow",
    "evaluate_local_holdout",
    "fit_local_workflow",
    "preflight_local",
    "prepare_local_workflow",
    "read_local_source",
    "run_batch",
    "score_local_source",
]
