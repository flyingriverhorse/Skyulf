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
from .local_incremental import IncrementalBatchResult, run_incremental_local_batch
from .local_publish import run_local_batch
from .local_retraining import (
    LocalCandidateResult,
    LocalTrainingSpec,
    read_training_snapshot,
    split_labeled_snapshot,
    train_local_candidate,
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
from .training_dates import TrainingDateSpec

__all__ = [
    "BatchResult",
    "BatchSpec",
    "IncrementalBatchResult",
    "InputSource",
    "LocalWorkflowConfig",
    "LocalScoreResult",
    "LocalSourceSpec",
    "LocalCandidateResult",
    "LocalTrainingSpec",
    "ModelSelection",
    "OutputSink",
    "PreflightError",
    "PreflightIssue",
    "PreflightResult",
    "PreparedLocalWorkflow",
    "TrainingDateSpec",
    "evaluate_local_holdout",
    "fit_local_workflow",
    "preflight_local",
    "prepare_local_workflow",
    "read_local_source",
    "read_training_snapshot",
    "run_batch",
    "run_incremental_local_batch",
    "run_local_batch",
    "score_local_source",
    "split_labeled_snapshot",
    "train_local_candidate",
]
