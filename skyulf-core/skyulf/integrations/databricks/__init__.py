"""Optional Databricks batch and Delta integrations."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

# Keep old module imports available without loading every runtime component.
__path__ = [*__path__, str(Path(__file__).with_name("_compat"))]

from .data.training.training_dates import TrainingDateSpec
from .scoring.batch.batch import BatchResult, BatchSpec, run_batch
from .scoring.batch.local_batch import (
    LocalScoreResult,
    LocalSourceSpec,
    evaluate_local_holdout,
    fit_local_workflow,
    read_local_source,
    score_local_source,
)
from .scoring.incremental.local_incremental import (
    IncrementalBatchResult,
    run_incremental_local_batch,
)
from .scoring.local_publish import run_local_batch
from .scoring.local_sdk import (
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
from .training.fitting.local_retraining import (
    LocalCandidateResult,
    LocalTrainingSpec,
    read_training_snapshot,
    split_labeled_snapshot,
    train_local_candidate,
)

if TYPE_CHECKING:
    from .training.local_branches import (
        BranchTrainingResult,
        TrainingBranch,
        branch_training_payload,
        prepare_training_branches,
        restore_training_branches,
        train_local_branches,
    )

_BRANCH_EXPORTS = {
    "BranchTrainingResult",
    "TrainingBranch",
    "branch_training_payload",
    "prepare_training_branches",
    "restore_training_branches",
    "train_local_branches",
}


def __getattr__(name: str) -> Any:
    """Load training orchestration only after low-level alias imports have finished."""
    if name not in _BRANCH_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .training import local_branches  # noqa: PLC0415 - avoid promotion/admission import cycle

    value = getattr(local_branches, name)
    globals()[name] = value
    return value


__all__ = [
    "BatchResult",
    "BatchSpec",
    "BranchTrainingResult",
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
    "TrainingBranch",
    "branch_training_payload",
    "evaluate_local_holdout",
    "fit_local_workflow",
    "preflight_local",
    "prepare_local_workflow",
    "prepare_training_branches",
    "read_local_source",
    "read_training_snapshot",
    "restore_training_branches",
    "run_batch",
    "run_incremental_local_batch",
    "run_local_batch",
    "score_local_source",
    "split_labeled_snapshot",
    "train_local_candidate",
    "train_local_branches",
]
