"""Optional Databricks batch and Delta integrations."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

# Keep old module imports available without loading every runtime component.
__path__ = [*__path__, str(Path(__file__).with_name("_compat"))]

from .data.training.training_dates import TrainingDateSpec
from .scoring.batch.batch import BatchResult, BatchSpec, run_batch
from .scoring.batch.frame_batch import (
    ScoreResult,
    SourceSpec,
    evaluate_holdout,
    fit_workflow,
    read_source,
    score_source,
)
from .scoring.incremental.incremental_batch import (
    IncrementalBatchResult,
    run_incremental_batch,
)
from .scoring.publish import run_frame_batch
from .scoring.workflow import (
    InputSource,
    ModelSelection,
    OutputSink,
    PreflightError,
    PreflightIssue,
    PreflightResult,
    PreparedWorkflow,
    WorkflowConfig,
    preflight,
    prepare_workflow,
)
from .training.fitting.candidate import (
    CandidateResult,
    TrainingSpec,
    read_training_snapshot,
    split_labeled_snapshot,
    train_candidate,
)

# Keep previous public spellings as aliases to the same implementations.
LocalScoreResult = ScoreResult
LocalSourceSpec = SourceSpec
LocalWorkflowConfig = WorkflowConfig
PreparedLocalWorkflow = PreparedWorkflow
LocalCandidateResult = CandidateResult
LocalTrainingSpec = TrainingSpec
evaluate_local_holdout = evaluate_holdout
fit_local_workflow = fit_workflow
read_local_source = read_source
score_local_source = score_source
run_incremental_local_batch = run_incremental_batch
run_local_batch = run_frame_batch
preflight_local = preflight
prepare_local_workflow = prepare_workflow
train_local_candidate = train_candidate

if TYPE_CHECKING:
    from .training.branches import (
        BranchTrainingResult,
        TrainingBranch,
        branch_training_payload,
        prepare_training_branches,
        restore_training_branches,
        train_branches,
        train_local_branches,
    )

_BRANCH_EXPORTS = {
    "BranchTrainingResult",
    "TrainingBranch",
    "branch_training_payload",
    "prepare_training_branches",
    "restore_training_branches",
    "train_local_branches",
    "train_branches",
}


def __getattr__(name: str) -> Any:
    """Load training orchestration only after low-level alias imports have finished."""
    if name not in _BRANCH_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .training import branches  # noqa: PLC0415 - avoid promotion/admission import cycle

    value = getattr(branches, name)
    globals()[name] = value
    return value


__all__ = [
    "LocalScoreResult",
    "LocalSourceSpec",
    "LocalWorkflowConfig",
    "PreparedLocalWorkflow",
    "LocalCandidateResult",
    "LocalTrainingSpec",
    "evaluate_holdout",
    "fit_workflow",
    "read_source",
    "score_source",
    "run_incremental_batch",
    "run_frame_batch",
    "preflight",
    "prepare_workflow",
    "train_candidate",
    "train_branches",
    "BatchResult",
    "BatchSpec",
    "BranchTrainingResult",
    "IncrementalBatchResult",
    "InputSource",
    "WorkflowConfig",
    "ScoreResult",
    "SourceSpec",
    "CandidateResult",
    "TrainingSpec",
    "ModelSelection",
    "OutputSink",
    "PreflightError",
    "PreflightIssue",
    "PreflightResult",
    "PreparedWorkflow",
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
