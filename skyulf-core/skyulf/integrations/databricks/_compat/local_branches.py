"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.local_branches`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training import branches as _implementation

if TYPE_CHECKING:
    from ..training.branches import (
        BranchTrainingResult as BranchTrainingResult,
    )
    from ..training.branches import (
        LocalCandidateResult as LocalCandidateResult,
    )
    from ..training.branches import (
        LocalCVSpec as LocalCVSpec,
    )
    from ..training.branches import (
        LocalTrainingSpec as LocalTrainingSpec,
    )
    from ..training.branches import (
        TrainingBranch as TrainingBranch,
    )
    from ..training.branches import (
        branch_training_payload as branch_training_payload,
    )
    from ..training.branches import (
        log_progress as log_progress,
    )
    from ..training.branches import (
        prepare_training_branches as prepare_training_branches,
    )
    from ..training.branches import (
        restore_training_branches as restore_training_branches,
    )
    from ..training.branches import (
        train_branch as train_branch,
    )
    from ..training.branches import (
        train_branches as train_branches,
    )
    from ..training.branches import (
        train_local_branches as train_local_branches,
    )

sys.modules[__name__] = _implementation
