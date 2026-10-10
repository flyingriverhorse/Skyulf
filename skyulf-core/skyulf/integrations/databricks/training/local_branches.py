"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.branches`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training import branches as _implementation

if TYPE_CHECKING:
    from .branches import (
        BranchTrainingResult as BranchTrainingResult,
    )
    from .branches import (
        LocalCandidateResult as LocalCandidateResult,
    )
    from .branches import (
        LocalCVSpec as LocalCVSpec,
    )
    from .branches import (
        LocalTrainingSpec as LocalTrainingSpec,
    )
    from .branches import (
        TrainingBranch as TrainingBranch,
    )
    from .branches import (
        branch_training_payload as branch_training_payload,
    )
    from .branches import (
        log_progress as log_progress,
    )
    from .branches import (
        prepare_training_branches as prepare_training_branches,
    )
    from .branches import (
        restore_training_branches as restore_training_branches,
    )
    from .branches import (
        train_branch as train_branch,
    )
    from .branches import (
        train_branches as train_branches,
    )
    from .branches import (
        train_local_branches as train_local_branches,
    )

sys.modules[__name__] = _implementation
