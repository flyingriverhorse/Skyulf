"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.incremental.local_incremental`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.scoring.incremental import incremental_batch as _implementation

if TYPE_CHECKING:
    from ..scoring.incremental.incremental_batch import (
        IncrementalBatchResult as IncrementalBatchResult,
    )
    from ..scoring.incremental.incremental_batch import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from ..scoring.incremental.incremental_batch import (
        LocalWorkflowConfig as LocalWorkflowConfig,
    )
    from ..scoring.incremental.incremental_batch import (
        PreparedLocalWorkflow as PreparedLocalWorkflow,
    )
    from ..scoring.incremental.incremental_batch import (
        SourceChangeRequiresRebuild as SourceChangeRequiresRebuild,
    )
    from ..scoring.incremental.incremental_batch import (
        bounded_frame as bounded_frame,
    )
    from ..scoring.incremental.incremental_batch import (
        check_incremental_bootstrap as check_incremental_bootstrap,
    )
    from ..scoring.incremental.incremental_batch import (
        importlib as importlib,
    )
    from ..scoring.incremental.incremental_batch import (
        last_receipt as last_receipt,
    )
    from ..scoring.incremental.incremental_batch import (
        latest_source_version as latest_source_version,
    )
    from ..scoring.incremental.incremental_batch import (
        require_incremental_change_feed as require_incremental_change_feed,
    )
    from ..scoring.incremental.incremental_batch import (
        run_incremental_batch as run_incremental_batch,
    )
    from ..scoring.incremental.incremental_batch import (
        run_incremental_local_batch as run_incremental_local_batch,
    )
    from ..scoring.incremental.incremental_batch import (
        score_distributed_single as score_distributed_single,
    )
    from ..scoring.incremental.incremental_batch import (
        select_incremental_rows as select_incremental_rows,
    )
    from ..scoring.incremental.incremental_batch import (
        validate_source_change_policy as validate_source_change_policy,
    )

sys.modules[__name__] = _implementation
