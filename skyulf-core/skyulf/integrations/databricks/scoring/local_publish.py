"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.publish`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.scoring import publish as _implementation

if TYPE_CHECKING:
    from .publish import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from .publish import (
        LocalSourceSpec as LocalSourceSpec,
    )
    from .publish import (
        LocalWorkflowConfig as LocalWorkflowConfig,
    )
    from .publish import (
        PreparedLocalWorkflow as PreparedLocalWorkflow,
    )
    from .publish import (
        check_target as check_target,
    )
    from .publish import (
        run_frame_batch as run_frame_batch,
    )
    from .publish import (
        run_local_batch as run_local_batch,
    )

sys.modules[__name__] = _implementation
