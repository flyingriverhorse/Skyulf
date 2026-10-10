"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.local_publish`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.scoring import publish as _implementation

if TYPE_CHECKING:
    from ..scoring.publish import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from ..scoring.publish import (
        LocalSourceSpec as LocalSourceSpec,
    )
    from ..scoring.publish import (
        LocalWorkflowConfig as LocalWorkflowConfig,
    )
    from ..scoring.publish import (
        PreparedLocalWorkflow as PreparedLocalWorkflow,
    )
    from ..scoring.publish import (
        check_target as check_target,
    )
    from ..scoring.publish import (
        run_frame_batch as run_frame_batch,
    )
    from ..scoring.publish import (
        run_local_batch as run_local_batch,
    )

sys.modules[__name__] = _implementation
