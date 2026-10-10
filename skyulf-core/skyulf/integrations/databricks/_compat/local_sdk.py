"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.local_sdk`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.scoring import workflow as _implementation

if TYPE_CHECKING:
    from ..scoring.workflow import (
        InputSource as InputSource,
    )
    from ..scoring.workflow import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from ..scoring.workflow import (
        LocalWorkflowConfig as LocalWorkflowConfig,
    )
    from ..scoring.workflow import (
        ModelSelection as ModelSelection,
    )
    from ..scoring.workflow import (
        OutputSink as OutputSink,
    )
    from ..scoring.workflow import (
        PreflightError as PreflightError,
    )
    from ..scoring.workflow import (
        PreflightIssue as PreflightIssue,
    )
    from ..scoring.workflow import (
        PreflightResult as PreflightResult,
    )
    from ..scoring.workflow import (
        PreparedLocalWorkflow as PreparedLocalWorkflow,
    )
    from ..scoring.workflow import (
        PreparedWorkflow as PreparedWorkflow,
    )
    from ..scoring.workflow import (
        WorkflowConfig as WorkflowConfig,
    )
    from ..scoring.workflow import (
        preflight as preflight,
    )
    from ..scoring.workflow import (
        preflight_local as preflight_local,
    )
    from ..scoring.workflow import (
        prepare_local_workflow as prepare_local_workflow,
    )
    from ..scoring.workflow import (
        prepare_workflow as prepare_workflow,
    )

sys.modules[__name__] = _implementation
