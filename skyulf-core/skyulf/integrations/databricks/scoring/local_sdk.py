"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.workflow`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.scoring import workflow as _implementation

if TYPE_CHECKING:
    from .workflow import (
        InputSource as InputSource,
    )
    from .workflow import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from .workflow import (
        LocalWorkflowConfig as LocalWorkflowConfig,
    )
    from .workflow import (
        ModelSelection as ModelSelection,
    )
    from .workflow import (
        OutputSink as OutputSink,
    )
    from .workflow import (
        PreflightError as PreflightError,
    )
    from .workflow import (
        PreflightIssue as PreflightIssue,
    )
    from .workflow import (
        PreflightResult as PreflightResult,
    )
    from .workflow import (
        PreparedLocalWorkflow as PreparedLocalWorkflow,
    )
    from .workflow import (
        PreparedWorkflow as PreparedWorkflow,
    )
    from .workflow import (
        WorkflowConfig as WorkflowConfig,
    )
    from .workflow import (
        preflight as preflight,
    )
    from .workflow import (
        preflight_local as preflight_local,
    )
    from .workflow import (
        prepare_local_workflow as prepare_local_workflow,
    )
    from .workflow import (
        prepare_workflow as prepare_workflow,
    )

sys.modules[__name__] = _implementation
