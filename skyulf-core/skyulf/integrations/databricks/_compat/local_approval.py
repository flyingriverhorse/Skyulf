"""Compatibility alias for :mod:`skyulf.integrations.databricks.lifecycle.local_approval`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.lifecycle import approval as _implementation

if TYPE_CHECKING:
    from ..lifecycle.approval import (
        approve_candidate as approve_candidate,
    )
    from ..lifecycle.approval import (
        approve_local_candidate as approve_local_candidate,
    )
    from ..lifecycle.approval import (
        load_candidate_evidence as load_candidate_evidence,
    )
    from ..lifecycle.approval import reject_candidate as reject_candidate
    from ..lifecycle.approval import (
        reject_local_candidate as reject_local_candidate,
    )
    from ..lifecycle.approval import (
        reject_workflow_candidate as reject_workflow_candidate,
    )
    from ..lifecycle.approval import (
        resolve_candidate_comparison_digest as resolve_candidate_comparison_digest,
    )
    from ..lifecycle.approval import (
        validate_training_evidence as validate_training_evidence,
    )

sys.modules[__name__] = _implementation
