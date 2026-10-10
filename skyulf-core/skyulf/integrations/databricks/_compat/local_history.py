"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.incremental.local_history`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.scoring.incremental import history as _implementation

if TYPE_CHECKING:
    from ..scoring.incremental.history import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from ..scoring.incremental.history import (
        bind_period_history as bind_period_history,
    )
    from ..scoring.incremental.history import (
        history_receipt as history_receipt,
    )
    from ..scoring.incremental.history import (
        incremental_history as incremental_history,
    )
    from ..scoring.incremental.history import (
        prediction_history as prediction_history,
    )

sys.modules[__name__] = _implementation
