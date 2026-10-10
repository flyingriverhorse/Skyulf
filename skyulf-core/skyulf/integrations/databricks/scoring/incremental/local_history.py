"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.incremental.history`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.scoring.incremental import history as _implementation

if TYPE_CHECKING:
    from .history import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from .history import (
        bind_period_history as bind_period_history,
    )
    from .history import (
        history_receipt as history_receipt,
    )
    from .history import (
        incremental_history as incremental_history,
    )
    from .history import (
        prediction_history as prediction_history,
    )

sys.modules[__name__] = _implementation
