"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.thresholds.decision_thresholds`."""

import sys

from skyulf.integrations.databricks.training.thresholds import (
    decision_thresholds as _implementation,
)

sys.modules[__name__] = _implementation
