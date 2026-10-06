"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.performance.performance_retraining`."""

import sys

from skyulf.integrations.databricks.observability.monitoring.performance import (
    performance_retraining as _implementation,
)

sys.modules[__name__] = _implementation
