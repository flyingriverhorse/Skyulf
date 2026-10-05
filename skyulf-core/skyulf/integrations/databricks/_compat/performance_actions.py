"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.performance.performance_actions`."""

import sys

from skyulf.integrations.databricks.observability.monitoring.performance import (
    performance_actions as _implementation,
)

sys.modules[__name__] = _implementation
