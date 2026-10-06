"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.performance.performance_policy`."""

import sys

from skyulf.integrations.databricks.observability.monitoring.performance import (
    performance_policy as _implementation,
)

sys.modules[__name__] = _implementation
