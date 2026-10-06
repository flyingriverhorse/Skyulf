"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.local.monitoring_performance`."""

import sys

from skyulf.integrations.databricks.observability.monitoring.local import (
    monitoring_performance as _implementation,
)

sys.modules[__name__] = _implementation
