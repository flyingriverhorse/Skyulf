"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.local.monitoring_metrics`."""

import sys

from skyulf.integrations.databricks.observability.monitoring.local import (
    monitoring_metrics as _implementation,
)

sys.modules[__name__] = _implementation
