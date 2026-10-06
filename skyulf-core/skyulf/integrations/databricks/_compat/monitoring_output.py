"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.monitoring_output`."""

import sys

from skyulf.integrations.databricks.observability.monitoring import (
    monitoring_output as _implementation,
)

sys.modules[__name__] = _implementation
