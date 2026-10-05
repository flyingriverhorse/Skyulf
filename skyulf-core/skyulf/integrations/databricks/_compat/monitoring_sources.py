"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.monitoring_sources`."""

import sys

from skyulf.integrations.databricks.observability.monitoring import (
    monitoring_sources as _implementation,
)

sys.modules[__name__] = _implementation
