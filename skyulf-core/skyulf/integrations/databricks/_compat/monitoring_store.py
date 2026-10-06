"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.monitoring_store`."""

import sys

from skyulf.integrations.databricks.observability.monitoring import (
    monitoring_store as _implementation,
)

sys.modules[__name__] = _implementation
