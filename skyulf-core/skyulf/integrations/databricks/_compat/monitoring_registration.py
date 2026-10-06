"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.monitoring_registration`."""

import sys

from skyulf.integrations.databricks.observability.monitoring import (
    monitoring_registration as _implementation,
)

sys.modules[__name__] = _implementation
