"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.monitoring_reference`."""

import sys

from skyulf.integrations.databricks.observability.monitoring import (
    monitoring_reference as _implementation,
)

sys.modules[__name__] = _implementation
