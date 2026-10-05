"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.monitoring_config`."""

import sys

from skyulf.integrations.databricks.observability.monitoring import (
    monitoring_config as _implementation,
)

sys.modules[__name__] = _implementation
