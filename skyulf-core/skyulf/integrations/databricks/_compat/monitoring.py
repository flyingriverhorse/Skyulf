"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.local.monitoring`."""

import sys

from skyulf.integrations.databricks.observability.monitoring.local import (
    monitoring as _implementation,
)

sys.modules[__name__] = _implementation
