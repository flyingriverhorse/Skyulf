"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.monitoring_source_evidence`."""

import sys

from skyulf.integrations.databricks.observability.monitoring import (
    monitoring_source_evidence as _implementation,
)

sys.modules[__name__] = _implementation
