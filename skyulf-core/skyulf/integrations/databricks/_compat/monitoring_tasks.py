"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.monitoring.monitoring_tasks`."""

import sys

from skyulf.integrations.databricks.jobs.monitoring import monitoring_tasks as _implementation

sys.modules[__name__] = _implementation
