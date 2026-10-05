"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.monitoring.spark_monitoring_job`."""

import sys

from skyulf.integrations.databricks.jobs.monitoring import spark_monitoring_job as _implementation

sys.modules[__name__] = _implementation
