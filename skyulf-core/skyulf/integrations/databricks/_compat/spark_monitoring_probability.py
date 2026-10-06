"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_probability`."""

import sys

from skyulf.integrations.databricks.observability.monitoring.spark import (
    spark_monitoring_probability as _implementation,
)

sys.modules[__name__] = _implementation
