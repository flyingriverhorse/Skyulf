"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_reference`."""

import sys

from skyulf.integrations.databricks.observability.monitoring.spark import (
    spark_monitoring_reference as _implementation,
)

sys.modules[__name__] = _implementation
