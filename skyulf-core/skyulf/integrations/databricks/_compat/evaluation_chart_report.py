"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.charts.evaluation_chart_report`."""

import sys

from skyulf.integrations.databricks.observability.charts import (
    evaluation_chart_report as _implementation,
)

sys.modules[__name__] = _implementation
