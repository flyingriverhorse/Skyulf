"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.charts.evaluation_chart_data`."""

import sys

from skyulf.integrations.databricks.observability.charts import (
    evaluation_chart_data as _implementation,
)

sys.modules[__name__] = _implementation
