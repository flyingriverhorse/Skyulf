"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.charts.evaluation_charts`."""

import sys

from skyulf.integrations.databricks.observability.charts import evaluation_charts as _implementation

sys.modules[__name__] = _implementation
