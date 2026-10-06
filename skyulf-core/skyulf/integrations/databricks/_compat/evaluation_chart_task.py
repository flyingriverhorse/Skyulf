"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.evaluation_chart_task`."""

import sys

from skyulf.integrations.databricks.jobs import evaluation_chart_task as _implementation

sys.modules[__name__] = _implementation
