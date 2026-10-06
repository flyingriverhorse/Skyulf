"""Compatibility alias for :mod:`skyulf.integrations.databricks.model_sets.monitoring_model_set`."""

import sys

from skyulf.integrations.databricks.model_sets import monitoring_model_set as _implementation

sys.modules[__name__] = _implementation
