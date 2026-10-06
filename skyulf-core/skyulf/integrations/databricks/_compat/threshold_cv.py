"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.thresholds.threshold_cv`."""

import sys

from skyulf.integrations.databricks.training.thresholds import threshold_cv as _implementation

sys.modules[__name__] = _implementation
