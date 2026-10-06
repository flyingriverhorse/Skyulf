"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.thresholds.threshold_training`."""

import sys

from skyulf.integrations.databricks.training.thresholds import threshold_training as _implementation

sys.modules[__name__] = _implementation
