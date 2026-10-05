"""Compatibility alias for :mod:`skyulf.integrations.databricks.data.training.training_dates`."""

import sys

from skyulf.integrations.databricks.data.training import training_dates as _implementation

sys.modules[__name__] = _implementation
