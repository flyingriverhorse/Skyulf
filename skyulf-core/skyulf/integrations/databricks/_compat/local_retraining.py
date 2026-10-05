"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.fitting.local_retraining`."""

import sys

from skyulf.integrations.databricks.training.fitting import local_retraining as _implementation

sys.modules[__name__] = _implementation
