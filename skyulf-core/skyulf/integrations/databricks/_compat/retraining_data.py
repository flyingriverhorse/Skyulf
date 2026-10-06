"""Compatibility alias for :mod:`skyulf.integrations.databricks.data.training.retraining_data`."""

import sys

from skyulf.integrations.databricks.data.training import retraining_data as _implementation

sys.modules[__name__] = _implementation
