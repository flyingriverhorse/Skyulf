"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.weights.weight_config`."""

import sys

from skyulf.integrations.databricks.training.weights import weight_config as _implementation

sys.modules[__name__] = _implementation
