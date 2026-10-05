"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.weights.local_weights`."""

import sys

from skyulf.integrations.databricks.training.weights import local_weights as _implementation

sys.modules[__name__] = _implementation
