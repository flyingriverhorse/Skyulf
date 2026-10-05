"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.tuning.local_cv`."""

import sys

from skyulf.integrations.databricks.training.tuning import local_cv as _implementation

sys.modules[__name__] = _implementation
