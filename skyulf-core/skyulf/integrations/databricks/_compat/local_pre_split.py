"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.fitting.local_pre_split`."""

import sys

from skyulf.integrations.databricks.training.fitting import local_pre_split as _implementation

sys.modules[__name__] = _implementation
