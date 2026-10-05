"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.local_branches`."""

import sys

from skyulf.integrations.databricks.training import local_branches as _implementation

sys.modules[__name__] = _implementation
