"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.incremental.local_history`."""

import sys

from skyulf.integrations.databricks.scoring.incremental import local_history as _implementation

sys.modules[__name__] = _implementation
