"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.incremental.local_incremental`."""

import sys

from skyulf.integrations.databricks.scoring.incremental import local_incremental as _implementation

sys.modules[__name__] = _implementation
