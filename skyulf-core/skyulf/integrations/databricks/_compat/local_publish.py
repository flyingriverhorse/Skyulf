"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.local_publish`."""

import sys

from skyulf.integrations.databricks.scoring import local_publish as _implementation

sys.modules[__name__] = _implementation
