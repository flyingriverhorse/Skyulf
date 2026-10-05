"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.local_sdk`."""

import sys

from skyulf.integrations.databricks.scoring import local_sdk as _implementation

sys.modules[__name__] = _implementation
