"""Compatibility alias for :mod:`skyulf.integrations.databricks.lifecycle.local_approval`."""

import sys

from skyulf.integrations.databricks.lifecycle import local_approval as _implementation

sys.modules[__name__] = _implementation
