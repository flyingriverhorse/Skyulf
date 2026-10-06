"""Compatibility alias for :mod:`skyulf.integrations.databricks.lifecycle._lifecycle_state`."""

import sys

from skyulf.integrations.databricks.lifecycle import _lifecycle_state as _implementation

sys.modules[__name__] = _implementation
