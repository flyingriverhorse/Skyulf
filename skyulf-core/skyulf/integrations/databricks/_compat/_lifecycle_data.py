"""Compatibility alias for :mod:`skyulf.integrations.databricks.lifecycle._lifecycle_data`."""

import sys

from skyulf.integrations.databricks.lifecycle import _lifecycle_data as _implementation

sys.modules[__name__] = _implementation
