"""Compatibility alias for :mod:`skyulf.integrations.databricks.data.delta_io.delta`."""

import sys

from skyulf.integrations.databricks.data.delta_io import delta as _implementation

sys.modules[__name__] = _implementation
