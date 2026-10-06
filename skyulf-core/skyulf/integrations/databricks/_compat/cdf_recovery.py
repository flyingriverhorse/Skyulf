"""Compatibility alias for :mod:`skyulf.integrations.databricks.data.delta_io.cdf_recovery`."""

import sys

from skyulf.integrations.databricks.data.delta_io import cdf_recovery as _implementation

sys.modules[__name__] = _implementation
