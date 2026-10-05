"""Compatibility alias for :mod:`skyulf.integrations.databricks.data.delta_io.delta_admission`."""

import sys

from skyulf.integrations.databricks.data.delta_io import delta_admission as _implementation

sys.modules[__name__] = _implementation
