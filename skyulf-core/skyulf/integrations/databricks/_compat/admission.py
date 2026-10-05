"""Compatibility alias for :mod:`skyulf.integrations.databricks.data.admission`."""

import sys

from skyulf.integrations.databricks.data import admission as _implementation

sys.modules[__name__] = _implementation
