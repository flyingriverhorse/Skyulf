"""Compatibility alias for :mod:`skyulf.integrations.databricks.shared._contracts`."""

import sys

from skyulf.integrations.databricks.shared import _contracts as _implementation

sys.modules[__name__] = _implementation
