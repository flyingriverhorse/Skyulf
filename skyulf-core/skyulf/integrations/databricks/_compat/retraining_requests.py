"""Compatibility alias for :mod:`skyulf.integrations.databricks.lifecycle.retraining_requests`."""

import sys

from skyulf.integrations.databricks.lifecycle import retraining_requests as _implementation

sys.modules[__name__] = _implementation
