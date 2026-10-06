"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.fitting.local_ensemble`."""

import sys

from skyulf.integrations.databricks.training.fitting import local_ensemble as _implementation

sys.modules[__name__] = _implementation
