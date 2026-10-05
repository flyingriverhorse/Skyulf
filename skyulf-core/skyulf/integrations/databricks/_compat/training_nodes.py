"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.training.training_nodes`."""

import sys

from skyulf.integrations.databricks.jobs.training import training_nodes as _implementation

sys.modules[__name__] = _implementation
