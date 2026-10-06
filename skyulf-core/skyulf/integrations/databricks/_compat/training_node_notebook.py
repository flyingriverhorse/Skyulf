"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.training.training_node_notebook`."""

import sys

from skyulf.integrations.databricks.jobs.training import training_node_notebook as _implementation

sys.modules[__name__] = _implementation
