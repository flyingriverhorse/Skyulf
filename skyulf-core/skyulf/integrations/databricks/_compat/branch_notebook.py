"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.training.branch_notebook`."""

import sys

from skyulf.integrations.databricks.jobs.training import branch_notebook as _implementation

sys.modules[__name__] = _implementation
