"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.training.branch_tasks`."""

import sys

from skyulf.integrations.databricks.jobs.training import branch_tasks as _implementation

sys.modules[__name__] = _implementation
