"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.lifecycle.lifecycle_tasks`."""

import sys

from skyulf.integrations.databricks.jobs.lifecycle import lifecycle_tasks as _implementation

sys.modules[__name__] = _implementation
