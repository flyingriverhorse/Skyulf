"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.lifecycle.retraining_task`."""

import sys

from skyulf.integrations.databricks.jobs.lifecycle import retraining_task as _implementation

sys.modules[__name__] = _implementation
