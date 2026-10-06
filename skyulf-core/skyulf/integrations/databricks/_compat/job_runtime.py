"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.shared.job_runtime`."""

import sys

from skyulf.integrations.databricks.jobs.shared import job_runtime as _implementation

sys.modules[__name__] = _implementation
