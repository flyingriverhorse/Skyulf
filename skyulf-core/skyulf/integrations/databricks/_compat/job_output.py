"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.shared.job_output`."""

import sys

from skyulf.integrations.databricks.jobs.shared import job_output as _implementation

sys.modules[__name__] = _implementation
