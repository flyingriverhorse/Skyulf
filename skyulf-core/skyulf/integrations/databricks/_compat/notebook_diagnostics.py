"""Compatibility alias for :mod:`skyulf.integrations.databricks.jobs.shared.notebook_diagnostics`."""

import sys

from skyulf.integrations.databricks.jobs.shared import notebook_diagnostics as _implementation

sys.modules[__name__] = _implementation
