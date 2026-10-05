"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.reports.explanation_report`."""

import sys

from skyulf.integrations.databricks.observability.reports import (
    explanation_report as _implementation,
)

sys.modules[__name__] = _implementation
