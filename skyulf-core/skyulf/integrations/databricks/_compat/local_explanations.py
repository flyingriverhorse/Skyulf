"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.reports.local_explanations`."""

import sys

from skyulf.integrations.databricks.observability.reports import (
    local_explanations as _implementation,
)

sys.modules[__name__] = _implementation
