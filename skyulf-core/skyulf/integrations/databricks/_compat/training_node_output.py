"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.reports.training_node_output`."""

import sys

from skyulf.integrations.databricks.observability.reports import (
    training_node_output as _implementation,
)

sys.modules[__name__] = _implementation
