"""Compatibility alias for :mod:`skyulf.integrations.databricks.model_sets.model_set_output`."""

import sys

from skyulf.integrations.databricks.model_sets import model_set_output as _implementation

sys.modules[__name__] = _implementation
