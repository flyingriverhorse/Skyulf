"""Compatibility alias for :mod:`skyulf.integrations.databricks.model_sets.model_set_project`."""

import sys

from skyulf.integrations.databricks.model_sets import model_set_project as _implementation

sys.modules[__name__] = _implementation
