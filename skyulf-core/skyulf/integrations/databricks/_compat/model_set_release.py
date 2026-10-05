"""Compatibility alias for :mod:`skyulf.integrations.databricks.model_sets.model_set_release`."""

import sys

from skyulf.integrations.databricks.model_sets import model_set_release as _implementation

sys.modules[__name__] = _implementation
