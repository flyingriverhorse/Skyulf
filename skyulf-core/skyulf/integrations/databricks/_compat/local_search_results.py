"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.tuning.local_search_results`."""

import sys

from skyulf.integrations.databricks.training.tuning import local_search_results as _implementation

sys.modules[__name__] = _implementation
