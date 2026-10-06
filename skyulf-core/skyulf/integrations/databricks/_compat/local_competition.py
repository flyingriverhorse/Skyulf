"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.competition.local_competition`."""

import sys

from skyulf.integrations.databricks.training.competition import local_competition as _implementation

sys.modules[__name__] = _implementation
