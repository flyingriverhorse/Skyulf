"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.competition.competition_evaluation`."""

import sys

from skyulf.integrations.databricks.training.competition import (
    competition_evaluation as _implementation,
)

sys.modules[__name__] = _implementation
