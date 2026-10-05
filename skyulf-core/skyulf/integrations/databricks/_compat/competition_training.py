"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.competition.competition_training`."""

import sys

from skyulf.integrations.databricks.training.competition import (
    competition_training as _implementation,
)

sys.modules[__name__] = _implementation
