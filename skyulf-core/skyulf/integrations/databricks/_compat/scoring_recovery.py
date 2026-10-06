"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.incremental.scoring_recovery`."""

import sys

from skyulf.integrations.databricks.scoring.incremental import scoring_recovery as _implementation

sys.modules[__name__] = _implementation
