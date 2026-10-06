"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.shared.scoring_pre_split`."""

import sys

from skyulf.integrations.databricks.scoring.shared import scoring_pre_split as _implementation

sys.modules[__name__] = _implementation
