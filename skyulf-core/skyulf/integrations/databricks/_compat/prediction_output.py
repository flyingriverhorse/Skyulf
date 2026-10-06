"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.shared.prediction_output`."""

import sys

from skyulf.integrations.databricks.scoring.shared import prediction_output as _implementation

sys.modules[__name__] = _implementation
