"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.shared.training_parameters`."""

import sys

from skyulf.integrations.databricks.training.shared import training_parameters as _implementation

sys.modules[__name__] = _implementation
