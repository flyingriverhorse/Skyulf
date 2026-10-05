"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.shared.local_training_evidence`."""

import sys

from skyulf.integrations.databricks.training.shared import (
    local_training_evidence as _implementation,
)

sys.modules[__name__] = _implementation
