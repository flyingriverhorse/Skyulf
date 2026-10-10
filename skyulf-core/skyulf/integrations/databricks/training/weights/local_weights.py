"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.weights.weights`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.weights import weights as _implementation

if TYPE_CHECKING:
    from .weights import (
        extract_training_weights as extract_training_weights,
    )
    from .weights import (
        training_weight_evidence as training_weight_evidence,
    )
    from .weights import (
        validate_weight_snapshot as validate_weight_snapshot,
    )

sys.modules[__name__] = _implementation
