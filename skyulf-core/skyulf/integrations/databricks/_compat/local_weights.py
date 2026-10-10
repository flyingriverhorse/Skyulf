"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.weights.local_weights`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.weights import weights as _implementation

if TYPE_CHECKING:
    from ..training.weights.weights import (
        extract_training_weights as extract_training_weights,
    )
    from ..training.weights.weights import (
        training_weight_evidence as training_weight_evidence,
    )
    from ..training.weights.weights import (
        validate_weight_snapshot as validate_weight_snapshot,
    )

sys.modules[__name__] = _implementation
