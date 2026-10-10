"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.fitting.ensemble`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.fitting import ensemble as _implementation

if TYPE_CHECKING:
    from .ensemble import (
        ENSEMBLE_MODELS as ENSEMBLE_MODELS,
    )
    from .ensemble import (
        ensemble_structural_keys as ensemble_structural_keys,
    )
    from .ensemble import (
        merge_ensemble_fixed_space as merge_ensemble_fixed_space,
    )
    from .ensemble import (
        prepare_ensemble_model as prepare_ensemble_model,
    )

sys.modules[__name__] = _implementation
