"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.tuning.local_search`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.tuning import search as _implementation

if TYPE_CHECKING:
    from ..training.tuning.search import (
        base_model_config as base_model_config,
    )
    from ..training.tuning.search import (
        bounded_space as bounded_space,
    )
    from ..training.tuning.search import (
        prepare_search_pipeline as prepare_search_pipeline,
    )
    from ..training.tuning.search import (
        validate_metric as validate_metric,
    )

sys.modules[__name__] = _implementation
