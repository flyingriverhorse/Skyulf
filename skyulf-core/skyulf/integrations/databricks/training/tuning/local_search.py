"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.tuning.search`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.tuning import search as _implementation

if TYPE_CHECKING:
    from .search import (
        base_model_config as base_model_config,
    )
    from .search import (
        bounded_space as bounded_space,
    )
    from .search import (
        prepare_search_pipeline as prepare_search_pipeline,
    )
    from .search import (
        validate_metric as validate_metric,
    )

sys.modules[__name__] = _implementation
