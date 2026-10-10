"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.tuning.search_results`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.tuning import search_results as _implementation

if TYPE_CHECKING:
    from .search_results import (
        LocalCVSpec as LocalCVSpec,
    )
    from .search_results import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from .search_results import (
        parameter_preview as parameter_preview,
    )
    from .search_results import (
        post_selection_cv as post_selection_cv,
    )
    from .search_results import (
        tuning_evidence as tuning_evidence,
    )
    from .search_results import (
        tuning_run_params as tuning_run_params,
    )
    from .search_results import (
        validate_search_membership as validate_search_membership,
    )

sys.modules[__name__] = _implementation
