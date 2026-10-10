"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.tuning.local_search_results`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.tuning import search_results as _implementation

if TYPE_CHECKING:
    from ..training.tuning.search_results import (
        LocalCVSpec as LocalCVSpec,
    )
    from ..training.tuning.search_results import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from ..training.tuning.search_results import (
        parameter_preview as parameter_preview,
    )
    from ..training.tuning.search_results import (
        post_selection_cv as post_selection_cv,
    )
    from ..training.tuning.search_results import (
        tuning_evidence as tuning_evidence,
    )
    from ..training.tuning.search_results import (
        tuning_run_params as tuning_run_params,
    )
    from ..training.tuning.search_results import (
        validate_search_membership as validate_search_membership,
    )

sys.modules[__name__] = _implementation
