"""Compatibility alias for :mod:`skyulf.integrations.databricks.observability.reports.local_explanations`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.observability.reports import (
    explanations as _implementation,
)

if TYPE_CHECKING:
    from ..observability.reports.explanations import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from ..observability.reports.explanations import (
        explain_training_artifact as explain_training_artifact,
    )
    from ..observability.reports.explanations import (
        log_training_explanations as log_training_explanations,
    )
    from ..observability.reports.explanations import (
        validate_explanation_config as validate_explanation_config,
    )

sys.modules[__name__] = _implementation
