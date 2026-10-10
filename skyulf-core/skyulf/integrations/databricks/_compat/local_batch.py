"""Compatibility alias for :mod:`skyulf.integrations.databricks.scoring.batch.local_batch`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.scoring.batch import frame_batch as _implementation

if TYPE_CHECKING:
    from ..scoring.batch.frame_batch import (
        LocalPipelineArtifact as LocalPipelineArtifact,
    )
    from ..scoring.batch.frame_batch import (
        LocalScoreResult as LocalScoreResult,
    )
    from ..scoring.batch.frame_batch import (
        LocalSourceSpec as LocalSourceSpec,
    )
    from ..scoring.batch.frame_batch import (
        PreparedLocalWorkflow as PreparedLocalWorkflow,
    )
    from ..scoring.batch.frame_batch import (
        ScoreResult as ScoreResult,
    )
    from ..scoring.batch.frame_batch import (
        SourceSpec as SourceSpec,
    )
    from ..scoring.batch.frame_batch import (
        evaluate_local_holdout as evaluate_local_holdout,
    )
    from ..scoring.batch.frame_batch import (
        fit_local_workflow as fit_local_workflow,
    )
    from ..scoring.batch.frame_batch import (
        fit_workflow as fit_workflow,
    )
    from ..scoring.batch.frame_batch import (
        read_local_source as read_local_source,
    )
    from ..scoring.batch.frame_batch import (
        read_source as read_source,
    )
    from ..scoring.batch.frame_batch import (
        score_local_source as score_local_source,
    )
    from ..scoring.batch.frame_batch import (
        score_source as score_source,
    )

sys.modules[__name__] = _implementation
