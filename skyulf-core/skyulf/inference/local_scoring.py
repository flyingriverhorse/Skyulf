"""Compatibility alias for :mod:`skyulf.inference.pipeline_scoring`."""

import sys
from typing import TYPE_CHECKING

from skyulf.inference import pipeline_scoring as _implementation

if TYPE_CHECKING:
    from skyulf.inference.pipeline_scoring import (
        LocalPrediction as LocalPrediction,
    )
    from skyulf.inference.pipeline_scoring import (
        PipelinePrediction as PipelinePrediction,
    )
    from skyulf.inference.pipeline_scoring import (
        _preserve_history as _preserve_history,
    )
    from skyulf.inference.pipeline_scoring import (
        _validate_history_request as _validate_history_request,
    )
    from skyulf.inference.pipeline_scoring import (
        local_history_session as local_history_session,
    )
    from skyulf.inference.pipeline_scoring import (
        pipeline_history_session as pipeline_history_session,
    )
    from skyulf.inference.pipeline_scoring import (
        prediction_output_schema as prediction_output_schema,
    )
    from skyulf.inference.pipeline_scoring import (
        score_local_pipeline as score_local_pipeline,
    )
    from skyulf.inference.pipeline_scoring import (
        score_local_pipeline_with_history as score_local_pipeline_with_history,
    )
    from skyulf.inference.pipeline_scoring import (
        score_pipeline as score_pipeline,
    )
    from skyulf.inference.pipeline_scoring import (
        score_pipeline_with_history as score_pipeline_with_history,
    )
    from skyulf.inference.pipeline_scoring import (
        scoring_counts as scoring_counts,
    )
    from skyulf.inference.pipeline_scoring import (
        scoring_output_schema as scoring_output_schema,
    )

sys.modules[__name__] = _implementation
