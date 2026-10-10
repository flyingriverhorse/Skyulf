"""Compatibility alias for :mod:`skyulf.inference.pipeline_evaluation`."""

import sys
from typing import TYPE_CHECKING

from skyulf.inference import pipeline_evaluation as _implementation

if TYPE_CHECKING:
    from skyulf.inference.pipeline_evaluation import (
        _holdout_metrics as _holdout_metrics,
    )
    from skyulf.inference.pipeline_evaluation import (
        _validate_holdout as _validate_holdout,
    )
    from skyulf.inference.pipeline_evaluation import (
        evaluate_holdout as evaluate_holdout,
    )
    from skyulf.inference.pipeline_evaluation import (
        evaluate_local_holdout as evaluate_local_holdout,
    )

sys.modules[__name__] = _implementation
