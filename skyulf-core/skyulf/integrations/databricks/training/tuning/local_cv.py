"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.tuning.cv`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.tuning import cv as _implementation

if TYPE_CHECKING:
    from .cv import (
        CV_FIELDS as CV_FIELDS,
    )
    from .cv import (
        CVSpec as CVSpec,
    )
    from .cv import (
        LocalCVSpec as LocalCVSpec,
    )
    from .cv import (
        evaluate_training_cv as evaluate_training_cv,
    )
    from .cv import (
        validate_fold_membership as validate_fold_membership,
    )

sys.modules[__name__] = _implementation
