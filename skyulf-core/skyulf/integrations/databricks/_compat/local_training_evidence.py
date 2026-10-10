"""Compatibility alias for :mod:`skyulf.integrations.databricks.training.shared.local_training_evidence`."""

import sys
from typing import TYPE_CHECKING

from skyulf.integrations.databricks.training.shared import (
    training_evidence as _implementation,
)

if TYPE_CHECKING:
    from ..training.shared.training_evidence import (
        build_training_evidence as build_training_evidence,
    )
    from ..training.shared.training_evidence import (
        evidence_digest as evidence_digest,
    )
    from ..training.shared.training_evidence import (
        load_candidate_evidence as load_candidate_evidence,
    )
    from ..training.shared.training_evidence import (
        validate_training_evidence as validate_training_evidence,
    )

sys.modules[__name__] = _implementation
