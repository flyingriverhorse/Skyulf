"""Track exact built-in outlier filters without losing requested row identities."""

from typing import Any

import numpy as np

from ...registry import NodeRegistry
from .elliptic import EllipticEnvelopeApplier
from .iqr import IQRApplier
from .manual_bounds import ManualBoundsApplier
from .zscore import ZScoreApplier

OUTLIER_APPLIERS = {
    "IQR": IQRApplier,
    "ZScore": ZScoreApplier,
    "EllipticEnvelope": EllipticEnvelopeApplier,
    "ManualBounds": ManualBoundsApplier,
}


def is_outlier_step(record: dict) -> bool:
    """Recognize only original, stateless registered filter implementations."""
    owner = OUTLIER_APPLIERS.get(record["type"])
    if owner is None:
        return False
    applier = record["applier"]
    if (
        type(applier) is not owner
        or vars(applier)
        or NodeRegistry.get_applier(record["type"]) is not owner
    ):
        raise ValueError("Outlier exclusion requires an unchanged built-in applier.")
    return True


def apply_outlier_filter(frame: Any, record: dict) -> tuple[Any, np.ndarray]:
    """Reuse the filter's actual target slicing to identify survivors positionally."""
    if not is_outlier_step(record):
        raise ValueError("Only fitted built-in outlier filters support exclusion tracking.")
    positions = np.arange(len(frame))
    transformed, kept = record["applier"].apply((frame, positions), record["artifact"])
    kept = np.asarray(kept)
    if (
        kept.dtype.kind not in "iu"
        or kept.ndim != 1
        or len(kept) != len(transformed)
        or not np.array_equal(kept, positions[np.isin(positions, kept)])
    ):
        raise ValueError("Outlier filter did not preserve an ordered subset of input rows.")
    return transformed, kept
