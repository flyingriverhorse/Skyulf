"""Track exact built-in exclusion filters without losing requested row identities."""

from typing import Any

import numpy as np

from ..registry import NodeRegistry
from .outliers.elliptic import EllipticEnvelopeApplier
from .outliers.iqr import IQRApplier
from .outliers.manual_bounds import ManualBoundsApplier
from .outliers.zscore import ZScoreApplier
from .transformations.power import PowerTransformerApplier

EXCLUSION_APPLIERS = {
    "IQR": IQRApplier,
    "ZScore": ZScoreApplier,
    "EllipticEnvelope": EllipticEnvelopeApplier,
    "ManualBounds": ManualBoundsApplier,
    "PowerTransformer": PowerTransformerApplier,
}


def is_exclusion_step(record: dict) -> bool:
    """Recognize only original, stateless registered filter implementations."""
    owner = EXCLUSION_APPLIERS.get(record["type"])
    if owner is None:
        return False
    applier = record["applier"]
    if (
        type(applier) is not owner
        or vars(applier)
        or NodeRegistry.get_applier(record["type"]) is not owner
    ):
        raise ValueError("Row exclusion requires an unchanged built-in applier.")
    return True


def exclusion_reason(record: dict) -> str:
    """Distinguish invalid transformation values from fitted outlier decisions."""
    prefix = "invalid" if record["type"] == "PowerTransformer" else "outlier"
    return f"{prefix}:{record['name']}"


def exclusion_action(record: dict) -> str:
    """Name the reviewed filtering semantics in diagnostic and worker certificates."""
    return "exclude_invalid" if record["type"] == "PowerTransformer" else "exclude_outliers"


def apply_exclusion_filter(frame: Any, record: dict) -> tuple[Any, np.ndarray]:
    """Reuse the filter's actual target slicing to identify survivors positionally."""
    if not is_exclusion_step(record):
        raise ValueError("Only fitted built-in filters support exclusion tracking.")
    positions = np.arange(len(frame))
    transformed, kept = record["applier"].apply((frame, positions), record["artifact"])
    kept = np.asarray(kept)
    if (
        kept.dtype.kind not in "iu"
        or kept.ndim != 1
        or len(kept) != len(transformed)
        or not np.array_equal(kept, positions[np.isin(positions, kept)])
    ):
        raise ValueError("Exclusion filter did not preserve an ordered subset of input rows.")
    return transformed, kept
