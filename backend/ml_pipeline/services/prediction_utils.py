"""
Prediction Utilities
--------------------
Shared helper functions for decoding predictions and extracting label encoders.
Used by DeploymentService and EvaluationService.
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def extract_target_label_encoder(
    feature_engineer: Any,
    target_column: Optional[str] = None,
) -> Optional[Any]:
    """Extract the target LabelEncoder from a fitted FeatureEngineer pipeline.

    Resolution order (walks steps backwards, most recent wins):
      1. ``encoders["__target__"]`` — explicit target encoder (LabelEncoder with no columns,
         or columns list that includes the target after Feature/Target Split).
      2. ``encoders[target_column]`` — fallback for pipelines where the LabelEncoder ran
         *before* the Feature/Target Split, so the target was encoded as a regular
         feature column rather than under the ``__target__`` key.
    """
    fitted_steps = getattr(feature_engineer, "fitted_steps", None)
    if not isinstance(fitted_steps, list):
        return None

    # Walk backwards so the most recent LabelEncoder wins
    for step in reversed(fitted_steps):
        if not isinstance(step, dict):
            continue
        if step.get("type") != "LabelEncoder":
            continue
        artifact = step.get("artifact")
        if not isinstance(artifact, dict):
            continue
        encoders = artifact.get("encoders")
        if not isinstance(encoders, dict):
            continue

        # Priority 1: explicit __target__ key
        target_encoder = encoders.get("__target__")
        if target_encoder is not None and hasattr(target_encoder, "inverse_transform"):
            return target_encoder

        # Priority 2: encoder keyed by the target column name
        # (happens when LabelEncoder runs before Feature/Target Split)
        if target_column:
            col_encoder = encoders.get(target_column)
            if col_encoder is not None and hasattr(col_encoder, "inverse_transform"):
                return col_encoder

    return None


def decode_int_like(values: List[Any], label_encoder: Any) -> List[Any]:
    """
    Best-effort decode for lists of encoded class indices.
    If values are not int-like (or decoding fails), returns the original list.
    """
    try:
        import numpy as np

        arr = np.asarray(values)
        if arr.size == 0:
            return values

        # Only attempt decoding for numeric arrays that are *integer-like*.
        # This prevents accidentally decoding true regression targets/predictions
        # (floats) when a target LabelEncoder exists in the artifact bundle.
        if arr.dtype.kind not in {"i", "u", "b", "f"}:
            return values

        if not np.all(np.isfinite(arr)):
            return values

        int_arr = arr.astype(int)
        if not np.allclose(arr, int_arr):
            return values

        decoded = label_encoder.inverse_transform(int_arr)
        return decoded.tolist() if hasattr(decoded, "tolist") else list(decoded)
    except Exception:
        return values
