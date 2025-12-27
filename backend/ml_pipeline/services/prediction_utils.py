"""
Prediction Utilities
--------------------
Shared helper functions for decoding predictions and extracting label encoders.
Used by DeploymentService and EvaluationService.
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def extract_target_label_encoder(feature_engineer: Any) -> Optional[Any]:
    """
    Extracts the target LabelEncoder from a fitted FeatureEngineer pipeline.
    Walks backwards through steps to find the most recent LabelEncoder with a '__target__' encoder.
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
        target_encoder = encoders.get("__target__")
        if target_encoder is not None and hasattr(target_encoder, "inverse_transform"):
            return target_encoder

    return None


def decode_int_like(values: List[Any], label_encoder: Any) -> List[Any]:
    """
    Best-effort decode for lists of encoded class indices.
    If values are not int-like (or decoding fails), returns the original list.
    """
    try:
        import numpy as np

        arr = np.asarray(values)
        # Check if numeric
        if arr.dtype.kind in {"i", "u", "b", "f"}:
             decoded = label_encoder.inverse_transform(arr.astype(int))
             return decoded.tolist() if hasattr(decoded, "tolist") else list(decoded)
        return values
    except Exception:
        return values
