"""
Prediction Utilities
--------------------
Shared helper functions for decoding predictions and extracting label encoders.
Used by DeploymentService and EvaluationService.
"""

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


def extract_target_label_encoder(
    feature_engineer: Any,
    target_column: str | None = None,
) -> Any | None:
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
    for raw_step in reversed(fitted_steps):
        encoders = _get_label_encoder_map(raw_step)
        if encoders is None:
            continue
        encoder = _resolve_target_encoder(encoders, target_column)
        if encoder is not None:
            return encoder

    return None


def _get_label_encoder_map(raw_step: Any) -> dict[str, Any] | None:
    """Return the `encoders` mapping from a fitted LabelEncoder step, if any."""
    if not isinstance(raw_step, dict):
        return None
    step = cast(dict[str, Any], raw_step)
    if step.get("type") != "LabelEncoder":
        return None
    artifact = step.get("artifact")
    if not isinstance(artifact, dict):
        return None
    encoders = artifact.get("encoders")
    if not isinstance(encoders, dict):
        return None
    return encoders


def _resolve_target_encoder(encoders: dict[str, Any], target_column: str | None) -> Any | None:
    """Pick the target's LabelEncoder from a step's `encoders` mapping.

    Priority 1: explicit ``__target__`` key. Priority 2: encoder keyed by the
    target column name (happens when LabelEncoder runs before Feature/Target
    Split).
    """
    target_encoder = encoders.get("__target__")
    if target_encoder is not None and hasattr(target_encoder, "inverse_transform"):
        return target_encoder

    if target_column:
        col_encoder = encoders.get(target_column)
        if col_encoder is not None and hasattr(col_encoder, "inverse_transform"):
            return col_encoder

    return None


def _to_int_like_array(arr: Any) -> Any | None:
    """Coerce `arr` to an integer numpy array if it losslessly represents integer-like values.

    Returns None when the array can't be treated as integer-like (e.g. it's
    non-numeric, contains non-finite values, or has fractional components).
    """
    import numpy as np

    # y_proba's "classes" list comes from DataFrame column names (e.g.
    # ``model.classes_``/``predict_proba`` columns), which sklearn/pandas
    # often stringifies (e.g. "0"/"1"/"2") even though they represent the
    # same encoded integer indices as y_true/y_pred. Without this, such
    # arrays have dtype.kind "U"/"O" and were silently skipped below,
    # leaving y_proba's "classes"/"labels" undecoded while y_true/y_pred
    # (real int arrays) decoded fine — causing the frontend's positive-
    # class lookup (which matches against decoded "labels") to fail.
    if arr.dtype.kind in {"U", "S", "O"}:
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError):
            return None

    # Only attempt decoding for numeric arrays that are *integer-like*.
    # This prevents accidentally decoding true regression targets/predictions
    # (floats) when a target LabelEncoder exists in the artifact bundle.
    if arr.dtype.kind not in {"i", "u", "b", "f"}:
        return None

    if not np.all(np.isfinite(arr)):
        return None

    int_arr = arr.astype(int)
    if not np.allclose(arr, int_arr):
        return None

    return int_arr


def decode_int_like(values: list[Any], label_encoder: Any) -> list[Any]:
    """
    Best-effort decode for lists of encoded class indices.
    If values are not int-like (or decoding fails), returns the original list.
    """
    try:
        import numpy as np

        arr = np.asarray(values)
        if arr.size == 0:
            return values

        int_arr = _to_int_like_array(arr)
        if int_arr is None:
            return values

        decoded = label_encoder.inverse_transform(int_arr)
        return decoded.tolist() if hasattr(decoded, "tolist") else list(decoded)
    except Exception:
        logging.getLogger(__name__).debug(
            "Label decoding failed, returning raw values", exc_info=True
        )
        return values
