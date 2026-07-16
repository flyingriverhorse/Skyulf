"""SHAP-based model explainability summaries.

Computes a per-feature mean(|SHAP value|) summary for a trained sklearn-style
estimator. This is a lightweight, model-agnostic "global" explanation —
`shap.Explainer` auto-selects the appropriate algorithm (Tree/Linear/Kernel)
based on the model type.

`shap` is an optional dependency (see the `explainability` extra in
`setup.py`); callers should treat a `None` return as "explainability
unavailable" rather than an error.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _mean_abs_per_feature(shap_values: Any, feature_names: list[str]) -> dict[str, float] | None:
    """Reduce a raw SHAP values array to mean(|value|) per feature.

    Handles both single-output (n_samples, n_features) and multi-class
    (n_samples, n_features, n_classes) SHAP value arrays.
    """
    values = np.asarray(shap_values)
    if values.ndim == 3:
        # Multi-class output. Take abs *before* averaging over classes —
        # per-class contributions can have opposite signs (e.g. binary
        # classification), so averaging signed values first cancels out.
        if values.shape[1] != len(feature_names):
            return None
        mean_abs = np.abs(values).mean(axis=(0, 2))
    elif values.ndim == 2:
        if values.shape[1] != len(feature_names):
            return None
        mean_abs = np.abs(values).mean(axis=0)
    else:
        return None
    return {name: round(float(val), 6) for name, val in zip(feature_names, mean_abs, strict=True)}


def compute_shap_summary(
    model: Any, X: pd.DataFrame, max_samples: int = 200
) -> dict[str, float] | None:
    """Compute a mean(|SHAP value|) summary per feature for a trained model.

    Best-effort: returns `None` (never raises) if `shap` isn't installed,
    the model type is unsupported, or computation fails for any reason.
    Sampling is capped at `max_samples` rows to keep computation fast.
    """
    try:
        import shap
    except ImportError:
        return None

    try:
        if X is None or X.empty:
            return None

        sample = X.sample(n=max_samples, random_state=42) if len(X) > max_samples else X

        feature_names = list(sample.columns)
        if not feature_names:
            return None

        explainer = shap.Explainer(model, sample)
        explanation = explainer(sample)
        shap_values = getattr(explanation, "values", explanation)

        return _mean_abs_per_feature(shap_values, feature_names)
    except Exception:
        logger.debug(
            "Failed to compute SHAP summary for model_type=%s",
            type(model).__name__,
            exc_info=True,
        )
        return None
