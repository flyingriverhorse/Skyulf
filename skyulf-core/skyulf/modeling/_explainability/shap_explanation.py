"""SHAP-based model explainability for a trained sklearn-style estimator.

Computes both a global, per-feature mean(|SHAP value|) summary (for a bar
chart, comparable across runs) and a capped set of per-sample explanations
(feature values + SHAP values + base value) that drive richer, single-run
visualisations: beeswarm/distribution plots, dependence plots, and waterfall
plots for an individual prediction — mirroring the graph types the `shap`
library itself provides.

`shap.Explainer` auto-selects the appropriate algorithm (Tree/Linear/Kernel)
based on the model type. `shap` is an optional dependency (see the
`explainability` extra in `setup.py`); callers should treat a `None` return
as "explainability unavailable" rather than an error.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cap on how many rows are kept in the "samples" list returned alongside the
# summary. This is independent of `max_samples` (the SHAP *computation*
# sample size) — we compute over more rows for a stable mean(|value|), but
# only need a much smaller subset for per-row visualisations to keep the
# stored payload small.
_DEFAULT_MAX_DISPLAY_SAMPLES = 50

# Interaction values are O(features^2) per row, so the returned matrix is
# capped to the top-K features (by total interaction strength) to keep the
# payload small regardless of how many columns the dataset has.
_DEFAULT_MAX_INTERACTION_FEATURES = 8


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


def _predicted_class_index(model: Any, sample: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Best-effort per-row predicted-class index for a multi-class (3+) model.

    Falls back to all-zeros (first class) if the model can't predict or its
    `classes_` don't line up with the SHAP output's class axis.
    """
    try:
        classes = list(getattr(model, "classes_", []))
        preds = model.predict(sample)
        if classes and len(classes) == n_classes:
            return np.array([classes.index(p) if p in classes else 0 for p in preds], dtype=int)
    except Exception:
        logger.debug("Falling back to class 0 for per-row SHAP selection", exc_info=True)
    return np.zeros(len(sample), dtype=int)


def _resolve_base_per_row(
    expected_value: Any, n_samples: int, n_classes: int | None, class_idx: np.ndarray | None
) -> np.ndarray:
    """Reduce SHAP's `base_values` (which may be scalar, per-class, per-row,
    or per-row-per-class) down to one base value per row.

    Newer `shap` versions return `base_values` shaped `(n_samples, n_classes)`
    for multi-output models — a naive class-only or row-only reduction would
    silently pick the wrong value, so every shape is handled explicitly.
    """
    base_arr = np.asarray(expected_value, dtype=float)

    if base_arr.ndim == 0:
        return np.full(n_samples, float(base_arr))

    if base_arr.ndim == 1:
        if class_idx is not None and n_classes is not None and base_arr.shape[0] == n_classes:
            # Constant per-class expected value, broadcast via each row's class.
            return base_arr[class_idx]
        if base_arr.shape[0] == n_samples:
            return base_arr
        return np.full(n_samples, float(base_arr[0]))

    if base_arr.ndim == 2:
        # (n_samples, n_classes) — index each row by its own resolved class.
        if class_idx is not None and base_arr.shape[0] == n_samples:
            cols = np.clip(class_idx, 0, base_arr.shape[1] - 1)
            return base_arr[np.arange(n_samples), cols]
        return np.full(n_samples, float(base_arr.ravel()[0]))

    return np.full(n_samples, float(np.ravel(base_arr)[0]))


def _per_sample_shap_and_base(
    shap_values: Any, expected_value: Any, model: Any, sample: pd.DataFrame
) -> tuple[np.ndarray, list[float]] | None:
    """Resolve per-row SHAP values and a matching per-row base value.

    For binary classification, always uses the positive class (index 1) —
    the conventional choice for SHAP visualisations. For multi-class (3+),
    uses each row's predicted class so waterfall/dependence plots reflect
    the class the model actually chose for that row. Regression / other
    single-output SHAP arrays pass through unchanged.
    """
    values = np.asarray(shap_values)
    n_samples = values.shape[0]

    if values.ndim == 2:
        base_per_row = _resolve_base_per_row(expected_value, n_samples, None, None)
        return values, base_per_row.tolist()

    if values.ndim == 3:
        n_classes = values.shape[2]
        class_idx = (
            np.full(n_samples, 1, dtype=int)
            if n_classes == 2
            else _predicted_class_index(model, sample, n_classes)
        )
        rows = np.arange(n_samples)
        shap_rows = values[rows, :, class_idx]
        base_per_row = _resolve_base_per_row(expected_value, n_samples, n_classes, class_idx)
        return shap_rows, base_per_row.tolist()

    return None


def _compute_interaction_summary(
    model: Any,
    sample: pd.DataFrame,
    feature_names: list[str],
    max_interaction_features: int = _DEFAULT_MAX_INTERACTION_FEATURES,
) -> dict[str, Any] | None:
    """Best-effort global SHAP feature-interaction summary for tree models.

    `shap.TreeExplainer.shap_interaction_values` is only implemented for
    tree-based estimators (RandomForest, GradientBoosting, XGBoost, etc.) —
    it raises for anything else (linear/kernel models), which is caught here
    and treated as "unavailable" rather than an error.

    Interaction values are O(features^2) per row, so this returns a single
    mean(|value|) matrix aggregated over all sampled rows (and, for
    multi-class models, over classes too), capped to the top-K features by
    total interaction strength to keep the payload small.

    Returns `None` if the model isn't tree-based or computation fails.
    """
    try:
        import shap  # ty: ignore[unresolved-import]

        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_interaction_values(sample)
    except Exception:
        logger.debug(
            "SHAP interaction values unavailable for model_type=%s",
            type(model).__name__,
            exc_info=True,
        )
        return None

    values = np.asarray(raw)
    n_features = len(feature_names)

    if values.ndim == 4:
        # Multi-class: (n_samples, n_features, n_features, n_classes).
        if values.shape[1] != n_features or values.shape[2] != n_features:
            return None
        mean_abs = np.abs(values).mean(axis=(0, 3))
    elif values.ndim == 3:
        # Binary/regression: (n_samples, n_features, n_features).
        if values.shape[1] != n_features or values.shape[2] != n_features:
            return None
        mean_abs = np.abs(values).mean(axis=0)
    else:
        return None

    if n_features > max_interaction_features:
        # Keep the top-K features by total interaction strength (row sums),
        # preserving a square, symmetric matrix over the reduced feature set.
        strength = mean_abs.sum(axis=1)
        top_idx = np.argsort(strength)[::-1][:max_interaction_features]
        top_idx = np.sort(top_idx)
        mean_abs = mean_abs[np.ix_(top_idx, top_idx)]
        selected_names = [feature_names[i] for i in top_idx]
    else:
        selected_names = feature_names

    return {
        "feature_names": selected_names,
        "matrix": [[round(float(v), 6) for v in row] for row in mean_abs],
    }


def compute_shap_explanation(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 200,
    max_display_samples: int = _DEFAULT_MAX_DISPLAY_SAMPLES,
) -> dict[str, Any] | None:
    """Compute a SHAP explanation for a trained model: a global summary plus
    a small set of per-sample explanations for richer single-run plots.

    Best-effort: returns `None` (never raises) if `shap` isn't installed,
    the model type is unsupported, or computation fails for any reason.

    Returns a dict shaped as:
        {
            "feature_names": [...],
            "mean_abs_importance": {feature: value, ...},
            "samples": [
                {
                    "base_value": float,
                    "feature_values": {feature: value, ...},
                    "shap_values": {feature: value, ...},
                },
                ...
            ],
            "interactions": {
                "feature_names": [...],  # top-K features, or None if unavailable
                "matrix": [[...], ...],  # mean(|interaction value|), same order as feature_names
            } | None,
        }
    """
    try:
        import shap  # ty: ignore[unresolved-import]
        import shap.maskers  # ty: ignore[unresolved-import]
    except ImportError:
        return None

    try:
        if X is None or X.empty:
            return None

        sample = X.sample(n=max_samples, random_state=42) if len(X) > max_samples else X

        feature_names = list(sample.columns)
        if not feature_names:
            return None

        # `shap.Explainer(model, sample)` builds a default `Independent` masker
        # whose own `max_samples` defaults to 100, independent of the `sample`
        # trimming above — so a `sample` between 101 and `max_samples` rows
        # would silently get re-subsampled a second time and warn. Build the
        # masker explicitly so it matches the size we already chose.
        masker = shap.maskers.Independent(sample, max_samples=len(sample))
        explainer = shap.Explainer(model, masker)
        explanation = explainer(sample)
        shap_values = getattr(explanation, "values", explanation)

        mean_abs_importance = _mean_abs_per_feature(shap_values, feature_names)
        if mean_abs_importance is None:
            return None

        resolved = _per_sample_shap_and_base(
            shap_values, getattr(explanation, "base_values", 0.0), model, sample
        )
        samples: list[dict[str, Any]] = []
        if resolved is not None:
            shap_rows, base_values = resolved
            display_sample = sample.iloc[:max_display_samples]
            for i in range(len(display_sample)):
                row = display_sample.iloc[i]
                samples.append(
                    {
                        "base_value": round(float(base_values[i]), 6),
                        "feature_values": {
                            name: round(float(row[name]), 6) for name in feature_names
                        },
                        "shap_values": {
                            name: round(float(shap_rows[i, j]), 6)
                            for j, name in enumerate(feature_names)
                        },
                    }
                )

        interactions = _compute_interaction_summary(model, sample, feature_names)

        return {
            "feature_names": feature_names,
            "mean_abs_importance": mean_abs_importance,
            "samples": samples,
            "interactions": interactions,
        }
    except Exception:
        logger.debug(
            "Failed to compute SHAP explanation for model_type=%s",
            type(model).__name__,
            exc_info=True,
        )
        return None
