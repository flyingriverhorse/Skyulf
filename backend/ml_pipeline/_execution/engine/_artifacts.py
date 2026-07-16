"""Artifact persistence helpers for :class:`PipelineEngine`.

Mixin slice — owns: feature-importance extraction, SHAP explainability
summaries, training-artifact finalization, and reference-data persistence
for drift detection.

These methods rely on attributes provided by :class:`PipelineEngine`
(``artifact_store``, ``dataset_name``, ``log``).
"""

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from skyulf.data.dataset import SplitDataset
from skyulf.modeling._explainability import compute_shap_explanation

if TYPE_CHECKING:
    from ...artifacts.store import ArtifactStore

logger = logging.getLogger(__name__)


class ArtifactsMixin:
    """Persistence helpers split out of :class:`PipelineEngine`."""

    # Type-only stubs so ty resolves attributes/methods supplied by
    # the concrete :class:`PipelineEngine` (or its sibling mixins) at
    # runtime via the cooperative-mixin pattern. No runtime impact.
    artifact_store: "ArtifactStore"
    dataset_name: str | None
    log: Callable[[str], None]

    def _feature_names_from_split_train(self, train: Any, target_col: str) -> list[str]:
        """Resolve feature names from the `.train` side of a split dataset."""
        if isinstance(train, pd.DataFrame):
            return [c for c in train.columns if c != target_col]
        if isinstance(train, tuple) and len(train) >= 1 and hasattr(train[0], "columns"):
            return list(train[0].columns)
        return []

    def _feature_names_from_tuple(self, data: Any) -> list[str]:
        """Resolve feature names from an (X, y)-style tuple, or `[]` if not shaped that way."""
        if isinstance(data, tuple) and len(data) >= 1 and hasattr(data[0], "columns"):
            return list(data[0].columns)
        return []

    def _feature_names_for_importance(self, data: Any, target_col: str) -> list[str]:
        """Resolve the feature column names for `data`, excluding the target column."""
        if isinstance(data, pd.DataFrame):
            return [c for c in data.columns if c != target_col]
        if hasattr(data, "train"):
            return self._feature_names_from_split_train(data.train, target_col)
        return self._feature_names_from_tuple(data)

    def _model_importance_values(self, actual_model: Any) -> Any | None:
        """Read raw importance/coefficient values off a trained sklearn-style model."""
        if hasattr(actual_model, "feature_importances_"):
            return actual_model.feature_importances_
        if hasattr(actual_model, "coef_"):
            coef = actual_model.coef_
            # For multi-class, coef_ is 2D — take mean of absolute values
            if hasattr(coef, "ndim") and coef.ndim > 1:
                return abs(coef).mean(axis=0)
            return abs(coef)
        return None

    def _extract_feature_importances(
        self, model: Any, data: Any, target_col: str
    ) -> dict[str, float] | None:
        """Extract feature importances from a trained sklearn-style model."""
        try:
            # Unwrap tuple (model, tuning_result) from advanced tuning
            actual_model = model[0] if isinstance(model, tuple) else model

            feature_names = self._feature_names_for_importance(data, target_col)
            if not feature_names:
                return None

            importances = self._model_importance_values(actual_model)

            if importances is not None and len(importances) == len(feature_names):
                return {
                    name: round(float(val), 6)
                    for name, val in zip(feature_names, importances, strict=True)
                }
        except Exception:
            logger.debug(
                "Failed to extract feature importances for step_type=%s",
                type(model).__name__,
                exc_info=True,
            )
        return None

    def _shap_input_frame(self, data: Any, target_col: str) -> pd.DataFrame | None:
        """Resolve a feature-only DataFrame to sample from for SHAP, or `None`."""
        train_df = self._normalize_train_frame(data, target_col)
        if train_df is None or train_df.empty:
            return None
        return train_df.drop(columns=[target_col], errors="ignore")

    def _extract_shap_explanation(
        self, model: Any, data: Any, target_col: str, max_samples: int = 200
    ) -> dict[str, Any] | None:
        """Compute a SHAP explanation for a trained model.

        Resolves the pipeline-specific data shape (`SplitDataset`, `(X, y)`
        tuple, or plain DataFrame) into a feature-only DataFrame, then
        delegates the actual SHAP computation to
        :func:`skyulf.modeling._explainability.compute_shap_explanation`.
        Best-effort: returns `None` if `shap` isn't installed, the model
        type is unsupported, or computation fails for any reason.

        Returns a dict with `feature_names`, `mean_abs_importance` (for the
        cross-run comparison bar chart) and `samples` (per-row feature/SHAP
        values for beeswarm, dependence, and waterfall plots of a single run).
        """
        X = self._shap_input_frame(data, target_col)
        if X is None:
            return None

        actual_model = model[0] if isinstance(model, tuple) else model
        return compute_shap_explanation(actual_model, X, max_samples=max_samples)

    def _finalize_training_artifacts(
        self, data: Any, job_id: str, target_col: str, node_id: str, model_artifact: Any
    ):
        """
        Finalize training by saving standard artifacts:
        1. Reference Data for Drift Detection (if not already present).
        2. Model Artifact (node_id and job_id).
        """
        # Save Reference Data for Drift Detection
        # Only overwrite if it doesn't exist (prefer Raw/Splitter data over Scaled/Transformed data)
        # Construct the key manually to check existence
        if job_id and job_id != "unknown":
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", self.dataset_name or "")
            key = f"reference_data_{safe_name}_{job_id}"
            if not self.artifact_store.exists(key):
                self._save_reference_data(data, job_id, target_col)
        else:
            self._save_reference_data(data, job_id, target_col)

        # Manually save the model artifact
        self.artifact_store.save(node_id, model_artifact)
        if job_id and job_id != "unknown":
            self.log(f"Saving model artifact to job key: {job_id}")
            self.artifact_store.save(job_id, model_artifact)

    def _normalize_train_frame(self, data: Any, target_col: str) -> pd.DataFrame | None:
        """Extract and normalize training data to a DataFrame, or `None` if not derivable.

        Handles `SplitDataset` (extracts `.train`) and DataFrame/(X, y)-tuple formats.
        """
        raw_train = data.train if isinstance(data, SplitDataset) else data

        if isinstance(raw_train, pd.DataFrame):
            return raw_train
        if isinstance(raw_train, tuple) and len(raw_train) == 2:
            # (X, y) tuple
            X, y = raw_train
            if isinstance(X, pd.DataFrame):
                train_df = X.copy()
                # Add target column back if y is compatible
                if isinstance(y, (pd.Series, np.ndarray, list)):
                    train_df[target_col] = y
                return train_df
        return None

    def _persist_reference_frame(self, train_df: pd.DataFrame | None, job_id: str) -> None:
        """Save `train_df` as the reference dataset for `job_id`, or warn if it's unusable."""
        if train_df is not None and not train_df.empty:
            # Sanitize dataset name
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", self.dataset_name or "")
            key = f"reference_data_{safe_name}_{job_id}"

            self.log(f"Saving reference training data for drift detection: {key}")
            self.artifact_store.save(key, train_df)
        else:
            logger.warning(f"Could not extract reference data for job {job_id}")

    def _save_reference_data(self, data: Any, job_id: str, target_col: str):
        """
        Saves the training data as a reference dataset for future drift detection.
        Handles SplitDataset (extracts train) and DataFrame/Tuple formats.
        """
        if not job_id or job_id == "unknown":
            return

        try:
            train_df = self._normalize_train_frame(data, target_col)
            self._persist_reference_frame(train_df, job_id)
        except Exception as e:
            logger.warning(f"Failed to save reference data: {e}")
