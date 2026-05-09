"""Artifact persistence helpers for :class:`PipelineEngine`.

Mixin slice — owns: feature-importance extraction, training-artifact
finalization, and reference-data persistence for drift detection.

These methods rely on attributes provided by :class:`PipelineEngine`
(``artifact_store``, ``dataset_name``, ``log``).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from skyulf.data.dataset import SplitDataset

if TYPE_CHECKING:
    from ...artifacts.store import ArtifactStore

logger = logging.getLogger(__name__)


class ArtifactsMixin:
    """Persistence helpers split out of :class:`PipelineEngine`."""

    # Type-only stubs so ty resolves attributes/methods supplied by
    # the concrete :class:`PipelineEngine` (or its sibling mixins) at
    # runtime via the cooperative-mixin pattern. No runtime impact.
    artifact_store: "ArtifactStore"
    dataset_name: Optional[str]
    log: Callable[[str], None]

    def _extract_feature_importances(
        self, model: Any, data: Any, target_col: str
    ) -> Optional[Dict[str, float]]:
        """Extract feature importances from a trained sklearn-style model."""
        try:
            # Unwrap tuple (model, tuning_result) from advanced tuning
            actual_model = model[0] if isinstance(model, tuple) else model

            # Get feature names from data
            feature_names: List[str] = []
            if isinstance(data, pd.DataFrame):
                feature_names = [c for c in data.columns if c != target_col]
            elif hasattr(data, "train"):
                train = data.train
                if isinstance(train, pd.DataFrame):
                    feature_names = [c for c in train.columns if c != target_col]
                elif isinstance(train, tuple) and len(train) >= 1 and hasattr(train[0], "columns"):
                    feature_names = list(train[0].columns)
            elif isinstance(data, tuple) and len(data) >= 1 and hasattr(data[0], "columns"):
                feature_names = list(data[0].columns)

            if not feature_names:
                return None

            # Extract importances
            importances: Optional[Any] = None
            if hasattr(actual_model, "feature_importances_"):
                importances = actual_model.feature_importances_
            elif hasattr(actual_model, "coef_"):
                coef = actual_model.coef_
                # For multi-class, coef_ is 2D — take mean of absolute values
                if hasattr(coef, "ndim") and coef.ndim > 1:
                    importances = abs(coef).mean(axis=0)
                else:
                    importances = abs(coef)

            if importances is not None and len(importances) == len(feature_names):
                return {name: round(float(val), 6) for name, val in zip(feature_names, importances)}
        except Exception:
            pass
        return None

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

    def _save_reference_data(self, data: Any, job_id: str, target_col: str):
        """
        Saves the training data as a reference dataset for future drift detection.
        Handles SplitDataset (extracts train) and DataFrame/Tuple formats.
        """
        if not job_id or job_id == "unknown":
            return

        try:
            train_df = None

            # 1. Extract Training Data
            if isinstance(data, SplitDataset):
                raw_train = data.train
            else:
                raw_train = data  # Assume it's the full dataset if not split

            # 2. Normalize to DataFrame
            if isinstance(raw_train, pd.DataFrame):
                train_df = raw_train
            elif isinstance(raw_train, tuple) and len(raw_train) == 2:
                # (X, y) tuple
                X, y = raw_train
                if isinstance(X, pd.DataFrame):
                    train_df = X.copy()
                    # Add target column back if y is compatible
                    if isinstance(y, (pd.Series, np.ndarray, list)):
                        train_df[target_col] = y

            # 3. Save if valid
            if train_df is not None and not train_df.empty:
                # Sanitize dataset name
                safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", self.dataset_name or "")
                key = f"reference_data_{safe_name}_{job_id}"

                self.log(f"Saving reference training data for drift detection: {key}")
                self.artifact_store.save(key, train_df)
            else:
                logger.warning(f"Could not extract reference data for job {job_id}")

        except Exception as e:
            logger.warning(f"Failed to save reference data: {e}")
