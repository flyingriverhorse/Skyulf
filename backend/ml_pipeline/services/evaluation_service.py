"""
Evaluation Service
------------------
Service for retrieving and processing evaluation results (y_true, y_pred)
for training and tuning jobs.
"""

import logging
import os
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.ml_pipeline.artifacts.factory import ArtifactFactory
from backend.ml_pipeline.services.job_service import JobService
from backend.ml_pipeline.services.prediction_utils import (
    decode_int_like,
    extract_target_label_encoder,
)

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for managing evaluation data retrieval."""

    @staticmethod
    async def get_job_evaluation(session: AsyncSession, job_id: str) -> Dict[str, Any]:
        """
        Retrieves the raw evaluation data (y_true, y_pred) for a job.
        Decodes target labels if a LabelEncoder was used.
        """
        # 1. Get Job Info
        job = await JobService.get_job_by_id(session, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # 2. Determine Artifact Path
        artifact_uri = str(job.artifact_uri)
        if not artifact_uri:
            # Fallback for old jobs or local jobs without explicit URI
            settings = get_settings()
            artifact_uri = os.path.join(settings.TRAINING_ARTIFACT_DIR, job_id)

        artifact_store = ArtifactFactory.get_artifact_store(artifact_uri)

        # 3. Load Evaluation Artifact
        # Key format: {node_id}_evaluation_data
        # Or {job_id}_evaluation_data if saved with job_id (which we do now)

        # Try job_id key first (preferred)
        key = f"{job_id}_evaluation_data"
        if not artifact_store.exists(key):
            # Fallback to node_id key
            key = f"{job.node_id}_evaluation_data"

        if not artifact_store.exists(key):
            # Fallback: Check if the job failed or is still running
            if job.status not in ["completed", "succeeded"]:
                raise ValueError(
                    f"Job is {job.status}, evaluation data not available yet."
                )

            # Debug info
            path = artifact_store._get_path(key)
            raise FileNotFoundError(
                f"Evaluation data artifact not found. Key: {key}, Path: {path}"
            )

        try:
            data = artifact_store.load(key)
            # Verify it belongs to this job (since we share the folder)
            if isinstance(data, dict) and data.get("job_id") != job_id:
                logger.warning(
                    f"Evaluation data job_id mismatch. Requested: {job_id}, Found: {data.get('job_id')}"
                )

            # Optional: decode target labels for nicer UI (ROC selector, confusion matrix)
            try:
                if artifact_store.exists(job_id):
                    bundle = artifact_store.load(job_id)
                    if isinstance(bundle, dict) and "feature_engineer" in bundle:
                        feature_engineer = bundle.get("feature_engineer")
                        label_encoder = extract_target_label_encoder(feature_engineer)

                        if label_encoder is not None and isinstance(data, dict):
                            splits = data.get("splits")
                            if isinstance(splits, dict):
                                for split_data in splits.values():
                                    if not isinstance(split_data, dict):
                                        continue

                                    if isinstance(split_data.get("y_true"), list):
                                        split_data["y_true"] = decode_int_like(
                                            split_data["y_true"], label_encoder
                                        )
                                    if isinstance(split_data.get("y_pred"), list):
                                        split_data["y_pred"] = decode_int_like(
                                            split_data["y_pred"], label_encoder
                                        )

                                    y_proba = split_data.get("y_proba")
                                    if (
                                        isinstance(y_proba, dict)
                                        and isinstance(y_proba.get("classes"), list)
                                    ):
                                        # Preserve original classes (often numeric) but also attach decoded labels.
                                        y_proba["labels"] = decode_int_like(
                                            y_proba["classes"], label_encoder
                                        )
            except Exception as e:
                logger.debug(f"Evaluation decode skipped/failed: {e}")

            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load evaluation data: {str(e)}")
