"""
Evaluation Service
------------------
Service for retrieving and processing evaluation results (y_true, y_pred)
for training and tuning jobs.
"""

import logging
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.ml_pipeline._services.job_service import JobService
from backend.ml_pipeline._services.prediction_utils import (
    decode_int_like,
    extract_target_label_encoder,
)
from backend.ml_pipeline.artifacts.factory import ArtifactFactory

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for managing evaluation data retrieval."""

    @staticmethod
    def _resolve_artifact_uri(job: Any, job_id: str) -> str:
        """Resolves the artifact URI for a job, falling back to the default training artifact dir."""
        artifact_uri = str(job.artifact_uri)
        if not artifact_uri:
            # Fallback for old jobs or local jobs without explicit URI
            settings = get_settings()
            artifact_uri = str(Path(settings.TRAINING_ARTIFACT_DIR) / job_id)
        return artifact_uri

    @staticmethod
    def _resolve_evaluation_key(artifact_store: Any, job: Any, job_id: str) -> str:
        """Finds the evaluation-data artifact key for a job, raising if it isn't available."""
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
                raise ValueError(f"Job is {job.status}, evaluation data not available yet.")
            raise FileNotFoundError(f"Evaluation data not found for job {job_id}")

        return key

    @staticmethod
    def _check_job_id_matches(data: Any, job_id: str) -> None:
        """Raises if loaded evaluation data belongs to a different job than requested.

        A mismatch means we are about to serve stale/foreign evaluation results (e.g. a
        leftover artifact from a previous job that reused the same node_id key) — treat
        this as an error rather than silently returning it.
        """
        if isinstance(data, dict) and data.get("job_id") != job_id:
            logger.error(
                f"Evaluation data job_id mismatch. Requested: {job_id}, Found: {data.get('job_id')}"
            )
            raise ValueError(
                f"Evaluation data job_id mismatch for job {job_id} "
                f"(found data belonging to job {data.get('job_id')!r})"
            )

    @staticmethod
    def _decode_split_labels(split_data: dict[str, Any], label_encoder: Any) -> None:
        """Decodes y_true/y_pred/y_proba classes in-place for a single evaluation split."""
        if isinstance(split_data.get("y_true"), list):
            split_data["y_true"] = decode_int_like(split_data["y_true"], label_encoder)
        if isinstance(split_data.get("y_pred"), list):
            split_data["y_pred"] = decode_int_like(split_data["y_pred"], label_encoder)

        y_proba = split_data.get("y_proba")
        if isinstance(y_proba, dict) and isinstance(y_proba.get("classes"), list):
            # Preserve original classes (often numeric) but also attach decoded labels.
            y_proba["labels"] = decode_int_like(y_proba["classes"], label_encoder)

    @staticmethod
    def _load_target_label_encoder(artifact_store: Any, job_id: str) -> Any:
        """Loads the job bundle and extracts its target label encoder, if any."""
        if not artifact_store.exists(job_id):
            return None
        bundle = artifact_store.load(job_id)
        if not (isinstance(bundle, dict) and "feature_engineer" in bundle):
            return None
        feature_engineer = bundle.get("feature_engineer")
        target_col_name = bundle.get("target_column")
        return extract_target_label_encoder(feature_engineer, target_column=target_col_name)

    @staticmethod
    def _decode_target_labels(data: Any, artifact_store: Any, job_id: str) -> None:
        """Best-effort decode of target labels (ROC selector, confusion matrix) in-place.

        Silently skips if no label encoder/feature engineer bundle is available.
        """
        try:
            label_encoder = EvaluationService._load_target_label_encoder(artifact_store, job_id)

            if (
                label_encoder is not None
                and isinstance(data, dict)
                and data.get("problem_type") == "classification"
            ):
                splits = data.get("splits")
                if isinstance(splits, dict):
                    for split_data in splits.values():
                        if not isinstance(split_data, dict):
                            continue
                        EvaluationService._decode_split_labels(split_data, label_encoder)
        except Exception as e:
            logger.debug(f"Evaluation decode skipped/failed: {e}")

    @staticmethod
    async def get_job_evaluation(session: AsyncSession, job_id: str) -> dict[str, Any]:
        """
        Retrieves the raw evaluation data (y_true, y_pred) for a job.
        Decodes target labels if a LabelEncoder was used.
        """
        # 1. Get Job Info
        job = await JobService.get_job_by_id(session, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # 2. Determine Artifact Path
        artifact_uri = EvaluationService._resolve_artifact_uri(job, job_id)
        artifact_store = ArtifactFactory.get_artifact_store(artifact_uri)

        # 3. Load Evaluation Artifact
        key = EvaluationService._resolve_evaluation_key(artifact_store, job, job_id)

        try:
            data = artifact_store.load(key)
            # Verify it belongs to this job (since we share the folder).
            EvaluationService._check_job_id_matches(data, job_id)

            # Optional: decode target labels for nicer UI (ROC selector, confusion matrix)
            EvaluationService._decode_target_labels(data, artifact_store, job_id)

            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load evaluation data: {str(e)}") from e
