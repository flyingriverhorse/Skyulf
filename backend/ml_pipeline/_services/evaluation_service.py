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
    extract_column_label_encoder,
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
    def _load_feature_engineer_bundle(
        artifact_store: Any, job_id: str
    ) -> tuple[Any, str | None] | None:
        """Loads the job bundle and returns ``(feature_engineer, target_column)``, if present."""
        if not artifact_store.exists(job_id):
            return None
        bundle = artifact_store.load(job_id)
        if not (isinstance(bundle, dict) and "feature_engineer" in bundle):
            return None
        return bundle.get("feature_engineer"), bundle.get("target_column")

    @staticmethod
    def _load_target_label_encoder(artifact_store: Any, job_id: str) -> Any:
        """Loads the job bundle and extracts its target label encoder, if any."""
        loaded = EvaluationService._load_feature_engineer_bundle(artifact_store, job_id)
        if loaded is None:
            return None
        feature_engineer, target_col_name = loaded
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
    def _decode_reference_crosstab(crosstab: dict[str, Any], label_encoder: Any) -> dict[str, Any]:
        """Best-effort decode of a cluster's ``{reference_value: count}`` breakdown.

        Reference-column values that were label-encoded upstream (e.g. species
        name -> 0/1/2) show up here as numeric-looking string keys — decode
        them back to their original text via ``label_encoder`` so the UI shows
        "setosa" instead of "0".
        """
        decoded_keys = decode_int_like(list(crosstab.keys()), label_encoder)
        return dict(zip((str(k) for k in decoded_keys), crosstab.values(), strict=True))

    @staticmethod
    def _decode_reference_column(data: Any, artifact_store: Any, job_id: str) -> None:
        """Best-effort decode of clustering's reference-column crosstab labels in-place.

        Only meaningful when the user set a ``reference_column`` and it was
        label-encoded somewhere upstream in the pipeline — silently skips
        otherwise (e.g. a text reference column that was never encoded is
        already human-readable and needs no decoding).
        """
        try:
            if not (isinstance(data, dict) and data.get("problem_type") == "clustering"):
                return
            splits = data.get("splits")
            if not isinstance(splits, dict):
                return

            reference_column: str | None = None
            for split_data in splits.values():
                if isinstance(split_data, dict):
                    clustering = split_data.get("clustering")
                    if isinstance(clustering, dict) and clustering.get("reference_column"):
                        reference_column = clustering["reference_column"]
                        break
            if not reference_column:
                return

            loaded = EvaluationService._load_feature_engineer_bundle(artifact_store, job_id)
            if loaded is None:
                return
            feature_engineer, _target_col_name = loaded
            label_encoder = extract_column_label_encoder(feature_engineer, reference_column)
            if label_encoder is None:
                return

            for split_data in splits.values():
                if not isinstance(split_data, dict):
                    continue
                clustering = split_data.get("clustering")
                if not isinstance(clustering, dict):
                    continue
                crosstab = clustering.get("reference_crosstab")
                if isinstance(crosstab, dict):
                    clustering["reference_crosstab"] = {
                        cluster_id: EvaluationService._decode_reference_crosstab(
                            counts, label_encoder
                        )
                        for cluster_id, counts in crosstab.items()
                        if isinstance(counts, dict)
                    }
        except Exception as e:
            logger.debug(f"Reference column decode skipped/failed: {e}")

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

            # Optional: decode a clustering reference column's crosstab labels
            EvaluationService._decode_reference_column(data, artifact_store, job_id)

            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load evaluation data: {str(e)}") from e
