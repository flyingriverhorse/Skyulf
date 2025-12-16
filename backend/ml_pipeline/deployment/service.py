import logging
import os
from typing import Any, List, Optional, Sequence, cast

import pandas as pd
import sklearn
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import Deployment, HyperparameterTuningJob, TrainingJob
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.execution.jobs import JobManager

# Ensure sklearn outputs pandas DataFrames where possible to preserve feature names
sklearn.set_config(transform_output="pandas")

logger = logging.getLogger(__name__)


class DeploymentService:

    @staticmethod
    async def deploy_model(
        session: AsyncSession, job_id: str, user_id: Optional[int] = None
    ) -> Deployment:
        # 1. Get Job Info
        job = await JobManager.get_job(session, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status.value not in ["completed", "succeeded"]:
            raise ValueError(f"Job {job_id} is not completed successfully")

        # 2. Get Artifact URI
        artifact_uri = None
        pipeline_id = job.pipeline_id

        if job.job_type == "training":
            stmt = select(TrainingJob).where(TrainingJob.id == job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()
            if db_job:
                artifact_uri = cast(str, db_job.artifact_uri)
        else:
            stmt = select(HyperparameterTuningJob).where(
                HyperparameterTuningJob.id == job_id
            )
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()
            if db_job:
                artifact_uri = cast(str, db_job.artifact_uri)

        if not artifact_uri:
            # Fallback: use node_id if artifact_uri is missing (legacy jobs)
            artifact_uri = cast(str, job.node_id)
            logger.warning(
                f"No artifact URI found for job {job_id}, falling back to node_id: {artifact_uri}"
            )

        # 3. Deactivate current active deployment
        await session.execute(
            update(Deployment).where(Deployment.is_active).values(is_active=False)
        )

        # 4. Create Deployment
        # We need to store the pipeline_id to know where to look for the artifact
        # For now, we'll append it to the artifact_uri if it's just a node_id
        # Or better, we assume the store path logic is consistent.
        # In api.py: persistent_path = os.path.join(os.getcwd(), "exports", "models", config.pipeline_id)
        # So we need pipeline_id to reconstruct the path.
        # Let's store "pipeline_id/node_id" as the URI if it's not already.

        final_uri = artifact_uri

        # If artifact_uri is a directory (doesn't end with .joblib or similar), we need to point to the specific file.
        # The PipelineEngine saves the FULL bundled artifact (model + transformers) using the job_id as the key.
        if artifact_uri:
            # Check if it's a directory on disk
            if os.path.isdir(artifact_uri):
                # The artifact is likely named {job_id}.joblib inside it.
                # We construct the full path to the file so predict() can parse it correctly.
                final_uri = os.path.join(artifact_uri, f"{job_id}.joblib")
            elif not artifact_uri.endswith(".joblib") and not artifact_uri.endswith(
                ".pkl"
            ):
                # Not a directory, and no extension. Assume it's a node_id or job_id.
                # Construct the abstract URI for exports/models
                final_uri = f"{pipeline_id}/{job_id}"
            else:
                # It's a file path (relative or absolute)
                final_uri = artifact_uri

        deployment = Deployment(
            job_id=job_id,
            model_type=job.model_type or "unknown",
            artifact_uri=final_uri,
            is_active=True,
            deployed_by=user_id,
        )
        session.add(deployment)
        await session.commit()
        await session.refresh(deployment)

        return deployment

    @staticmethod
    async def get_active_deployment(session: AsyncSession) -> Optional[Deployment]:
        stmt = (
            select(Deployment)
            .where(Deployment.is_active)
            .order_by(Deployment.created_at.desc())
        )
        result = await session.execute(stmt)
        return result.scalars().first()

    @staticmethod
    async def list_deployments(
        session: AsyncSession, limit: int = 50, skip: int = 0
    ) -> Sequence[Deployment]:
        """Lists deployment history."""
        stmt = (
            select(Deployment)
            .order_by(Deployment.created_at.desc())
            .limit(limit)
            .offset(skip)
        )
        result = await session.execute(stmt)
        return result.scalars().all()

    @staticmethod
    async def deactivate_current_deployment(session: AsyncSession):
        """Deactivates the currently active deployment."""
        await session.execute(
            update(Deployment).where(Deployment.is_active).values(is_active=False)
        )
        await session.commit()

    @staticmethod
    async def predict(session: AsyncSession, data: list[dict]) -> list:
        # 1. Get active deployment
        deployment = await DeploymentService.get_active_deployment(session)
        if not deployment:
            raise ValueError("No active model deployed")

        # 2. Load Artifact
        try:
            # Handle full path (absolute or relative with separators)
            if os.path.isabs(deployment.artifact_uri):
                base_path = os.path.dirname(deployment.artifact_uri)
                node_id = os.path.basename(deployment.artifact_uri)
            elif "/" in deployment.artifact_uri or "\\" in deployment.artifact_uri:
                # Check if it's a "pipeline_id/node_id" pattern that maps to exports/models
                # If the path doesn't exist locally, assume it's the internal format
                if not os.path.exists(deployment.artifact_uri) and not os.path.exists(
                    os.path.dirname(deployment.artifact_uri)
                ):
                    parts = deployment.artifact_uri.replace("\\", "/").split("/")
                    if len(parts) == 2:
                        pipeline_id = parts[0]
                        node_id = parts[1]
                        base_path = os.path.join(
                            os.getcwd(), "exports", "models", pipeline_id
                        )
                    else:
                        base_path = os.path.dirname(deployment.artifact_uri)
                        node_id = os.path.basename(deployment.artifact_uri)
                else:
                    base_path = os.path.dirname(deployment.artifact_uri)
                    node_id = os.path.basename(deployment.artifact_uri)
            else:
                # Legacy format: "pipeline_id/node_id" (handled by split above if separators exist)
                # OR just "node_id" (fallback)
                parts = deployment.artifact_uri.split("/")
                if len(parts) >= 2:
                    pipeline_id = parts[0]
                    node_id = parts[1]
                    base_path = os.path.join(
                        os.getcwd(), "exports", "models", pipeline_id
                    )
                else:
                    raise ValueError(
                        f"Invalid artifact URI format: {deployment.artifact_uri}"
                    )

            store = LocalArtifactStore(base_path)
            artifact = store.load(node_id)

        except Exception as e:
            logger.error(f"Failed to load artifact: {e}")
            raise ValueError(
                f"Could not load model artifact: {deployment.artifact_uri}"
            )

        # Handle tuple artifact (model, metadata/tuning_result) from TunerCalculator
        if isinstance(artifact, tuple) and len(artifact) >= 1:
            logger.info("Artifact is a tuple, using the first element as the model.")
            artifact = artifact[0]

        # 3. Prepare Data
        df = pd.DataFrame(data)

        # 4. Predict
        # Check for new SDK format: {"feature_engineer": ..., "model": ...}
        if (
            isinstance(artifact, dict)
            and "feature_engineer" in artifact
            and "model" in artifact
        ):
            feature_engineer = artifact["feature_engineer"]
            estimator = artifact["model"]

            # Handle tuple estimator inside dict (e.g. from TunerCalculator)
            if isinstance(estimator, tuple) and len(estimator) >= 1:
                logger.info("Estimator inside artifact is a tuple, using the first element.")
                estimator = estimator[0]

            # Transform
            try:
                X_transformed = feature_engineer.transform(df)
            except Exception as e:
                logger.error(f"Feature engineering failed: {e}")
                raise ValueError(f"Feature engineering failed: {str(e)}")

            # Predict
            try:
                predictions = estimator.predict(X_transformed)
                if hasattr(predictions, "tolist"):
                    return cast(List[Any], predictions.tolist())
                return list(predictions)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise ValueError(f"Prediction failed: {str(e)}")

        # Legacy support or direct model loading (if artifact is just the model)
        elif hasattr(artifact, "predict"):
            # Log columns for debugging
            if isinstance(df, pd.DataFrame):
                logger.info(f"Predicting with columns: {df.columns.tolist()}")
                # Check if model has feature names and if they match
                if hasattr(artifact, "feature_names_in_"):
                    model_cols = artifact.feature_names_in_.tolist()
                    missing_in_df = set(model_cols) - set(df.columns)
                    if missing_in_df:
                        logger.warning(
                            f"Missing columns in input DataFrame: {missing_in_df}"
                        )
                        for c in missing_in_df:
                            df[c] = 0
                    # Reorder columns to match model
                    df = df[model_cols]

            predictions = artifact.predict(df)
            if hasattr(predictions, "tolist"):
                return cast(List[Any], predictions.tolist())
            return list(predictions)
        else:
            raise ValueError(
                "Loaded artifact is not a valid predictor or recognized pipeline format"
            )
