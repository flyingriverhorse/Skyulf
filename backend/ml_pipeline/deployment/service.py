import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import pandas as pd
import sklearn
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import Deployment, HyperparameterTuningJob, TrainingJob
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.artifacts.s3 import S3ArtifactStore
from backend.ml_pipeline.execution.jobs import JobManager
from backend.config import get_settings

# Ensure sklearn outputs pandas DataFrames where possible to preserve feature names
sklearn.set_config(transform_output="pandas")

logger = logging.getLogger(__name__)


def _extract_target_label_encoder(feature_engineer: Any):
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


def _maybe_decode_predictions(predictions: Any, feature_engineer: Any) -> Any:
    """Decode numeric class predictions to original labels if possible.

    This uses the saved target LabelEncoder fitted during training (stored under
    the LabelEncoder step artifact as encoders['__target__']). If no encoder is
    present, or decoding fails, it returns predictions unchanged.
    """

    target_encoder = _extract_target_label_encoder(feature_engineer)
    if target_encoder is None:
        return predictions

    try:
        import numpy as np

        preds = np.asarray(predictions)
        # Best-effort: many sklearn classifiers output ints; decode expects int-like.
        if preds.dtype.kind in {"i", "u", "b"}:
            return target_encoder.inverse_transform(preds.astype(int))

        # If dtype isn't integer but values might still be numeric strings/floats,
        # try converting.
        return target_encoder.inverse_transform(preds.astype(int))
    except Exception as e:
        logger.debug(f"Could not decode predictions via LabelEncoder: {e}")
        return predictions


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
            if artifact_uri.startswith("s3://"):
                # Handle S3 URI
                if not artifact_uri.endswith(".joblib") and not artifact_uri.endswith(".pkl"):
                    # Assume it's a prefix, append job_id.joblib
                    final_uri = f"{artifact_uri.rstrip('/')}/{job_id}.joblib"
            # Check if it's a directory on disk
            elif os.path.isdir(artifact_uri):
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
    async def predict(session: AsyncSession, data: list[dict]) -> list:  # noqa: C901
        # 1. Get active deployment
        deployment = await DeploymentService.get_active_deployment(session)
        if not deployment:
            raise ValueError("No active model deployed")

        # 2. Load Artifact
        try:
            store: Union[S3ArtifactStore, LocalArtifactStore]
            if deployment.artifact_uri.startswith("s3://"):
                # Parse bucket and key: s3://bucket/key
                parts = deployment.artifact_uri.replace("s3://", "").split("/")
                bucket_name = parts[0]
                key = "/".join(parts[1:])
                
                settings = get_settings()
                storage_options = {
                    "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
                    "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
                    "endpoint_url": settings.AWS_ENDPOINT_URL,
                    "region_name": settings.AWS_DEFAULT_REGION,
                }
                # Filter None values
                storage_options = {k: v for k, v in storage_options.items() if v is not None}
                
                store = S3ArtifactStore(bucket_name=bucket_name, storage_options=storage_options)
                artifact = store.load(key)

            else:
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

            # Clean Data
            target_col = artifact.get("target_column")
            dropped_cols = artifact.get("dropped_columns", [])
            
            if target_col and target_col in df.columns:
                logger.info(f"Dropping target column '{target_col}' from inference data")
                df = df.drop(columns=[target_col])
                
            if dropped_cols:
                # Ensure dropped_cols is a list of strings
                if isinstance(dropped_cols, str):
                    dropped_cols = [dropped_cols]
                
                existing_dropped = [c for c in dropped_cols if c in df.columns]
                if existing_dropped:
                    logger.info(f"Dropping explicitly dropped columns {existing_dropped} from inference data")
                    df = df.drop(columns=existing_dropped)

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
                predictions = _maybe_decode_predictions(predictions, feature_engineer)
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

    @staticmethod
    async def get_deployment_details(session: AsyncSession, deployment: Deployment) -> Dict[str, Any]:
        """
        Returns deployment info enriched with input/output schema from the artifact.
        """
        info = deployment.to_dict()
        info["input_schema"] = None
        info["output_schema"] = None

        try:
            # Load Artifact (Reuse logic from predict - simplified)
            artifact_uri = str(deployment.artifact_uri)
            store: Union[S3ArtifactStore, LocalArtifactStore]

            if artifact_uri.startswith("s3://"):
                # Parse bucket and key: s3://bucket/key
                parts = artifact_uri.replace("s3://", "").split("/")
                bucket_name = parts[0]
                key = "/".join(parts[1:])
                
                settings = get_settings()
                storage_options = {
                    "key": settings.AWS_ACCESS_KEY_ID,
                    "secret": settings.AWS_SECRET_ACCESS_KEY,
                    "endpoint_url": settings.AWS_ENDPOINT_URL,
                    "region_name": settings.AWS_DEFAULT_REGION,
                }
                # Filter None values
                storage_options = {k: v for k, v in storage_options.items() if v is not None}
                
                store = S3ArtifactStore(bucket_name=bucket_name, storage_options=storage_options)
                artifact = store.load(key)

            elif os.path.isabs(artifact_uri):
                base_path = os.path.dirname(artifact_uri)
                node_id = os.path.basename(artifact_uri)
                store = LocalArtifactStore(base_path)
                if store.exists(node_id):
                    artifact = store.load(node_id)
                else:
                    artifact = None
            elif "/" in artifact_uri or "\\" in artifact_uri:
                if not os.path.exists(artifact_uri) and not os.path.exists(
                    os.path.dirname(artifact_uri)
                ):
                    parts = artifact_uri.replace("\\", "/").split("/")
                    if len(parts) == 2:
                        pipeline_id = parts[0]
                        node_id = parts[1]
                        base_path = os.path.join(
                            os.getcwd(), "exports", "models", pipeline_id
                        )
                    else:
                        base_path = os.path.dirname(artifact_uri)
                        node_id = os.path.basename(artifact_uri)
                else:
                    base_path = os.path.dirname(artifact_uri)
                    node_id = os.path.basename(artifact_uri)
                
                store = LocalArtifactStore(base_path)
                if store.exists(node_id):
                    artifact = store.load(node_id)
                else:
                    artifact = None
            else:
                # Fallback
                base_path = os.getcwd()
                node_id = artifact_uri
                store = LocalArtifactStore(base_path)
                if store.exists(node_id):
                    artifact = store.load(node_id)
                else:
                    artifact = None

            if artifact:
                # Handle tuple
                if isinstance(artifact, tuple) and len(artifact) >= 1:
                    artifact = artifact[0]

                # Extract Schema
                input_features = []
                
                # Check dict format
                if isinstance(artifact, dict) and "feature_engineer" in artifact:
                    fe = artifact["feature_engineer"]
                    # Try to get features from FeatureEngineer
                    if hasattr(fe, "feature_names_in_"):
                        input_features = fe.feature_names_in_
                    elif hasattr(fe, "steps") and fe.steps:
                        # Try first step
                        first_step = fe.steps[0]
                        # If it's a tuple (name, transformer)
                        if isinstance(first_step, tuple) and len(first_step) > 1:
                            transformer = first_step[1]
                            if hasattr(transformer, "feature_names_in_"):
                                input_features = transformer.feature_names_in_
                    
                    # If still empty, try model
                    if not input_features and "model" in artifact:
                        model = artifact["model"]
                        if isinstance(model, tuple): model = model[0]
                        if hasattr(model, "feature_names_in_"):
                            input_features = model.feature_names_in_

                # Check direct model
                elif hasattr(artifact, "feature_names_in_"):
                    input_features = artifact.feature_names_in_
                
                if hasattr(input_features, "tolist"):
                    input_features = input_features.tolist()
                
                if input_features:
                    info["input_schema"] = [{"name": str(f), "type": "unknown"} for f in input_features]

            # Extract Target Column from Job Graph
            from backend.ml_pipeline.execution.jobs import JobManager
            job = await JobManager.get_job(session, str(deployment.job_id))
            if job and job.graph:
                nodes = job.graph.get("nodes", [])
                for node in nodes:
                    # Handle both dict and object (though graph is usually dict from DB)
                    if isinstance(node, dict):
                        params = node.get("params", {})
                        if "target_column" in params:
                            info["target_column"] = params["target_column"]
                            break
                    elif hasattr(node, "params"):
                        params = getattr(node, "params", {})
                        if "target_column" in params:
                            info["target_column"] = params["target_column"]
                            break

        except Exception as e:
            logger.warning(f"Failed to extract schema for deployment {deployment.id}: {e}")

        return cast(Dict[str, Any], info)
