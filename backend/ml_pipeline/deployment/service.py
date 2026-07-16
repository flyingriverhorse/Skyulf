import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import sklearn
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database.models import Deployment
from backend.ml_pipeline._services.job_service import JobService
from backend.ml_pipeline._services.prediction_utils import extract_target_label_encoder
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.artifacts.s3 import S3ArtifactStore

logger = logging.getLogger(__name__)


def _maybe_decode_predictions(
    predictions: Any,
    feature_engineer: Any,
    target_column: str | None = None,
) -> Any:
    """Decode numeric class predictions to original labels if possible.

    Looks for the target LabelEncoder, first under encoders['__target__'], then
    under encoders[target_column] (for pipelines where encoding happened before
    the Feature/Target Split).
    """

    target_encoder = extract_target_label_encoder(feature_engineer, target_column=target_column)
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
    def _validate_job_for_deployment(db_job: Any, job_id: str) -> None:
        """Raises ValueError if the job doesn't exist or hasn't completed successfully."""
        if not db_job:
            raise ValueError(f"Job {job_id} not found")

        if db_job.status not in ["completed", "succeeded"]:
            raise ValueError(f"Job {job_id} is not completed successfully")

    @staticmethod
    def _resolve_final_deployment_uri(
        artifact_uri: str | None, job_id: str, pipeline_id: Any
    ) -> str | None:
        """Resolves the deployment's stored artifact_uri, pointing at the specific bundled artifact file.

        If artifact_uri is a directory (doesn't end with .joblib or similar), we need to point to the
        specific file. The PipelineEngine saves the FULL bundled artifact (model + transformers) using
        the job_id as the key.
        """
        if not artifact_uri:
            return artifact_uri

        if artifact_uri.startswith("s3://"):
            # Handle S3 URI
            if not artifact_uri.endswith(".joblib") and not artifact_uri.endswith(".pkl"):
                # Assume it's a prefix, append job_id.joblib
                return f"{artifact_uri.rstrip('/')}/{job_id}.joblib"
            return artifact_uri
        # Check if it's a directory on disk
        elif Path(artifact_uri).is_dir():
            # The artifact is likely named {job_id}.joblib inside it.
            # We construct the full path to the file so predict() can parse it correctly.
            return str(Path(artifact_uri) / f"{job_id}.joblib")
        elif not artifact_uri.endswith(".joblib") and not artifact_uri.endswith(".pkl"):
            # Not a directory, and no extension. Assume it's a node_id or job_id.
            # Construct the abstract URI for exports/models
            return f"{pipeline_id}/{job_id}"
        else:
            # It's a file path (relative or absolute)
            return artifact_uri

    @staticmethod
    async def deploy_model(
        session: AsyncSession, job_id: str, user_id: int | None = None
    ) -> Deployment:
        # 1. Get Job Entity
        db_job = await JobService.get_job_by_id(session, job_id)
        DeploymentService._validate_job_for_deployment(db_job, job_id)
        if db_job is None:
            # Unreachable: _validate_job_for_deployment raises ValueError on a falsy db_job.
            raise ValueError(f"Job {job_id} not found")

        # 2. Get Artifact URI
        artifact_uri = db_job.artifact_uri
        pipeline_id = db_job.pipeline_id

        if not artifact_uri:
            # Fallback: use node_id if artifact_uri is missing (legacy jobs)
            artifact_uri = str(db_job.node_id)
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
        # In api.py: persistent_path = Path.cwd() / "exports" / "models" / config.pipeline_id
        # So we need pipeline_id to reconstruct the path.
        # Let's store "pipeline_id/node_id" as the URI if it's not already.
        final_uri = DeploymentService._resolve_final_deployment_uri(
            artifact_uri, job_id, pipeline_id
        )

        deployment = Deployment(
            job_id=job_id,
            model_type=db_job.model_type or "unknown",
            artifact_uri=final_uri,
            is_active=True,
            deployed_by=user_id,
        )
        session.add(deployment)
        await session.commit()
        await session.refresh(deployment)

        return deployment

    @staticmethod
    async def get_active_deployment(session: AsyncSession) -> Deployment | None:
        stmt = select(Deployment).where(Deployment.is_active).order_by(Deployment.created_at.desc())
        result = await session.execute(stmt)
        return result.scalars().first()

    @staticmethod
    async def list_deployments(
        session: AsyncSession, limit: int | None = None, skip: int = 0
    ) -> Sequence[Deployment]:
        """Lists deployment history."""
        effective_limit = limit if limit is not None else get_settings().DEFAULT_PAGE_SIZE
        stmt = (
            select(Deployment)
            .order_by(Deployment.created_at.desc())
            .limit(effective_limit)
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
    def _resolve_predict_store_and_key_s3(uri: str) -> tuple[str, str]:
        """Resolves an s3:// artifact URI into (store_uri, artifact_key)."""
        if uri.endswith(".joblib"):
            store_uri = uri.rsplit("/", 1)[0]
            artifact_key = uri.rsplit("/", 1)[1].replace(".joblib", "")
        else:
            parts = uri.replace("s3://", "").split("/")
            bucket = parts[0]
            artifact_key = "/".join(parts[1:])
            store_uri = f"s3://{bucket}"
        return store_uri, artifact_key

    @staticmethod
    def _resolve_pipeline_node_path(pipeline_id: str, node_id: str) -> tuple[str, str]:
        """Builds the default exports/models store path for a pipeline_id/node_id pair."""
        store_uri = str(Path.cwd() / "exports" / "models" / pipeline_id)
        return store_uri, node_id

    @staticmethod
    def _resolve_predict_store_and_key_local(uri: str) -> tuple[str, str]:
        """Resolves a local artifact path (absolute, relative, or bare "pipeline_id/node_id") into (store_uri, artifact_key)."""
        if Path(uri).is_absolute():
            return str(Path(uri).parent), Path(uri).name

        if "/" in uri or "\\" in uri:
            if not Path(uri).exists() and not Path(uri).parent.exists():
                parts = uri.replace("\\", "/").split("/")
                if len(parts) == 2:
                    return DeploymentService._resolve_pipeline_node_path(parts[0], parts[1])
                return str(Path(uri).parent), Path(uri).name
            return str(Path(uri).parent), Path(uri).name

        parts = uri.split("/")
        if len(parts) >= 2:
            return DeploymentService._resolve_pipeline_node_path(parts[0], parts[1])
        raise ValueError(f"Invalid artifact URI format: {uri}")

    @staticmethod
    def _resolve_predict_store_and_key(uri: str) -> tuple[str, str]:
        """Resolves an artifact URI into (store_uri, artifact_key) for the predict artifact loader.

        Handles S3 URIs (with or without a .joblib/.pkl suffix) and local paths
        (absolute, relative with a separator, or bare "pipeline_id/node_id" strings).
        """
        if uri.startswith("s3://"):
            return DeploymentService._resolve_predict_store_and_key_s3(uri)
        return DeploymentService._resolve_predict_store_and_key_local(uri)

    @staticmethod
    def _load_predict_artifact(deployment: Deployment) -> Any:
        """Loads and unwraps the deployed artifact used by predict(), wrapping load failures in ValueError."""
        try:
            from backend.ml_pipeline.artifacts.factory import ArtifactFactory

            store_uri, artifact_key = DeploymentService._resolve_predict_store_and_key(
                deployment.artifact_uri
            )
            store = ArtifactFactory.get_artifact_store(store_uri)
            artifact = store.load(artifact_key)
        except Exception as e:
            logger.error(f"Failed to load artifact: {e}")
            raise ValueError(f"Could not load model artifact: {deployment.artifact_uri}") from e

        # Handle tuple artifact (model, metadata/tuning_result) from TunerCalculator
        if isinstance(artifact, tuple) and len(artifact) >= 1:
            logger.info("Artifact is a tuple, using the first element as the model.")
            artifact = artifact[0]

        return artifact

    @staticmethod
    def _drop_target_and_dropped_columns(
        df: pd.DataFrame, target_col: str | None, dropped_cols: Any
    ) -> pd.DataFrame:
        """Drops the target column and any explicitly dropped columns from the inference DataFrame."""
        if target_col and target_col in df.columns:
            logger.info(f"Dropping target column '{target_col}' from inference data")
            df = df.drop(columns=[target_col])

        if dropped_cols:
            # Ensure dropped_cols is a list of strings
            if isinstance(dropped_cols, str):
                dropped_cols = [dropped_cols]

            existing_dropped = [c for c in dropped_cols if c in df.columns]
            if existing_dropped:
                logger.info(
                    f"Dropping explicitly dropped columns {existing_dropped} from inference data"
                )
                df = df.drop(columns=existing_dropped)

        return df

    @staticmethod
    def _unwrap_tuple_estimator(estimator: Any) -> Any:
        """Unwraps a tuple estimator (e.g. from TunerCalculator) to its first element."""
        if isinstance(estimator, tuple) and len(estimator) >= 1:
            logger.info("Estimator inside artifact is a tuple, using the first element.")
            return estimator[0]
        return estimator

    @staticmethod
    def _transform_bundled_features(feature_engineer: Any, df: pd.DataFrame) -> Any:
        """Transforms the inference DataFrame via the bundled feature engineer, wrapping failures in ValueError."""
        try:
            # Use config_context so sklearn returns DataFrames with feature names during inference only
            with sklearn.config_context(transform_output="pandas"):
                return feature_engineer.transform(df)
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise ValueError(f"Feature engineering failed: {str(e)}") from e

    @staticmethod
    def _predict_and_decode(
        estimator: Any, X_transformed: Any, feature_engineer: Any, target_col: str | None
    ) -> list:
        """Runs the estimator's predict and decodes labels, wrapping failures in ValueError."""
        try:
            predictions = estimator.predict(X_transformed)
            predictions = _maybe_decode_predictions(
                predictions, feature_engineer, target_column=target_col
            )
            if hasattr(predictions, "tolist"):
                return cast(list[Any], predictions.tolist())
            return list(predictions)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {str(e)}") from e

    @staticmethod
    def _predict_with_bundled_artifact(artifact: dict, df: pd.DataFrame) -> list:
        """Predicts using the new SDK bundled artifact format: {"feature_engineer": ..., "model": ...}."""
        feature_engineer = artifact["feature_engineer"]
        estimator = artifact["model"]

        # Clean Data
        target_col = artifact.get("target_column")
        dropped_cols = artifact.get("dropped_columns", [])
        df = DeploymentService._drop_target_and_dropped_columns(df, target_col, dropped_cols)

        estimator = DeploymentService._unwrap_tuple_estimator(estimator)

        X_transformed = DeploymentService._transform_bundled_features(feature_engineer, df)

        return DeploymentService._predict_and_decode(
            estimator, X_transformed, feature_engineer, target_col
        )

    @staticmethod
    def _predict_with_legacy_artifact(artifact: Any, df: pd.DataFrame) -> list:
        """Predicts using a legacy artifact that is directly a fitted predictor (no bundled feature engineer)."""
        # Log columns for debugging
        if isinstance(df, pd.DataFrame):
            logger.info(f"Predicting with columns: {df.columns.tolist()}")
            # Check if model has feature names and if they match
            if hasattr(artifact, "feature_names_in_"):
                model_cols = artifact.feature_names_in_.tolist()
                missing_in_df = set(model_cols) - set(df.columns)
                if missing_in_df:
                    logger.warning(f"Missing columns in input DataFrame: {missing_in_df}")
                    for c in missing_in_df:
                        df[c] = 0
                # Reorder columns to match model
                df = df[model_cols]

        predictions = artifact.predict(df)
        if hasattr(predictions, "tolist"):
            return cast(list[Any], predictions.tolist())
        return list(predictions)

    @staticmethod
    async def predict(session: AsyncSession, data: list[dict]) -> list:
        # 1. Get active deployment
        deployment = await DeploymentService.get_active_deployment(session)
        if not deployment:
            raise ValueError("No active model deployed")

        # 2. Load Artifact
        artifact = DeploymentService._load_predict_artifact(deployment)

        # 3. Prepare Data
        df = pd.DataFrame(data)

        # 4. Predict
        # Check for new SDK format: {"feature_engineer": ..., "model": ...}
        if isinstance(artifact, dict) and "feature_engineer" in artifact and "model" in artifact:
            return DeploymentService._predict_with_bundled_artifact(artifact, df)
        # Legacy support or direct model loading (if artifact is just the model)
        elif hasattr(artifact, "predict"):
            return DeploymentService._predict_with_legacy_artifact(artifact, df)
        else:
            raise ValueError(
                "Loaded artifact is not a valid predictor or recognized pipeline format"
            )

    @staticmethod
    def _load_artifact_from_s3_for_details(artifact_uri: str) -> Any:
        """Loads an artifact from S3 for schema inspection, building storage options from settings."""
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
        return store.load(key)

    @staticmethod
    def _resolve_local_base_and_key_for_details(artifact_uri: str) -> tuple[str, str]:
        """Resolves a local artifact URI into (base_path, node_id) for schema-inspection loading."""
        if Path(artifact_uri).is_absolute():
            return str(Path(artifact_uri).parent), Path(artifact_uri).name
        elif "/" in artifact_uri or "\\" in artifact_uri:
            if not Path(artifact_uri).exists() and not Path(artifact_uri).parent.exists():
                parts = artifact_uri.replace("\\", "/").split("/")
                if len(parts) == 2:
                    pipeline_id = parts[0]
                    node_id = parts[1]
                    base_path = str(Path.cwd() / "exports" / "models" / pipeline_id)
                    return base_path, node_id
            return str(Path(artifact_uri).parent), Path(artifact_uri).name
        else:
            # Fallback
            return str(Path.cwd()), artifact_uri

    @staticmethod
    def _load_artifact_for_details(artifact_uri: str) -> Any:
        """Loads the deployed artifact for schema inspection, mirroring predict()'s URI resolution.

        Unlike predict(), this instantiates S3ArtifactStore/LocalArtifactStore directly and
        returns None (rather than raising) when the artifact does not exist locally.
        """
        if artifact_uri.startswith("s3://"):
            return DeploymentService._load_artifact_from_s3_for_details(artifact_uri)

        base_path, node_id = DeploymentService._resolve_local_base_and_key_for_details(artifact_uri)
        store = LocalArtifactStore(base_path)
        return store.load(node_id) if store.exists(node_id) else None

    @staticmethod
    def _extract_features_from_engineer(fe: Any) -> Any:
        """Best-effort extraction of feature names from a feature engineer or its first pipeline step."""
        if hasattr(fe, "feature_names_in_"):
            return fe.feature_names_in_

        if hasattr(fe, "steps") and fe.steps:
            # Try first step
            first_step = fe.steps[0]
            # If it's a tuple (name, transformer)
            if isinstance(first_step, tuple) and len(first_step) > 1:
                transformer = first_step[1]
                if hasattr(transformer, "feature_names_in_"):
                    return transformer.feature_names_in_

        return []

    @staticmethod
    def _extract_features_from_bundled_artifact(artifact: dict) -> Any:
        """Best-effort extraction of input feature names from a bundled artifact's feature engineer or model."""
        fe = artifact["feature_engineer"]
        input_features = DeploymentService._extract_features_from_engineer(fe)

        # If still empty, try model
        if len(input_features) == 0 and "model" in artifact:
            model = artifact["model"]
            if isinstance(model, tuple):
                model = model[0]
            if hasattr(model, "feature_names_in_"):
                input_features = model.feature_names_in_

        return input_features

    @staticmethod
    def _extract_input_features(artifact: Any) -> Any:
        """Best-effort extraction of the input feature name list from an artifact's feature engineer or model."""
        input_features = []

        # Check dict format
        if isinstance(artifact, dict) and "feature_engineer" in artifact:
            input_features = DeploymentService._extract_features_from_bundled_artifact(artifact)
        # Check direct model
        elif hasattr(artifact, "feature_names_in_"):
            input_features = artifact.feature_names_in_

        if hasattr(input_features, "tolist"):
            input_features = cast(Any, input_features).tolist()

        return input_features

    @staticmethod
    def _extract_target_column_from_graph(graph: dict) -> str | None:
        """Finds the first `target_column` param among a job graph's nodes, or None if not found."""
        nodes = graph.get("nodes", [])
        for node in nodes:
            # Handle both dict and object (though graph is usually dict from DB)
            if isinstance(node, dict):
                params = node.get("params", {})
                if "target_column" in params:
                    return cast(str, params["target_column"])
            elif hasattr(node, "params"):
                params = getattr(node, "params", {})
                if "target_column" in params:
                    return cast(str, params["target_column"])
        return None

    @staticmethod
    def _build_input_schema_from_artifact(artifact_uri: str) -> list[dict[str, str]] | None:
        """Loads the artifact and extracts its input schema, unwrapping tuple artifacts first."""
        artifact = DeploymentService._load_artifact_for_details(artifact_uri)
        if not artifact:
            return None

        if isinstance(artifact, tuple) and len(artifact) >= 1:
            artifact = artifact[0]

        input_features = DeploymentService._extract_input_features(artifact)
        if not input_features:
            return None

        return [{"name": str(f), "type": "unknown"} for f in input_features]

    @staticmethod
    async def _lookup_target_column(session: AsyncSession, job_id: Any) -> str | None:
        """Looks up the job graph for a deployment's job and extracts its target column, if any."""
        from backend.ml_pipeline._execution.jobs import JobManager

        job = await JobManager.get_job(session, str(job_id))
        if job and job.graph:
            return DeploymentService._extract_target_column_from_graph(job.graph)
        return None

    @staticmethod
    async def get_deployment_details(
        session: AsyncSession, deployment: Deployment
    ) -> dict[str, Any]:
        """
        Returns deployment info enriched with input/output schema from the artifact.
        """
        info = deployment.to_dict()
        info["input_schema"] = None
        info["output_schema"] = None

        try:
            artifact_uri = str(deployment.artifact_uri)
            input_schema = DeploymentService._build_input_schema_from_artifact(artifact_uri)
            if input_schema:
                info["input_schema"] = input_schema

            target_column = await DeploymentService._lookup_target_column(
                session, deployment.job_id
            )
            if target_column is not None:
                info["target_column"] = target_column

        except Exception as e:
            logger.warning(f"Failed to extract schema for deployment {deployment.id}: {e}")

        return cast(dict[str, Any], info)
