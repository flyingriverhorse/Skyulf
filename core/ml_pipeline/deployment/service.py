from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from core.database.models import Deployment, TrainingJob, HyperparameterTuningJob
from core.ml_pipeline.artifacts.local import LocalArtifactStore
from core.ml_pipeline.execution.jobs import JobManager
import pandas as pd
import logging
import os
import sklearn

# Ensure sklearn outputs pandas DataFrames where possible to preserve feature names
sklearn.set_config(transform_output="pandas")

from core.ml_pipeline.preprocessing.encoding import (
    OneHotEncoderApplier, LabelEncoderApplier, OrdinalEncoderApplier, 
    TargetEncoderApplier, HashEncoderApplier, DummyEncoderApplier
)
from core.ml_pipeline.preprocessing.scaling import (
    StandardScalerApplier, MinMaxScalerApplier, RobustScalerApplier, MaxAbsScalerApplier
)
from core.ml_pipeline.preprocessing.imputation import (
    SimpleImputerApplier, IterativeImputerApplier, KNNImputerApplier
)
from core.ml_pipeline.preprocessing.transformations import (
    PowerTransformerApplier, SimpleTransformationApplier, GeneralTransformationApplier
)
from core.ml_pipeline.preprocessing.bucketing import GeneralBinningApplier
from core.ml_pipeline.preprocessing.outliers import IQRApplier, ZScoreApplier
from core.ml_pipeline.preprocessing.cleaning import TextCleaningApplier
from core.ml_pipeline.preprocessing.feature_generation import PolynomialFeaturesApplier
from core.ml_pipeline.preprocessing.feature_selection import VarianceThresholdApplier
from core.ml_pipeline.preprocessing.casting import CastingApplier
from core.ml_pipeline.preprocessing.drop_and_missing import (
    DeduplicateApplier, DropMissingColumnsApplier, DropMissingRowsApplier, MissingIndicatorApplier
)

logger = logging.getLogger(__name__)

APPLIER_MAP = {
    "onehot": OneHotEncoderApplier,
    "label": LabelEncoderApplier,
    "ordinal": OrdinalEncoderApplier,
    "target": TargetEncoderApplier,
    "hash": HashEncoderApplier,
    "dummy": DummyEncoderApplier,
    "standard_scaler": StandardScalerApplier,
    "minmax_scaler": MinMaxScalerApplier,
    "robust_scaler": RobustScalerApplier,
    "maxabs_scaler": MaxAbsScalerApplier,
    "simple_imputer": SimpleImputerApplier,
    "iterative_imputer": IterativeImputerApplier,
    "knn_imputer": KNNImputerApplier,
    "power_transformer": PowerTransformerApplier,
    "simple_transformation": SimpleTransformationApplier,
    "general_transformation": GeneralTransformationApplier,
    "general_binning": GeneralBinningApplier,
    "iqr": IQRApplier,
    "zscore": ZScoreApplier,
    "text_cleaning": TextCleaningApplier,
    "polynomial_features": PolynomialFeaturesApplier,
    "variance_threshold": VarianceThresholdApplier,
    "casting": CastingApplier,
    "deduplicate": DeduplicateApplier,
    "drop_missing_columns": DropMissingColumnsApplier,
    "drop_missing_rows": DropMissingRowsApplier,
    "missing_indicator": MissingIndicatorApplier,
}

class DeploymentService:
    
    @staticmethod
    async def deploy_model(session: AsyncSession, job_id: str, user_id: int = None) -> Deployment:
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
                 artifact_uri = db_job.artifact_uri
        else:
             stmt = select(HyperparameterTuningJob).where(HyperparameterTuningJob.id == job_id)
             result = await session.execute(stmt)
             db_job = result.scalar_one_or_none()
             if db_job:
                 artifact_uri = db_job.artifact_uri
                 
        if not artifact_uri:
            # Fallback: use node_id if artifact_uri is missing (legacy jobs)
            artifact_uri = job.node_id
            logger.warning(f"No artifact URI found for job {job_id}, falling back to node_id: {artifact_uri}")

        # 3. Deactivate current active deployment
        await session.execute(
            update(Deployment).where(Deployment.is_active == True).values(is_active=False)
        )
        
        # 4. Create Deployment
        # We need to store the pipeline_id to know where to look for the artifact
        # For now, we'll append it to the artifact_uri if it's just a node_id
        # Or better, we assume the store path logic is consistent.
        # In api.py: persistent_path = os.path.join(os.getcwd(), "exports", "models", config.pipeline_id)
        # So we need pipeline_id to reconstruct the path.
        # Let's store "pipeline_id/node_id" as the URI if it's not already.
        
        final_uri = artifact_uri
        if "/" not in artifact_uri and "\\" not in artifact_uri:
            final_uri = f"{pipeline_id}/{artifact_uri}"

        deployment = Deployment(
            job_id=job_id,
            model_type=job.model_type or "unknown",
            artifact_uri=final_uri,
            is_active=True,
            deployed_by=user_id
        )
        session.add(deployment)
        await session.commit()
        await session.refresh(deployment)
        
        return deployment

    @staticmethod
    async def get_active_deployment(session: AsyncSession) -> Deployment:
        stmt = select(Deployment).where(Deployment.is_active == True).order_by(Deployment.created_at.desc())
        result = await session.execute(stmt)
        return result.scalars().first()

    @staticmethod
    async def deactivate_current_deployment(session: AsyncSession):
        """Deactivates the currently active deployment."""
        await session.execute(
            update(Deployment).where(Deployment.is_active == True).values(is_active=False)
        )
        await session.commit()

    @staticmethod
    async def predict(session: AsyncSession, data: list[dict]) -> list:
        # 1. Get active deployment
        deployment = await DeploymentService.get_active_deployment(session)
        if not deployment:
            raise ValueError("No active model deployed")
            
        # 2. Load Artifact
        # Reconstruct store path
        # URI format: "pipeline_id/node_id"
        try:
            parts = deployment.artifact_uri.split("/")
            if len(parts) >= 2:
                pipeline_id = parts[0]
                node_id = parts[1]
                base_path = os.path.join(os.getcwd(), "exports", "models", pipeline_id)
            else:
                # Fallback for absolute paths or other formats
                base_path = os.path.dirname(deployment.artifact_uri)
                node_id = os.path.basename(deployment.artifact_uri)

            store = LocalArtifactStore(base_path)
            model = store.load(node_id)
            
        except Exception as e:
            logger.error(f"Failed to load artifact: {e}")
            raise ValueError(f"Could not load model artifact: {deployment.artifact_uri}")
        
        # 3. Prepare Data
        df = pd.DataFrame(data)
        
        # 4. Predict
        final_model = model
        
        # Check if artifact is a full pipeline (dict with transformers)
        if isinstance(model, dict) and "model" in model:
            final_model = model["model"]
            transformers = model.get("transformers", [])
            plan = model.get("transformer_plan", [])
            
            # Build lookup for transformer objects
            # Key: (node_id, transformer_name, column_name)
            t_objs = {}
            for t in transformers:
                # Handle None column_name by converting to 'None' string or keeping None
                # The plan uses what was saved. _collect_transformers converts None to None.
                # But let's be safe and match exactly what's in the dict.
                t_node = t.get("node_id")
                t_name = t.get("transformer_name")
                t_col = t.get("column_name")
                t_objs[(t_node, t_name, t_col)] = t.get("transformer")

            # Apply transformations in order
            for step in plan:
                node_id = step.get("node_id")
                for t_spec in step.get("transformers", []):
                    t_name = t_spec.get("transformer_name")
                    t_col = t_spec.get("column_name")
                    metadata = t_spec.get("metadata") or {}
                    
                    # Get object
                    obj = t_objs.get((node_id, t_name, t_col))
                    
                    t_type = metadata.get("type")
                    ApplierCls = APPLIER_MAP.get(t_type)
                    
                    if ApplierCls:
                        try:
                            applier = ApplierCls()
                            # Prepare params
                            params = metadata.copy()
                            # Inject object into common keys
                            if obj is not None:
                                params["encoder_object"] = obj
                                params["scaler_object"] = obj
                                params["imputer_object"] = obj
                                params["transformer_object"] = obj
                            
                            # Apply
                            res = applier.apply(df, params)
                            
                            # Unpack result
                            if isinstance(res, tuple):
                                df = res[0]
                            else:
                                df = res
                        except Exception as e:
                            logger.warning(f"Failed to apply transformer {t_type} for {t_col}: {e}")
                            # Continue? Or fail? 
                            # If we fail, the whole prediction fails. 
                            # If we continue, the model might fail later due to missing columns.
                            # Better to log and try to continue.

        # Ensure sklearn outputs pandas DataFrames if final_model is a Pipeline
        if hasattr(final_model, "set_output"):
            try:
                final_model.set_output(transform="pandas")
            except Exception:
                pass

        if hasattr(final_model, "predict"):
            # Log columns for debugging
            if isinstance(df, pd.DataFrame):
                logger.info(f"Predicting with columns: {df.columns.tolist()}")
                # Check if model has feature names and if they match
                if hasattr(final_model, "feature_names_in_"):
                    model_cols = final_model.feature_names_in_.tolist()
                    missing_in_df = set(model_cols) - set(df.columns)
                    if missing_in_df:
                        logger.warning(f"Missing columns in input DataFrame: {missing_in_df}")
                        # Add missing columns as NaN?
                        # If we add them, the model might handle them if it can handle NaNs (e.g. HistGradientBoosting)
                        # But RandomForest usually can't.
                        # However, if we don't add them, we get shape mismatch.
                        # Let's try to add them as 0 or NaN?
                        # Adding as 0 is safer for tree models if it's one-hot encoded features.
                        for c in missing_in_df:
                            df[c] = 0 # Assuming 0 is a safe default for missing features (e.g. one-hot)
                            
                    # Reorder columns to match model
                    df = df[model_cols]
            else:
                logger.warning(f"Predicting with non-DataFrame input of type {type(df)}")

            predictions = final_model.predict(df)
            if hasattr(predictions, "tolist"):
                return predictions.tolist()
            return list(predictions)
        else:
            raise ValueError("Loaded artifact is not a valid predictor")
