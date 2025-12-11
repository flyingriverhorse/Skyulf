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
from core.ml_pipeline.preprocessing.cleaning import (
    TextCleaningApplier, ValueReplacementApplier, AliasReplacementApplier, 
    InvalidValueReplacementApplier
)
from core.ml_pipeline.preprocessing.feature_generation import (
    PolynomialFeaturesApplier, FeatureGenerationApplier
)
from core.ml_pipeline.preprocessing.feature_selection import (
    VarianceThresholdApplier, CorrelationThresholdApplier, 
    UnivariateSelectionApplier, ModelBasedSelectionApplier, FeatureSelectionApplier
)
from core.ml_pipeline.preprocessing.casting import CastingApplier
from core.ml_pipeline.preprocessing.drop_and_missing import (
    DeduplicateApplier, DropMissingColumnsApplier, DropMissingRowsApplier, MissingIndicatorApplier
)
from core.ml_pipeline.preprocessing.outliers import (
    IQRApplier, ZScoreApplier, WinsorizeApplier, ManualBoundsApplier
)

logger = logging.getLogger(__name__)

APPLIER_MAP = {
    # Encoding
    "onehot": OneHotEncoderApplier,
    "OneHotEncoder": OneHotEncoderApplier,
    "label": LabelEncoderApplier,
    "LabelEncoder": LabelEncoderApplier,
    "ordinal": OrdinalEncoderApplier,
    "OrdinalEncoder": OrdinalEncoderApplier,
    "target": TargetEncoderApplier,
    "TargetEncoder": TargetEncoderApplier,
    "hash": HashEncoderApplier,
    "HashEncoder": HashEncoderApplier,
    "dummy": DummyEncoderApplier,
    "DummyEncoder": DummyEncoderApplier,
    
    # Scaling
    "standard_scaler": StandardScalerApplier,
    "StandardScaler": StandardScalerApplier,
    "minmax_scaler": MinMaxScalerApplier,
    "MinMaxScaler": MinMaxScalerApplier,
    "robust_scaler": RobustScalerApplier,
    "RobustScaler": RobustScalerApplier,
    "maxabs_scaler": MaxAbsScalerApplier,
    "MaxAbsScaler": MaxAbsScalerApplier,
    
    # Imputation
    "simple_imputer": SimpleImputerApplier,
    "SimpleImputer": SimpleImputerApplier,
    "iterative_imputer": IterativeImputerApplier,
    "IterativeImputer": IterativeImputerApplier,
    "knn_imputer": KNNImputerApplier,
    "KNNImputer": KNNImputerApplier,
    
    # Transformations
    "power_transformer": PowerTransformerApplier,
    "PowerTransformer": PowerTransformerApplier,
    "simple_transformation": SimpleTransformationApplier,
    "SimpleTransformation": SimpleTransformationApplier,
    "general_transformation": GeneralTransformationApplier,
    "GeneralTransformation": GeneralTransformationApplier,
    
    # Bucketing
    "general_binning": GeneralBinningApplier,
    "GeneralBinning": GeneralBinningApplier,
    
    # Outliers
    "iqr": IQRApplier,
    "IQR": IQRApplier,
    "zscore": ZScoreApplier,
    "ZScore": ZScoreApplier,
    "winsorize": WinsorizeApplier,
    "Winsorize": WinsorizeApplier,
    "manual_bounds": ManualBoundsApplier,
    "ManualBounds": ManualBoundsApplier,
    
    # Cleaning
    "text_cleaning": TextCleaningApplier,
    "TextCleaning": TextCleaningApplier,
    "value_replacement": ValueReplacementApplier,
    "ValueReplacement": ValueReplacementApplier,
    "alias_replacement": AliasReplacementApplier,
    "AliasReplacement": AliasReplacementApplier,
    "invalid_value_replacement": InvalidValueReplacementApplier,
    "InvalidValueReplacement": InvalidValueReplacementApplier,
    
    # Feature Engineering
    "polynomial_features": PolynomialFeaturesApplier,
    "PolynomialFeatures": PolynomialFeaturesApplier,
    "feature_generation": FeatureGenerationApplier,
    "FeatureGenerationNode": FeatureGenerationApplier,
    "FeatureMath": FeatureGenerationApplier,
    
    # Feature Selection
    "variance_threshold": VarianceThresholdApplier,
    "VarianceThreshold": VarianceThresholdApplier,
    "correlation_threshold": CorrelationThresholdApplier,
    "CorrelationThreshold": CorrelationThresholdApplier,
    "univariate_selection": UnivariateSelectionApplier,
    "UnivariateSelection": UnivariateSelectionApplier,
    "model_based_selection": ModelBasedSelectionApplier,
    "ModelBasedSelection": ModelBasedSelectionApplier,
    "feature_selection": FeatureSelectionApplier,
    "FeatureSelection": FeatureSelectionApplier,
    
    # Others
    "casting": CastingApplier,
    "Casting": CastingApplier,
    "deduplicate": DeduplicateApplier,
    "Deduplicate": DeduplicateApplier,
    "drop_missing_columns": DropMissingColumnsApplier,
    "DropMissingColumns": DropMissingColumnsApplier,
    "drop_missing_rows": DropMissingRowsApplier,
    "DropMissingRows": DropMissingRowsApplier,
    "missing_indicator": MissingIndicatorApplier,
    "MissingIndicator": MissingIndicatorApplier,
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
        
        # If artifact_uri is a directory (doesn't end with .joblib or similar), we need to point to the specific file.
        # The PipelineEngine saves the FULL bundled artifact (model + transformers) using the job_id as the key.
        # It also saves the raw model using node_id, but we want the bundled one for deployment.
        if artifact_uri:
            # Check if it looks like a directory path (no extension)
            if not artifact_uri.endswith(".joblib") and not artifact_uri.endswith(".pkl"):
                # If it's a directory, append job_id to get the bundled artifact
                if "/" in artifact_uri or "\\" in artifact_uri:
                    final_uri = os.path.join(artifact_uri, job_id)
                else:
                    # Legacy case: artifact_uri might be just a node_id or partial path
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
    async def list_deployments(session: AsyncSession, limit: int = 50, skip: int = 0) -> list[Deployment]:
        """Lists deployment history."""
        stmt = select(Deployment).order_by(Deployment.created_at.desc()).limit(limit).offset(skip)
        result = await session.execute(stmt)
        return result.scalars().all()

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
        try:
            # Handle full path (absolute or relative with separators)
            if os.path.isabs(deployment.artifact_uri):
                base_path = os.path.dirname(deployment.artifact_uri)
                node_id = os.path.basename(deployment.artifact_uri)
            elif "/" in deployment.artifact_uri or "\\" in deployment.artifact_uri:
                # Check if it's a "pipeline_id/node_id" pattern that maps to exports/models
                # If the path doesn't exist locally, assume it's the internal format
                if not os.path.exists(deployment.artifact_uri) and not os.path.exists(os.path.dirname(deployment.artifact_uri)):
                    parts = deployment.artifact_uri.replace("\\", "/").split("/")
                    if len(parts) == 2:
                        pipeline_id = parts[0]
                        node_id = parts[1]
                        base_path = os.path.join(os.getcwd(), "exports", "models", pipeline_id)
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
                    base_path = os.path.join(os.getcwd(), "exports", "models", pipeline_id)
                else:
                    # Fallback for just node_id? This shouldn't happen with new logic.
                    # But if it does, we assume it's in exports/models/{job_id} ??
                    # No, we can't assume that.
                    raise ValueError(f"Invalid artifact URI format: {deployment.artifact_uri}")

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
                t_name = step.get("transformer_name")
                t_col = step.get("column_name")
                t_type = step.get("transformer_type")
                
                # Get object
                obj = t_objs.get((node_id, t_name, t_col))
                
                ApplierCls = APPLIER_MAP.get(t_type)
                
                if ApplierCls:
                    try:
                        applier = ApplierCls()
                        # Prepare params
                        # We use the fitted object as the source of truth for params
                        params = {}
                        if isinstance(obj, dict):
                            params = obj.copy()
                        
                        # Inject object into common keys for Appliers that need the raw object (like OneHotEncoder)
                        # Only inject if not already present (to avoid overwriting the real object with the wrapper dict)
                        if obj is not None:
                            if "encoder_object" not in params: params["encoder_object"] = obj
                            if "scaler_object" not in params: params["scaler_object"] = obj
                            if "imputer_object" not in params: params["imputer_object"] = obj
                            if "transformer_object" not in params: params["transformer_object"] = obj
                        
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
