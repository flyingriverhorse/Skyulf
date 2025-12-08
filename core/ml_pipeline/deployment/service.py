from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from core.database.models import Deployment, TrainingJob, HyperparameterTuningJob
from core.ml_pipeline.artifacts.local import LocalArtifactStore
from core.ml_pipeline.execution.jobs import JobManager
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

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
        if hasattr(model, "predict"):
            predictions = model.predict(df)
            if hasattr(predictions, "tolist"):
                return predictions.tolist()
            return list(predictions)
        else:
            raise ValueError("Loaded artifact is not a valid predictor")
