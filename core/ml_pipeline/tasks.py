import logging
import traceback
from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import os

from core.config import get_settings
from core.database.models import TrainingJob, HyperparameterTuningJob, DataSource
from core.ml_pipeline.execution.engine import PipelineEngine
from core.ml_pipeline.execution.schemas import PipelineConfig, NodeConfig
from core.ml_pipeline.artifacts.local import LocalArtifactStore

logger = logging.getLogger(__name__)

# Helper to get sync session
def get_db_session():
    settings = get_settings()
    if settings.DATABASE_URL.startswith("sqlite+aiosqlite://"):
        sync_url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
    else:
        sync_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    
    engine = create_engine(sync_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

@shared_task(name="core.ml_pipeline.tasks.run_pipeline_task")
def run_pipeline_task(job_id: str, pipeline_config_dict: dict):
    """
    Background task to execute a full ML pipeline.
    """
    logger.info(f"Starting pipeline execution for job {job_id}")
    session = get_db_session()
    
    try:
        # 1. Get Job (Try TrainingJob first, then HyperparameterTuningJob)
        job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            job = session.query(HyperparameterTuningJob).filter(HyperparameterTuningJob.id == job_id).first()
            
        if not job:
            logger.error(f"Job {job_id} not found in TrainingJob or HyperparameterTuningJob")
            return

        # Update status to running
        job.status = "running"
        job.started_at = datetime.now()
        job.progress = 0
        session.commit()

        # 2. Setup Artifact Store
        # We use a persistent path for training jobs
        settings = get_settings()
        base_artifact_path = os.path.join(settings.TRAINING_ARTIFACT_DIR, job_id)
        os.makedirs(base_artifact_path, exist_ok=True)
        artifact_store = LocalArtifactStore(base_artifact_path)

        # 3. Reconstruct Pipeline Config
        # Convert dict back to dataclasses
        nodes = []
        for n in pipeline_config_dict['nodes']:
            nodes.append(NodeConfig(
                node_id=n['node_id'],
                step_type=n['step_type'],
                params=n['params'],
                inputs=n['inputs']
            ))
            
        pipeline_config = PipelineConfig(
            pipeline_id=pipeline_config_dict['pipeline_id'],
            nodes=nodes,
            metadata=pipeline_config_dict.get('metadata', {})
        )

        # 4. Initialize Engine with Progress Callback
        job_logs = []
        last_log_update = datetime.now()
        
        def log_callback(msg):
            nonlocal last_log_update
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {msg}"
            logger.info(f"[Job {job_id}] {msg}")
            job_logs.append(log_entry)
            
            # Update logs in DB every 2 seconds to avoid locking
            if (datetime.now() - last_log_update).total_seconds() > 2:
                try:
                    job.logs = list(job_logs)
                    session.commit()
                    last_log_update = datetime.now()
                except Exception as e:
                    logger.warning(f"Failed to update logs: {e}")

        engine = PipelineEngine(artifact_store, log_callback=log_callback)

        # 5. Run Pipeline
        result = engine.run(pipeline_config, job_id=job_id)

        # 6. Handle Result
        job.logs = job_logs # Final log update
        
        if result.status == "success":
            job.status = "completed"
            job.progress = 100
            job.finished_at = datetime.now()
            job.artifact_uri = base_artifact_path
            
            # Extract metrics from the last node if available
            if result.node_results:
                last_node_id = list(result.node_results.keys())[-1]
                last_result = result.node_results[last_node_id]
                if last_result.metrics:
                    job.metrics = last_result.metrics
                    
                    # Special handling for HyperparameterTuningJob
                    if isinstance(job, HyperparameterTuningJob):
                        if "best_params" in last_result.metrics:
                            job.best_params = last_result.metrics["best_params"]
                        if "best_score" in last_result.metrics:
                            job.best_score = last_result.metrics["best_score"]
                        if "trials" in last_result.metrics:
                            job.results = last_result.metrics["trials"]
            
        else:
            job.status = "failed"
            # Extract error from the failed node if possible
            error_msg = "Pipeline execution failed"
            if result.node_results:
                for node_res in result.node_results.values():
                    if node_res.status == "failed":
                        error_msg = f"Error in node {node_res.node_id}: {node_res.error}"
                        break
            job.error_message = error_msg
            job.finished_at = datetime.now()

        session.commit()
        logger.info(f"Job {job_id} finished with status: {job.status}")

    except Exception as e:
        logger.error(f"Job {job_id} failed with exception: {str(e)}")
        logger.error(traceback.format_exc())
        if session:
            # Re-query to ensure session is valid
            job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                job = session.query(HyperparameterTuningJob).filter(HyperparameterTuningJob.id == job_id).first()
                
            if job:
                job.status = "failed"
                job.error_message = str(e)
                job.finished_at = datetime.now()
                session.commit()
    finally:
        session.close()
