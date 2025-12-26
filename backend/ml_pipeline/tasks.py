import logging
import os
import traceback
from datetime import datetime

from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import get_settings
from backend.data.catalog import FileSystemCatalog, SmartCatalog
from backend.database.models import DataSource, HyperparameterTuningJob, TrainingJob
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.artifacts.s3 import S3ArtifactStore
from backend.ml_pipeline.execution.engine import PipelineEngine
from backend.ml_pipeline.execution.schemas import NodeConfig, PipelineConfig
from backend.utils.file_utils import extract_file_path_from_source

logger = logging.getLogger(__name__)

# Helper to get sync session


def get_db_session():
    settings = get_settings()
    if settings.DATABASE_URL.startswith("sqlite+aiosqlite://"):
        sync_url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
    else:
        sync_url = settings.DATABASE_URL.replace(
            "postgresql+asyncpg://", "postgresql+psycopg2://"
        )

    engine = create_engine(sync_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


@shared_task(name="core.ml_pipeline.tasks.run_pipeline_task")
def run_pipeline_task(job_id: str, pipeline_config_dict: dict):  # noqa: C901
    """
    Background task to execute a full ML pipeline.
    """
    logger.info(f"Starting pipeline execution for job {job_id}")
    session = get_db_session()

    try:
        # 1. Get Job (Try TrainingJob first, then HyperparameterTuningJob)
        job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            job = (
                session.query(HyperparameterTuningJob)
                .filter(HyperparameterTuningJob.id == job_id)
                .first()
            )

        if not job:
            logger.error(
                f"Job {job_id} not found in TrainingJob or HyperparameterTuningJob"
            )
            return

        # Update status to running
        job.status = "running"
        job.started_at = datetime.now()
        job.progress = 0
        session.commit()

        # 2. Setup Artifact Store
        settings = get_settings()
        
        from typing import Union
        from backend.ml_pipeline.artifacts.local import LocalArtifactStore
        from backend.ml_pipeline.artifacts.s3 import S3ArtifactStore
        
        artifact_store: Union[S3ArtifactStore, LocalArtifactStore]

        # Check if S3 Artifact Store is configured
        s3_bucket = settings.S3_ARTIFACT_BUCKET
        
        # Determine Data Source Type to decide on Artifact Store
        is_s3_source = False
        nodes = pipeline_config_dict.get("nodes", [])
        for node in nodes:
            # Check dictionary access since it might be a dict or object depending on serialization
            # Here it is passed as a dict
            if isinstance(node, dict):
                step_type = node.get("step_type")
                params = node.get("params", {})
            else:
                # Fallback if it's an object
                step_type = getattr(node, "step_type", "")
                params = getattr(node, "params", {})

            if step_type == "data_loader":
                dataset_id = params.get("dataset_id") or params.get("path")
                # print(f"DEBUG: Found data_loader. dataset_id={dataset_id}")
                if dataset_id and str(dataset_id).startswith("s3://"):
                    is_s3_source = True
                break
        
        # print(f"DEBUG: is_s3_source={is_s3_source}, s3_bucket={s3_bucket}, UPLOAD_TO_S3={settings.UPLOAD_TO_S3_FOR_LOCAL_FILES}")
        
        use_s3_artifacts = False
        if s3_bucket:
            if is_s3_source:
                # Default is S3, but user can force local storage via config
                if not settings.SAVE_S3_ARTIFACTS_LOCALLY:
                    use_s3_artifacts = True
            elif settings.UPLOAD_TO_S3_FOR_LOCAL_FILES:
                # Local source, but user wants to upload to S3
                use_s3_artifacts = True

        if use_s3_artifacts:
            # Use S3 for artifacts
            # We use job_id as prefix to keep things organized
            
            # Prepare storage options from settings
            s3_options = {}
            if settings.AWS_ACCESS_KEY_ID:
                s3_options["key"] = settings.AWS_ACCESS_KEY_ID
            if settings.AWS_SECRET_ACCESS_KEY:
                s3_options["secret"] = settings.AWS_SECRET_ACCESS_KEY
            if settings.AWS_ENDPOINT_URL:
                s3_options["endpoint_url"] = settings.AWS_ENDPOINT_URL
            if settings.AWS_DEFAULT_REGION:
                s3_options["region"] = settings.AWS_DEFAULT_REGION
                
            artifact_store = S3ArtifactStore(bucket_name=s3_bucket, prefix=job_id, storage_options=s3_options)
            # For S3, the URI is the s3 path
            base_artifact_uri = f"s3://{s3_bucket}/{job_id}"
        else:
            # Fallback to Local
            base_artifact_path = os.path.join(settings.TRAINING_ARTIFACT_DIR, job_id)
            os.makedirs(base_artifact_path, exist_ok=True)
            artifact_store = LocalArtifactStore(base_artifact_path)
            base_artifact_uri = base_artifact_path

        # 3. Reconstruct Pipeline Config
        # Convert dict back to dataclasses
        nodes = []
        for n in pipeline_config_dict["nodes"]:
            # Note: ID resolution is now handled by SmartCatalog
            nodes.append(
                NodeConfig(
                    node_id=n["node_id"],
                    step_type=n["step_type"],
                    params=n["params"],
                    inputs=n["inputs"],
                )
            )

        pipeline_config = PipelineConfig(
            pipeline_id=pipeline_config_dict["pipeline_id"],
            nodes=nodes,
            metadata=pipeline_config_dict.get("metadata", {}),
        )

        # 4. Initialize Engine with Progress Callback
        # If storage_options are passed in config (from API resolution), use them to init S3Catalog
        # Otherwise use SmartCatalog for dynamic resolution (Sync only)
        
        storage_options = pipeline_config_dict.get("storage_options")
        from backend.data.catalog import create_catalog_from_options
        
        catalog = create_catalog_from_options(storage_options, nodes, session=session)
        
        job_logs = []
        last_log_update = datetime.now()

        # Add version info to logs
        timestamp = datetime.now().strftime("%H:%M:%S")
        if isinstance(job, TrainingJob):
            job_logs.append(f"[{timestamp}] Training Job Version: {job.version}")
        elif isinstance(job, HyperparameterTuningJob):
            job_logs.append(f"[{timestamp}] Tuning Job Run: {job.run_number}")

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

        engine = PipelineEngine(
            artifact_store, catalog=catalog, log_callback=log_callback
        )

        # 5. Run Pipeline
        result = engine.run(pipeline_config, job_id=job_id)

        # 6. Handle Result
        job.logs = job_logs  # Final log update

        if result.status == "success":
            job.status = "completed"
            job.progress = 100
            job.finished_at = datetime.now()
            job.artifact_uri = base_artifact_uri

            # Extract metrics from the last node if available
            if result.node_results:
                last_node_id = list(result.node_results.keys())[-1]
                last_result = result.node_results[last_node_id]

                final_metrics = (
                    last_result.metrics.copy() if last_result.metrics else {}
                )

                # Collect dropped columns from all nodes (e.g. Feature Selection, Drop Columns)
                all_dropped_columns = []

                for node_res in result.node_results.values():
                    if node_res.metrics and "dropped_columns" in node_res.metrics:
                        cols = node_res.metrics["dropped_columns"]
                        if isinstance(cols, list):
                            all_dropped_columns.extend(cols)

                if all_dropped_columns:
                    # Deduplicate
                    all_dropped_columns = list(set(all_dropped_columns))
                    final_metrics["dropped_columns"] = all_dropped_columns

                job.metrics = final_metrics

                # Special handling for HyperparameterTuningJob
                if isinstance(job, HyperparameterTuningJob):
                    if "best_params" in final_metrics:
                        job.best_params = final_metrics["best_params"]
                    if "best_score" in final_metrics:
                        job.best_score = final_metrics["best_score"]
                    if "trials" in final_metrics:
                        job.results = final_metrics["trials"]

        else:
            job.status = "failed"
            # Extract error from the failed node if possible
            error_msg = "Pipeline execution failed"
            if result.node_results:
                for node_res in result.node_results.values():
                    if node_res.status == "failed":
                        error_msg = (
                            f"Error in node {node_res.node_id}: {node_res.error}"
                        )
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
                job = (
                    session.query(HyperparameterTuningJob)
                    .filter(HyperparameterTuningJob.id == job_id)
                    .first()
                )

            if job:
                job.status = "failed"
                job.error_message = str(e)
                job.finished_at = datetime.now()
                session.commit()
    finally:
        session.close()
