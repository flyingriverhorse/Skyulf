import logging
import re
import traceback
from datetime import datetime

from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import get_settings
from backend.database.models import DataSource
from backend.ml_pipeline.execution.engine import PipelineEngine
from backend.ml_pipeline.execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.execution.strategies import JobStrategyFactory
from backend.ml_pipeline.constants import StepType
from backend.realtime.events import JobEvent, publish_job_event

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
def run_pipeline_task(job_id: str, pipeline_config_dict: dict):  # noqa: C901
    """
    Background task to execute a full ML pipeline.
    """
    logger.info(f"Starting pipeline execution for job {job_id}")
    session = get_db_session()

    try:
        # 1. Get Job and Strategy
        job, strategy = JobStrategyFactory.find_job(session, job_id)

        if not job or not strategy:
            logger.error(f"Job {job_id} not found in any known job tables")
            return

        # Update status to running
        job.status = "running"
        job.started_at = datetime.now()
        job.progress = 0
        session.commit()
        publish_job_event(
            JobEvent(event="status", job_id=job_id, status="running", progress=0)
        )

        # 2. Setup Artifact Store
        from backend.ml_pipeline.artifacts.factory import ArtifactFactory

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

            if step_type == StepType.DATA_LOADER:
                dataset_id = params.get("dataset_id") or params.get("path")
                if dataset_id and str(dataset_id).startswith("s3://"):
                    is_s3_source = True
                break

        # Construct folder name: dataset_name + time_created + job_id
        dataset_name = "unknown_dataset"
        if job.dataset_source_id:
            # Try to find by source_id
            ds = (
                session.query(DataSource)
                .filter(DataSource.source_id == job.dataset_source_id)
                .first()
            )
            if not ds:
                # Try to find by id (if it was stored as string)
                try:
                    ds_id = int(job.dataset_source_id)
                    ds = session.query(DataSource).filter(DataSource.id == ds_id).first()
                except ValueError:
                    pass

            if ds:
                dataset_name = ds.name

        # Sanitize dataset name
        dataset_name = re.sub(r"[^a-zA-Z0-9_-]", "_", dataset_name)

        # Format time
        # Use datetime.now() to ensure we get the local server time (user's time if running locally)
        # instead of potentially UTC time from the database
        time_created = datetime.now().strftime("%Y%m%d_%H%M%S")

        folder_name = f"{dataset_name}_{time_created}_{job_id}"

        artifact_store, base_artifact_uri = ArtifactFactory.create_store_for_job(
            job_id, is_s3_source, artifact_path_name=folder_name
        )

        # Save the URI to the job immediately so we know where to look later
        job.artifact_uri = base_artifact_uri
        session.commit()

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

        # Add version info to logs using strategy
        job_logs.append(strategy.get_initial_log(job))

        def log_callback(msg):
            nonlocal last_log_update
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {msg}"
            logger.info(f"[Job {job_id}] {msg}")
            job_logs.append(log_entry)

            # Update logs in DB every 2 seconds to avoid locking
            if (datetime.now() - last_log_update).total_seconds() > 2:
                try:
                    assert job is not None  # narrowing lost in closure
                    job.logs = list(job_logs)
                    session.commit()
                    last_log_update = datetime.now()
                    # Hint the frontend to refresh; payload stays minimal
                    # because the WS is an invalidator, not a data source.
                    publish_job_event(
                        JobEvent(
                            event="progress",
                            job_id=job_id,
                            status=job.status,
                            progress=job.progress,
                            current_step=job.current_step,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to update logs: {e}")

        engine = PipelineEngine(artifact_store, catalog=catalog, log_callback=log_callback)

        # 5. Run Pipeline
        result = engine.run(pipeline_config, job_id=job_id, dataset_name=dataset_name)

        # 6. Handle Result
        job.logs = job_logs  # Final log update

        # Cancellation guard: the user may have clicked Stop while this
        # worker was still in `model.fit`. Re-read the row from the DB so
        # we don't blindly overwrite a `cancelled` status (and its metrics
        # / best_params) with completed results. Without this, the
        # non-Celery thread path (which can't be hard-killed) would always
        # show a finished training run despite the cancel.
        session.refresh(job, ["status"])
        if job.status == "cancelled":
            logger.info(f"Job {job_id} was cancelled mid-run; skipping result write")
            session.commit()
            return

        if result.status == "success":
            job.status = "completed"
            job.progress = 100
            job.finished_at = datetime.now()
            job.artifact_uri = base_artifact_uri

            # Delegate success handling to strategy (metrics, results, etc.)
            strategy.handle_success(job, result)

        else:
            # Extract error from the failed node if possible
            error_msg = "Pipeline execution failed"
            if result.node_results:
                for node_res in result.node_results.values():
                    if node_res.status == "failed":
                        error_msg = f"Error in node {node_res.node_id}: {node_res.error}"
                        break

            # Delegate failure handling to strategy
            strategy.handle_failure(job, error_msg)

        session.commit()
        logger.info(f"Job {job_id} finished with status: {job.status}")
        publish_job_event(
            JobEvent(event="status", job_id=job_id, status=job.status, progress=job.progress)
        )

    except Exception as e:
        logger.error(f"Job {job_id} failed with exception: {str(e)}")
        logger.error(traceback.format_exc())
        if session:
            # Re-query to ensure session is valid
            job, strategy = JobStrategyFactory.find_job(session, job_id)

            if job and strategy:
                # Same cancellation guard for the exception path: a job that
                # was cancelled and then crashed in cleanup must stay
                # cancelled, not flip to failed.
                if job.status == "cancelled":
                    session.commit()
                else:
                    strategy.handle_failure(job, str(e))
                    session.commit()
                publish_job_event(
                    JobEvent(event="status", job_id=job_id, status=job.status)
                )
    finally:
        session.close()
