"""Pipeline execution service.

All business logic for running a pipeline job lives here so it can be
called (and unit-tested) without a Celery worker context.
``tasks.run_pipeline_task`` is the only caller in production; tests can
call ``execute_pipeline`` directly with a real or mocked session.
"""

import logging
import re
import traceback
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.database.models import DataSource
from backend.ml_pipeline.execution.engine import PipelineEngine
from backend.ml_pipeline.execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.execution.strategies import JobStrategyFactory
from backend.ml_pipeline.constants import StepType
from backend.realtime.events import JobEvent, publish_job_event

logger = logging.getLogger(__name__)


def execute_pipeline(job_id: str, pipeline_config_dict: dict, session: Session) -> None:  # noqa: C901
    """Run a full ML pipeline for *job_id* using the supplied sync *session*.

    Raises nothing — all exceptions are caught, written to the job row, and
    re-published via the WebSocket event bus. The caller (Celery task) only
    needs to ensure the session is closed in a ``finally`` block.
    """
    logger.info(f"Starting pipeline execution for job {job_id}")

    try:
        # 1. Locate job + strategy.
        job, strategy = JobStrategyFactory.find_job(session, job_id)
        if not job or not strategy:
            logger.error(f"Job {job_id} not found in any known job tables")
            return

        job.status = "running"
        job.started_at = datetime.now()
        job.progress = 0
        session.commit()
        publish_job_event(JobEvent(event="status", job_id=job_id, status="running", progress=0))

        # 2. Determine artifact store location.
        from backend.ml_pipeline.artifacts.factory import ArtifactFactory

        is_s3_source = False
        for node in pipeline_config_dict.get("nodes", []):
            if isinstance(node, dict):
                step_type = node.get("step_type")
                params = node.get("params", {})
            else:
                step_type = getattr(node, "step_type", "")
                params = getattr(node, "params", {})

            if step_type == StepType.DATA_LOADER:
                dataset_id = params.get("dataset_id") or params.get("path")
                if dataset_id and str(dataset_id).startswith("s3://"):
                    is_s3_source = True
                break

        dataset_name = _resolve_dataset_name(session, job)
        time_created = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{dataset_name}_{time_created}_{job_id}"

        artifact_store, base_artifact_uri = ArtifactFactory.create_store_for_job(
            job_id, is_s3_source, artifact_path_name=folder_name
        )
        job.artifact_uri = base_artifact_uri
        session.commit()

        # 3. Reconstruct typed PipelineConfig from the raw dict.
        nodes = [
            NodeConfig(
                node_id=n["node_id"],
                step_type=n["step_type"],
                params=n["params"],
                inputs=n["inputs"],
            )
            for n in pipeline_config_dict["nodes"]
        ]
        pipeline_config = PipelineConfig(
            pipeline_id=pipeline_config_dict["pipeline_id"],
            nodes=nodes,
            metadata=pipeline_config_dict.get("metadata", {}),
        )

        # 4. Build catalog + engine.
        from backend.data.catalog import create_catalog_from_options

        storage_options = pipeline_config_dict.get("storage_options")
        catalog = create_catalog_from_options(storage_options, nodes, session=session)

        job_logs = [strategy.get_initial_log(job)]
        last_log_update = datetime.now()

        def log_callback(msg: str) -> None:
            nonlocal last_log_update
            timestamp = datetime.now().strftime("%H:%M:%S")
            job_logs.append(f"[{timestamp}] {msg}")
            logger.info(f"[Job {job_id}] {msg}")

            # Throttle DB writes to avoid row-level locking churn.
            if (datetime.now() - last_log_update).total_seconds() > 2:
                try:
                    assert job is not None
                    job.logs = list(job_logs)
                    session.commit()
                    last_log_update = datetime.now()
                    publish_job_event(
                        JobEvent(
                            event="progress",
                            job_id=job_id,
                            status=job.status,
                            progress=job.progress,
                            current_step=job.current_step,
                        )
                    )
                except Exception as exc:
                    logger.warning(f"Failed to update logs: {exc}")

        engine = PipelineEngine(artifact_store, catalog=catalog, log_callback=log_callback)

        # 5. Execute.
        result = engine.run(pipeline_config, job_id=job_id, dataset_name=dataset_name)

        # 6. Write result — but respect a concurrent cancellation.
        job.logs = job_logs
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
            strategy.handle_success(job, result)
        else:
            error_msg = "Pipeline execution failed"
            for node_res in (result.node_results or {}).values():
                if node_res.status == "failed":
                    error_msg = f"Error in node {node_res.node_id}: {node_res.error}"
                    break
            strategy.handle_failure(job, error_msg)

        session.commit()
        logger.info(f"Job {job_id} finished with status: {job.status}")
        publish_job_event(
            JobEvent(event="status", job_id=job_id, status=job.status, progress=job.progress)
        )

    except Exception as exc:
        logger.error(f"Job {job_id} failed with exception: {exc}")
        logger.error(traceback.format_exc())

        job, strategy = JobStrategyFactory.find_job(session, job_id)
        if job and strategy:
            if job.status != "cancelled":
                strategy.handle_failure(job, str(exc))
            session.commit()
            publish_job_event(JobEvent(event="status", job_id=job_id, status=job.status))


def _resolve_dataset_name(session: Session, job: object) -> str:
    """Return a filesystem-safe dataset name derived from the job's source."""
    dataset_name = "unknown_dataset"
    source_id = getattr(job, "dataset_source_id", None)

    if source_id:
        ds = session.query(DataSource).filter(DataSource.source_id == source_id).first()
        if not ds:
            try:
                ds = (
                    session.query(DataSource)
                    .filter(DataSource.id == int(source_id))
                    .first()
                )
            except ValueError:
                pass
        if ds:
            dataset_name = ds.name

    return re.sub(r"[^a-zA-Z0-9_-]", "_", dataset_name)
