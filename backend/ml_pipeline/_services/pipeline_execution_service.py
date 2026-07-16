"""Pipeline execution service.

All business logic for running a pipeline job lives here so it can be
called (and unit-tested) without a Celery worker context.
``tasks.run_pipeline_task`` is the only caller in production; tests can
call ``execute_pipeline`` directly with a real or mocked session.
"""

import contextlib
import logging
import re
import traceback
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.database.models import DataSource, MLJob
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import (
    NodeConfig,
    PipelineConfig,
    PipelineExecutionResult,
)
from backend.ml_pipeline._execution.strategies import JobStrategy, JobStrategyFactory
from backend.ml_pipeline.artifacts.store import ArtifactStore
from backend.ml_pipeline.constants import StepType
from backend.realtime.events import JobEvent, publish_job_event

logger = logging.getLogger(__name__)


def _is_s3_pipeline_source(pipeline_config_dict: dict) -> bool:
    """Checks whether the pipeline's data loader node points at an s3:// source."""
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
                return True
            break
    return False


def _create_artifact_store(
    session: Session, job: MLJob, job_id: str, pipeline_config_dict: dict
) -> tuple[ArtifactStore, str, str]:
    """Determines the artifact store location for a job and creates the store.

    Returns the (artifact_store, base_artifact_uri, dataset_name) tuple.
    """
    from backend.ml_pipeline.artifacts.factory import ArtifactFactory

    is_s3_source = _is_s3_pipeline_source(pipeline_config_dict)

    dataset_name = _resolve_dataset_name(session, job)
    time_created = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{dataset_name}_{time_created}_{job_id}"

    artifact_store, base_artifact_uri = ArtifactFactory.create_store_for_job(
        job_id, is_s3_source, artifact_path_name=folder_name
    )
    return artifact_store, base_artifact_uri, dataset_name


def _build_pipeline_config(pipeline_config_dict: dict) -> PipelineConfig:
    """Reconstructs a typed PipelineConfig from the raw pipeline config dict."""
    nodes = [
        NodeConfig(
            node_id=n["node_id"],
            step_type=n["step_type"],
            params=n["params"],
            inputs=n["inputs"],
        )
        for n in pipeline_config_dict["nodes"]
    ]
    return PipelineConfig(
        pipeline_id=pipeline_config_dict["pipeline_id"],
        nodes=nodes,
        metadata=pipeline_config_dict.get("metadata", {}),
    )


def _make_log_callback(session: Session, job: MLJob, job_id: str, job_logs: list[str]):
    """Builds a log_callback closure that appends to job_logs and throttles DB writes."""
    state = {"last_log_update": datetime.now()}

    def log_callback(msg: str) -> None:
        """Appends a timestamped message to job_logs, persisting to the DB at most every 2s."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        job_logs.append(f"[{timestamp}] {msg}")
        logger.info(f"[Job {job_id}] {msg}")

        # Throttle DB writes to avoid row-level locking churn.
        if (datetime.now() - state["last_log_update"]).total_seconds() > 2:
            try:
                if job is None:
                    raise RuntimeError(f"Job {job_id} is no longer available for log update.")
                job.logs = list(job_logs)
                session.commit()
                state["last_log_update"] = datetime.now()
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

    return log_callback


def _write_pipeline_result(
    session: Session,
    job: MLJob,
    strategy: JobStrategy,
    job_id: str,
    result: PipelineExecutionResult,
    base_artifact_uri: str,
) -> None:
    """Persists the pipeline run's outcome (success or failure) onto the job row."""
    if result.status == "success":
        job.status = "completed"
        job.progress = 100
        job.finished_at = datetime.now(UTC)
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


def _handle_execution_exception(session: Session, job_id: str, exc: Exception) -> None:
    """Records a failure state for the job after an unhandled exception during execution."""
    logger.error(f"Job {job_id} failed with exception: {exc}")
    logger.error(traceback.format_exc())

    try:
        session.rollback()
        job, strategy = JobStrategyFactory.find_job(session, job_id)
        if job and strategy:
            if job.status != "cancelled":
                strategy.handle_failure(job, str(exc))
            session.commit()
            publish_job_event(JobEvent(event="status", job_id=job_id, status=job.status))
    except Exception:
        logger.exception("Failed to record failure state for job %s", job_id)


def execute_pipeline(job_id: str, pipeline_config_dict: dict, session: Session) -> None:
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
        job.started_at = datetime.now(UTC)
        job.progress = 0
        session.commit()
        publish_job_event(JobEvent(event="status", job_id=job_id, status="running", progress=0))

        # 2. Determine artifact store location.
        artifact_store, base_artifact_uri, dataset_name = _create_artifact_store(
            session, job, job_id, pipeline_config_dict
        )
        job.artifact_uri = base_artifact_uri
        session.commit()

        # 3. Reconstruct typed PipelineConfig from the raw dict.
        pipeline_config = _build_pipeline_config(pipeline_config_dict)

        # 4. Build catalog + engine.
        from backend.data.catalog import create_catalog_from_options

        storage_options = pipeline_config_dict.get("storage_options")
        catalog = create_catalog_from_options(
            storage_options, pipeline_config.nodes, session=session
        )

        job_logs = [strategy.get_initial_log(job)]
        log_callback = _make_log_callback(session, job, job_id, job_logs)

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

        _write_pipeline_result(session, job, strategy, job_id, result, base_artifact_uri)

    except Exception as exc:
        _handle_execution_exception(session, job_id, exc)


def _resolve_dataset_name(session: Session, job: object) -> str:
    """Return a filesystem-safe dataset name derived from the job's source."""
    dataset_name = "unknown_dataset"
    source_id = getattr(job, "dataset_source_id", None)

    if source_id:
        ds = session.query(DataSource).filter(DataSource.source_id == source_id).first()
        if not ds:
            with contextlib.suppress(ValueError):
                ds = session.query(DataSource).filter(DataSource.id == int(source_id)).first()
        if ds:
            dataset_name = ds.name

    return re.sub(r"[^a-zA-Z0-9_-]", "_", dataset_name)
