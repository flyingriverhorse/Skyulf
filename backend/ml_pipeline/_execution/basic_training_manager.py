import uuid
from datetime import UTC, datetime
from typing import Any
from typing import cast as t_cast
from typing import cast as type_cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database.models import TrainingJob
from backend.ml_pipeline._execution.graph_utils import extract_job_details
from backend.ml_pipeline._execution.job_manager_base import TrainingJobManagerBase
from backend.ml_pipeline._execution.schemas import JobInfo, JobStatus
from backend.ml_pipeline._execution.utils import (
    get_dataset_map,
    parse_branch_info,
    resolve_dataset_name,
)
from backend.ml_pipeline.model_registry.service import ModelRegistryService


class BasicTrainingManager(TrainingJobManagerBase):
    @staticmethod
    async def create_training_job(
        session: AsyncSession,
        pipeline_id: str,
        node_id: str,
        dataset_id: str = "unknown",
        user_id: int | None = None,
        model_type: str = "unknown",
        graph: dict[str, Any] | None = None,
        is_preview: bool = False,
        branch_index: int = 0,
    ) -> str:
        """Creates a new training job in the database (Async)."""
        job_id = str(uuid.uuid4())
        graph = graph or {}

        if is_preview:
            version = 0
            model_type_val = "preview"
        else:
            version = await ModelRegistryService.get_next_version(
                session, dataset_id, model_type, "training"
            )
            model_type_val = model_type

        job = TrainingJob(
            id=job_id,
            pipeline_id=pipeline_id,
            node_id=node_id,
            dataset_source_id=dataset_id,
            user_id=user_id,
            status=JobStatus.QUEUED.value,
            run_mode="fixed",
            version=version,
            model_type=model_type_val,
            graph=graph,
            job_metadata={"branch_index": branch_index},
            started_at=datetime.now(UTC),
        )

        session.add(job)
        await session.commit()
        return job_id

    @staticmethod
    def map_training_job_to_info(job: TrainingJob, dataset_name: str | None) -> JobInfo:
        # Extract details from graph
        (
            hyperparameters,
            target_column,
            dropped_columns,
        ) = extract_job_details(type_cast(dict[str, Any], job.graph), job.node_id)

        # Also check job metrics for runtime dropped columns (e.g. from Feature Selection)
        if job.metrics and isinstance(job.metrics, dict) and "dropped_columns" in job.metrics:
            metrics_dropped = job.metrics["dropped_columns"]
            if isinstance(metrics_dropped, list):
                dropped_columns.extend(metrics_dropped)

        # Deduplicate
        dropped_columns = list(set(dropped_columns))

        # Fallback for hyperparameters if not found in graph (though extract_job_details should find it)
        if not hyperparameters:
            hyperparameters = type_cast(dict[str, Any] | None, job.hyperparameters)

        return JobInfo(
            job_id=job.id,
            pipeline_id=job.pipeline_id,
            node_id=job.node_id,
            dataset_id=t_cast(str | None, job.dataset_source_id),
            dataset_name=dataset_name,
            job_type="training",
            status=JobStatus(job.status),
            start_time=job.started_at,
            end_time=job.finished_at,
            error=job.error_message,
            result={"metrics": job.metrics},
            model_type=job.model_type,
            hyperparameters=t_cast(dict[str, Any], hyperparameters),
            created_at=t_cast(datetime, job.created_at),
            metrics=t_cast(dict[str, Any] | None, job.metrics),
            version=t_cast(int | None, job.version),
            target_column=target_column,
            dropped_columns=dropped_columns,
            logs=t_cast(list[str] | None, job.logs),
            graph=type_cast(dict[str, Any], job.graph),
            promoted_at=job.promoted_at,
            parent_pipeline_id=parse_branch_info(job.pipeline_id)[0],
            branch_index=parse_branch_info(job.pipeline_id)[1],
        )

    @staticmethod
    async def cancel_training_job(session: AsyncSession, job_id: str) -> bool:
        """Cancels a training job if it is running or queued.

        Flips the DB row to CANCELLED *and* revokes the underlying Celery
        task (terminate=True so SIGTERM kills any in-flight `model.fit`).
        Without the revoke, the worker would keep training and overwrite
        the status back to COMPLETED via `update_status_sync` once it
        finished — which is the bug users reported as "Stop doesn't stop
        training". The `update_status_sync` cancelled-state guard further
        protects against any late writes that race past the revoke.
        """
        return await TrainingJobManagerBase._cancel_job(
            session, TrainingJob, job_id, run_mode="fixed"
        )

    @staticmethod
    def _update_training_result(job: TrainingJob, result: dict[str, Any]):
        if "metrics" in result:
            job.metrics = result["metrics"]
        if "artifact_uri" in result:
            job.artifact_uri = result["artifact_uri"]
        if "hyperparameters" in result:
            job.hyperparameters = result["hyperparameters"]

    @staticmethod
    def _append_job_logs(job: TrainingJob, logs: list[str]) -> None:
        """Appends new log lines to a job's existing logs list, in place."""
        TrainingJobManagerBase._append_job_logs(job, logs)

    @staticmethod
    def _handle_cancelled_status_update(
        session: Session, job: TrainingJob, logs: list[str] | None
    ) -> bool:
        """Handle a status update for an already-cancelled job: only append logs, never revive it."""
        return TrainingJobManagerBase._handle_cancelled_status_update(session, job, logs)

    @staticmethod
    def _apply_status_update_fields(
        job: TrainingJob,
        status: JobStatus | None,
        error: str | None,
        logs: list[str] | None,
        result: dict[str, Any] | None,
    ) -> None:
        """Apply status/error/logs/result fields onto a training job."""
        if status:
            job.status = status.value
        if error:
            job.error_message = error

        if logs:
            BasicTrainingManager._append_job_logs(job, logs)

        if result:
            BasicTrainingManager._update_training_result(job, result)

        if status in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.SUCCEEDED,
        ]:
            job.finished_at = datetime.now(UTC)

    @staticmethod
    def update_status_sync(
        session: Session,
        job_id: str,
        status: JobStatus | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
        logs: list[str] | None = None,
    ) -> bool:
        """Updates training job status (Sync). Returns True if job found and updated."""
        return TrainingJobManagerBase._update_status_sync(
            session,
            TrainingJob,
            job_id,
            status,
            error,
            result,
            logs,
            BasicTrainingManager._apply_status_update_fields,
            run_mode="fixed",
        )

    @staticmethod
    async def get_training_job(session: AsyncSession, job_id: str) -> JobInfo | None:
        """Retrieves a training job by ID."""
        # 1. Fetch Job
        stmt = select(TrainingJob).where(
            TrainingJob.id == job_id, TrainingJob.run_mode == "fixed"
        )
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            return None

        # 2. Resolve Dataset Name
        dataset_name = await resolve_dataset_name(session, job.dataset_source_id)

        return BasicTrainingManager.map_training_job_to_info(job, dataset_name)

    @staticmethod
    async def list_training_jobs(
        session: AsyncSession,
        limit: int | None = None,
        skip: int = 0,
    ) -> list[JobInfo]:
        """Lists recent training jobs (Async)."""
        effective_limit = limit if limit is not None else get_settings().DEFAULT_PAGE_SIZE
        # 1. Fetch all DataSources for robust name resolution
        ds_map = await get_dataset_map(session)

        # 2. Fetch Jobs
        result_train = await session.execute(
            select(TrainingJob)
            .where(TrainingJob.run_mode == "fixed", TrainingJob.model_type != "preview")
            .order_by(TrainingJob.started_at.desc())
            .limit(effective_limit)
            .offset(skip)
        )
        train_rows = result_train.scalars().all()

        # 3. Map to Info
        jobs = []
        for job in train_rows:
            ds_id = str(job.dataset_source_id) if job.dataset_source_id else None
            ds_name = ds_map.get(ds_id) if ds_id else None

            # Fallback: if name is missing but we have an ID, show "Dataset {id}"
            if not ds_name and ds_id:
                ds_name = f"Dataset {ds_id}"

            jobs.append(BasicTrainingManager.map_training_job_to_info(job, ds_name))

        return jobs
