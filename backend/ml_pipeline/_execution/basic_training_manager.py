import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, cast as t_cast, cast as type_cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.database.models import BasicTrainingJob
from backend.ml_pipeline.constants import StepType
from backend.ml_pipeline._execution.graph_utils import extract_job_details
from backend.ml_pipeline._execution.schemas import JobInfo, JobStatus
from backend.ml_pipeline.model_registry.service import ModelRegistryService


from backend.ml_pipeline._execution.utils import (
    resolve_dataset_name,
    get_dataset_map,
    parse_branch_info,
)


class BasicTrainingManager:
    @staticmethod
    async def create_training_job(
        session: AsyncSession,
        pipeline_id: str,
        node_id: str,
        dataset_id: str = "unknown",
        user_id: Optional[int] = None,
        model_type: str = "unknown",
        graph: Optional[Dict[str, Any]] = None,
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

        job = BasicTrainingJob(
            id=job_id,
            pipeline_id=pipeline_id,
            node_id=node_id,
            dataset_source_id=dataset_id,
            user_id=user_id,
            status=JobStatus.QUEUED.value,
            version=version,
            model_type=model_type_val,
            graph=graph,
            job_metadata={"branch_index": branch_index},
            started_at=datetime.now(),
        )

        session.add(job)
        await session.commit()
        return job_id

    @staticmethod
    def map_training_job_to_info(job: BasicTrainingJob, dataset_name: Optional[str]) -> JobInfo:
        # Extract details from graph
        (
            hyperparameters,
            target_column,
            dropped_columns,
        ) = extract_job_details(type_cast(Dict[str, Any], job.graph), job.node_id)

        # Also check job metrics for runtime dropped columns (e.g. from Feature Selection)
        if job.metrics and isinstance(job.metrics, dict) and "dropped_columns" in job.metrics:
            metrics_dropped = job.metrics["dropped_columns"]
            if isinstance(metrics_dropped, list):
                dropped_columns.extend(metrics_dropped)

        # Deduplicate
        dropped_columns = list(set(dropped_columns))

        # Fallback for hyperparameters if not found in graph (though extract_job_details should find it)
        if not hyperparameters:
            hyperparameters = type_cast(Optional[Dict[str, Any]], job.hyperparameters)

        return JobInfo(
            job_id=job.id,
            pipeline_id=job.pipeline_id,
            node_id=job.node_id,
            dataset_id=t_cast(Optional[str], job.dataset_source_id),
            dataset_name=dataset_name,
            job_type=StepType.BASIC_TRAINING.value,
            status=JobStatus(job.status),
            start_time=job.started_at,
            end_time=job.finished_at,
            error=job.error_message,
            result={"metrics": job.metrics},
            model_type=job.model_type,
            hyperparameters=t_cast(Dict[str, Any], hyperparameters),
            created_at=t_cast(datetime, job.created_at),
            metrics=t_cast(Optional[Dict[str, Any]], job.metrics),
            version=t_cast(Optional[int], job.version),
            target_column=target_column,
            dropped_columns=dropped_columns,
            logs=t_cast(Optional[List[str]], job.logs),
            graph=type_cast(Dict[str, Any], job.graph),
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
        stmt = select(BasicTrainingJob).where(BasicTrainingJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if job and job.status in [JobStatus.QUEUED.value, JobStatus.RUNNING.value]:
            job.status = JobStatus.CANCELLED.value
            job.error_message = "Job cancelled by user."
            job.finished_at = datetime.now()
            # Revoke the Celery task if we recorded its id at submission time.
            meta = (job.job_metadata or {}) if isinstance(job.job_metadata, dict) else {}
            task_id = meta.get("celery_task_id")
            if task_id:
                try:
                    from backend.celery_app import celery_app

                    celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
                except Exception:
                    # Best-effort: never let revoke errors block the user-visible
                    # cancel. The status guard in update_status_sync still keeps
                    # the row at CANCELLED even if the worker writes back.
                    pass
            await session.commit()
            return True
        return False

    @staticmethod
    def _update_training_result(job: BasicTrainingJob, result: Dict[str, Any]):
        if "metrics" in result:
            job.metrics = result["metrics"]
        if "artifact_uri" in result:
            job.artifact_uri = result["artifact_uri"]
        if "hyperparameters" in result:
            job.hyperparameters = result["hyperparameters"]

    @staticmethod
    def update_status_sync(
        session: Session,
        job_id: str,
        status: Optional[JobStatus] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        logs: Optional[List[str]] = None,
    ) -> bool:
        """Updates training job status (Sync). Returns True if job found and updated."""
        job = session.query(BasicTrainingJob).filter(BasicTrainingJob.id == job_id).first()
        if not job:
            return False

        # Guard: once a job is CANCELLED by the user, the worker may still be
        # mid-`fit` and try to flip it back to RUNNING/COMPLETED on its next
        # progress write. Refuse those overwrites so the user-visible state
        # stays accurate; logs are still appended so cancellation traces are
        # preserved for debugging.
        if job.status == JobStatus.CANCELLED.value:
            if logs:
                current_logs: List[str] = job.logs or []
                job.logs = current_logs + logs
                session.commit()
            return True

        if status:
            job.status = status.value
        if error:
            job.error_message = error

        if logs:
            current_logs: List[str] = job.logs or []
            job.logs = current_logs + logs

        if result:
            BasicTrainingManager._update_training_result(job, result)

        if status in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.SUCCEEDED,
        ]:
            job.finished_at = datetime.now()

        session.commit()
        return True

    @staticmethod
    async def get_training_job(session: AsyncSession, job_id: str) -> Optional[JobInfo]:
        """Retrieves a training job by ID."""
        # 1. Fetch Job
        stmt = select(BasicTrainingJob).where(BasicTrainingJob.id == job_id)
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
        limit: int = 50,
        skip: int = 0,
    ) -> List[JobInfo]:
        """Lists recent training jobs (Async)."""
        # 1. Fetch all DataSources for robust name resolution
        ds_map = await get_dataset_map(session)

        # 2. Fetch Jobs
        result_train = await session.execute(
            select(BasicTrainingJob)
            .where(BasicTrainingJob.model_type != "preview")
            .order_by(BasicTrainingJob.started_at.desc())
            .limit(limit)
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
