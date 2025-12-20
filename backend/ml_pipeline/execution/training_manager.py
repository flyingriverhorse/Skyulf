import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from typing import cast as t_cast

from sqlalchemy import String, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.database.models import DataSource, TrainingJob
from backend.ml_pipeline.execution.graph_utils import extract_job_details
from backend.ml_pipeline.execution.schemas import JobInfo, JobStatus
from backend.ml_pipeline.model_registry.service import ModelRegistryService


class TrainingManager:
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
            version=version,
            model_type=model_type_val,
            graph=graph,
            started_at=datetime.now(),
        )

        session.add(job)
        await session.commit()
        return job_id

    @staticmethod
    def map_training_job_to_info(
        job: TrainingJob, dataset_name: Optional[str]
    ) -> JobInfo:
        # Extract details from graph
        (
            hyperparameters,
            target_column,
            dropped_columns,
        ) = extract_job_details(job.graph, job.node_id)

        # Also check job metrics for runtime dropped columns (e.g. from Feature Selection)
        if (
            job.metrics
            and isinstance(job.metrics, dict)
            and "dropped_columns" in job.metrics
        ):
            metrics_dropped = job.metrics["dropped_columns"]
            if isinstance(metrics_dropped, list):
                dropped_columns.extend(metrics_dropped)

        # Deduplicate
        dropped_columns = list(set(dropped_columns))

        # Fallback for hyperparameters if not found in graph (though extract_job_details should find it)
        if not hyperparameters:
            hyperparameters = job.hyperparameters

        return JobInfo(
            job_id=t_cast(str, job.id),
            pipeline_id=t_cast(str, job.pipeline_id),
            node_id=t_cast(str, job.node_id),
            dataset_id=t_cast(Optional[str], job.dataset_source_id),
            dataset_name=dataset_name,
            job_type="training",
            status=JobStatus(job.status),
            start_time=t_cast(Optional[datetime], job.started_at),
            end_time=t_cast(Optional[datetime], job.finished_at),
            error=t_cast(Optional[str], job.error_message),
            result={"metrics": job.metrics},
            model_type=t_cast(str, job.model_type),
            hyperparameters=t_cast(Dict[str, Any], hyperparameters),
            created_at=t_cast(datetime, job.created_at),
            metrics=t_cast(Optional[Dict[str, Any]], job.metrics),
            version=t_cast(Optional[int], job.version),
            target_column=target_column,
            dropped_columns=dropped_columns,
            logs=t_cast(Optional[List[str]], job.logs),
        )

    @staticmethod
    async def cancel_training_job(session: AsyncSession, job_id: str) -> bool:
        """Cancels a training job if it is running or queued."""
        stmt = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if job and job.status in [JobStatus.QUEUED.value, JobStatus.RUNNING.value]:
            job.status = JobStatus.CANCELLED.value
            job.error_message = "Job cancelled by user."
            job.finished_at = datetime.now()
            await session.commit()
            return True
        return False

    @staticmethod
    def _update_training_result(job: TrainingJob, result: Dict[str, Any]):
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
        job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return False

        if status:
            job.status = status.value
        if error:
            job.error_message = error

        if logs:
            current_logs = job.logs or []
            job.logs = current_logs + logs

        if result:
            TrainingManager._update_training_result(job, result)

        if status in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.SUCCEEDED,
        ]:
            job.finished_at = datetime.now()

        session.commit()
        return True

    @staticmethod
    async def get_training_job(
        session: AsyncSession, job_id: str
    ) -> Optional[JobInfo]:
        """Retrieves a training job by ID."""
        stmt = (
            select(TrainingJob, DataSource.name)
            .outerjoin(
                DataSource, TrainingJob.dataset_source_id == DataSource.source_id
            )
            .where(TrainingJob.id == job_id)
        )
        result = await session.execute(stmt)
        row = result.first()

        if row:
            job, dataset_name = row
            return TrainingManager.map_training_job_to_info(job, dataset_name)
        return None

    @staticmethod
    async def list_training_jobs(
        session: AsyncSession,
        limit: int = 50,
        skip: int = 0,
    ) -> List[JobInfo]:
        """Lists recent training jobs (Async)."""
        result_train = await session.execute(
            select(TrainingJob, DataSource.name)
            .outerjoin(
                DataSource,
                or_(
                    TrainingJob.dataset_source_id == DataSource.source_id,
                    TrainingJob.dataset_source_id == cast(DataSource.id, String),
                ),
            )
            .where(TrainingJob.model_type != "preview")
            .order_by(TrainingJob.started_at.desc())
            .limit(limit)
            .offset(skip)
        )
        train_rows = result_train.all()

        return [
            TrainingManager.map_training_job_to_info(j, d_name)
            for j, d_name in train_rows
        ]
