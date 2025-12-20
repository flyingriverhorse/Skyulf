import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from typing import cast as t_cast

from sqlalchemy import String, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.database.models import DataSource, HyperparameterTuningJob
from backend.ml_pipeline.execution.graph_utils import (
    determine_search_strategy,
    extract_job_details,
)
from backend.ml_pipeline.execution.schemas import JobInfo, JobStatus
from backend.ml_pipeline.model_registry.service import ModelRegistryService


class TuningJobManager:
    @staticmethod
    async def create_tuning_job(
        session: AsyncSession,
        pipeline_id: str,
        node_id: str,
        dataset_id: str = "unknown",
        user_id: Optional[int] = None,
        model_type: str = "unknown",
        graph: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Creates a new tuning job in the database (Async)."""
        job_id = str(uuid.uuid4())
        graph = graph or {}

        next_version = await ModelRegistryService.get_next_version(
            session, dataset_id, model_type, "tuning"
        )

        search_strategy = determine_search_strategy(graph, node_id)

        job = HyperparameterTuningJob(
            id=job_id,
            pipeline_id=pipeline_id,
            node_id=node_id,
            dataset_source_id=dataset_id,
            user_id=user_id,
            status=JobStatus.QUEUED.value,
            run_number=next_version,
            model_type=model_type,
            search_strategy=search_strategy,
            graph=graph,
            started_at=datetime.now(),
        )

        session.add(job)
        await session.commit()
        return job_id

    @staticmethod
    def map_tuning_job_to_info(
        job: HyperparameterTuningJob, dataset_name: Optional[str]
    ) -> JobInfo:
        # Extract details from graph
        (
            _,
            target_column,
            dropped_columns,
        ) = extract_job_details(job.graph, job.node_id)

        # Extract hyperparameters (search space)
        # For display in Experiments table, users prefer to see the BEST params found
        # if the job is completed. Otherwise, show the search space.
        if job.status == JobStatus.COMPLETED.value and job.best_params:
            hyperparameters = job.best_params
        else:
            hyperparameters = job.search_space

        # Use stored metrics if available, otherwise fallback to best_score
        metrics = (
            job.metrics
            if job.metrics
            else ({"score": job.best_score} if job.best_score else None)
        )

        return JobInfo(
            job_id=t_cast(str, job.id),
            pipeline_id=t_cast(str, job.pipeline_id),
            node_id=t_cast(str, job.node_id),
            dataset_id=t_cast(Optional[str], job.dataset_source_id),
            dataset_name=dataset_name,
            job_type="tuning",
            status=JobStatus(job.status),
            start_time=t_cast(Optional[datetime], job.started_at),
            end_time=t_cast(Optional[datetime], job.finished_at),
            error=t_cast(Optional[str], job.error_message),
            result={
                "best_params": job.best_params,
                "best_score": job.best_score,
                "metrics": metrics,
            },
            # Ensure metrics are in result too
            model_type=t_cast(str, job.model_type),
            hyperparameters=t_cast(Dict[str, Any], hyperparameters),
            created_at=t_cast(datetime, job.created_at),
            metrics=t_cast(Optional[Dict[str, Any]], metrics),
            search_strategy=t_cast(Optional[str], job.search_strategy),
            version=t_cast(Optional[int], job.run_number),
            target_column=target_column,
            dropped_columns=dropped_columns,
            logs=t_cast(Optional[List[str]], job.logs),
        )

    @staticmethod
    async def cancel_tuning_job(session: AsyncSession, job_id: str) -> bool:
        """Cancels a tuning job if it is running or queued."""
        stmt = select(HyperparameterTuningJob).where(
            HyperparameterTuningJob.id == job_id
        )
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
    def _update_tuning_result(job: HyperparameterTuningJob, result: Dict[str, Any]):
        if "best_params" in result:
            job.best_params = result["best_params"]
        if "best_score" in result:
            job.best_score = result["best_score"]
        if "artifact_uri" in result:
            job.artifact_uri = result["artifact_uri"]

        job.metrics = result

    @staticmethod
    def update_status_sync(
        session: Session,
        job_id: str,
        status: Optional[JobStatus] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        logs: Optional[List[str]] = None,
    ) -> bool:
        """Updates tuning job status (Sync). Returns True if job found and updated."""
        job = (
            session.query(HyperparameterTuningJob)
            .filter(HyperparameterTuningJob.id == job_id)
            .first()
        )
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
            TuningJobManager._update_tuning_result(job, result)

        if status in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.SUCCEEDED,
        ]:
            job.finished_at = datetime.now()

        session.commit()
        return True

    @staticmethod
    async def get_tuning_job(
        session: AsyncSession, job_id: str
    ) -> Optional[JobInfo]:
        """Retrieves a tuning job by ID."""
        stmt = (
            select(HyperparameterTuningJob, DataSource.name)
            .outerjoin(
                DataSource,
                HyperparameterTuningJob.dataset_source_id == DataSource.source_id,
            )
            .where(HyperparameterTuningJob.id == job_id)
        )
        result = await session.execute(stmt)
        row = result.first()

        if row:
            job, dataset_name = row
            return TuningJobManager.map_tuning_job_to_info(job, dataset_name)
        return None

    @staticmethod
    async def list_tuning_jobs(
        session: AsyncSession,
        limit: int = 50,
        skip: int = 0,
    ) -> List[JobInfo]:
        """Lists recent tuning jobs (Async)."""
        result_tune = await session.execute(
            select(HyperparameterTuningJob, DataSource.name)
            .outerjoin(
                DataSource,
                or_(
                    HyperparameterTuningJob.dataset_source_id == DataSource.source_id,
                    HyperparameterTuningJob.dataset_source_id
                    == cast(DataSource.id, String),
                ),
            )
            .order_by(HyperparameterTuningJob.started_at.desc())
            .limit(limit)
            .offset(skip)
        )
        tune_rows = result_tune.all()

        return [
            TuningJobManager.map_tuning_job_to_info(j, d_name)
            for j, d_name in tune_rows
        ]

    @staticmethod
    async def get_latest_tuning_job_for_node(
        session: AsyncSession, node_id: str
    ) -> Optional[JobInfo]:
        result = await session.execute(
            select(HyperparameterTuningJob)
            .where(HyperparameterTuningJob.node_id == node_id)
            .where(HyperparameterTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(HyperparameterTuningJob.finished_at.desc())
            .limit(1)
        )
        job = result.scalars().first()

        if job:
            return TuningJobManager.map_tuning_job_to_info(job, None)
        return None

    @staticmethod
    async def get_best_tuning_job_for_model(
        session: AsyncSession, model_type: str
    ) -> Optional[JobInfo]:
        result = await session.execute(
            select(HyperparameterTuningJob)
            .where(HyperparameterTuningJob.model_type == model_type)
            .where(HyperparameterTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(HyperparameterTuningJob.finished_at.desc())
            .limit(1)
        )
        job = result.scalars().first()

        if job:
            return TuningJobManager.map_tuning_job_to_info(job, None)
        return None

    @staticmethod
    async def get_tuning_jobs_for_model(
        session: AsyncSession, model_type: str, limit: int = 20
    ) -> List[JobInfo]:
        result = await session.execute(
            select(HyperparameterTuningJob)
            .where(HyperparameterTuningJob.model_type == model_type)
            .where(HyperparameterTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(HyperparameterTuningJob.finished_at.desc())
            .limit(limit)
        )
        jobs = result.scalars().all()

        return [
            TuningJobManager.map_tuning_job_to_info(job, None)
            for job in jobs
        ]