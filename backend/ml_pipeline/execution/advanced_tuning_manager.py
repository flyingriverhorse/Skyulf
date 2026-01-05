import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from typing import cast as type_cast

from sqlalchemy import String, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.database.models import DataSource, AdvancedTuningJob
from backend.ml_pipeline.execution.graph_utils import (
    determine_search_strategy,
    extract_job_details,
)
from backend.ml_pipeline.execution.schemas import JobInfo, JobStatus
from backend.ml_pipeline.model_registry.service import ModelRegistryService
from backend.ml_pipeline.execution.utils import resolve_dataset_name, get_dataset_map


class AdvancedTuningManager:
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

        job = AdvancedTuningJob(
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
        job: AdvancedTuningJob, dataset_name: Optional[str]
    ) -> JobInfo:
        # Extract details from graph
        (
            _,
            target_column,
            dropped_columns,
        ) = extract_job_details(type_cast(Dict[str, Any], job.graph), type_cast(str, job.node_id))

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
            job_id=type_cast(str, job.id),
            pipeline_id=type_cast(str, job.pipeline_id),
            node_id=type_cast(str, job.node_id),
            dataset_id=type_cast(Optional[str], job.dataset_source_id),
            dataset_name=dataset_name,
            job_type="advanced_tuning",
            status=JobStatus(job.status),
            start_time=type_cast(Optional[datetime], job.started_at),
            end_time=type_cast(Optional[datetime], job.finished_at),
            error=type_cast(Optional[str], job.error_message),
            result={
                "best_params": job.best_params,
                "best_score": job.best_score,
                "metrics": metrics,
            },
            # Ensure metrics are in result too
            model_type=type_cast(str, job.model_type),
            hyperparameters=type_cast(Dict[str, Any], hyperparameters),
            created_at=type_cast(datetime, job.created_at),
            metrics=type_cast(Optional[Dict[str, Any]], metrics),
            search_strategy=type_cast(Optional[str], job.search_strategy),
            version=type_cast(Optional[int], job.run_number),
            target_column=target_column,
            dropped_columns=dropped_columns,
            logs=type_cast(Optional[List[str]], job.logs),
            graph=type_cast(Dict[str, Any], job.graph),
        )

    @staticmethod
    async def cancel_tuning_job(session: AsyncSession, job_id: str) -> bool:
        """Cancels a tuning job if it is running or queued."""
        stmt = select(AdvancedTuningJob).where(
            AdvancedTuningJob.id == job_id
        )
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if job and job.status in [JobStatus.QUEUED.value, JobStatus.RUNNING.value]:
            job.status = JobStatus.CANCELLED.value  # type: ignore
            job.error_message = "Job cancelled by user."  # type: ignore
            job.finished_at = datetime.now()  # type: ignore
            await session.commit()
            return True
        return False

    @staticmethod
    def _update_tuning_result(job: AdvancedTuningJob, result: Dict[str, Any]):
        if "best_params" in result:
            job.best_params = result["best_params"]
        if "best_score" in result:
            job.best_score = result["best_score"]
        if "artifact_uri" in result:
            job.artifact_uri = result["artifact_uri"]

        job.metrics = result  # type: ignore

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
            session.query(AdvancedTuningJob)
            .filter(AdvancedTuningJob.id == job_id)
            .first()
        )
        if not job:
            return False

        if status:
            job.status = status.value  # type: ignore
        if error:
            job.error_message = error  # type: ignore

        if logs:
            current_logs: List[str] = job.logs or []  # type: ignore
            job.logs = current_logs + logs  # type: ignore

        if result:
            AdvancedTuningManager._update_tuning_result(job, result)

        if status in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.SUCCEEDED,
        ]:
            job.finished_at = datetime.now()  # type: ignore

        session.commit()
        return True

    @staticmethod
    async def get_tuning_job(
        session: AsyncSession, job_id: str
    ) -> Optional[JobInfo]:
        """Retrieves a tuning job by ID."""
        stmt = select(AdvancedTuningJob).where(
            AdvancedTuningJob.id == job_id
        )
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if job:
            dataset_name = await resolve_dataset_name(session, job.dataset_source_id)
            return AdvancedTuningManager.map_tuning_job_to_info(job, dataset_name)
        return None

    @staticmethod
    async def list_tuning_jobs(
        session: AsyncSession,
        limit: int = 50,
        skip: int = 0,
    ) -> List[JobInfo]:
        """Lists recent tuning jobs (Async)."""
        # 1. Fetch all DataSources for robust name resolution
        ds_map = await get_dataset_map(session)

        # 2. Fetch Jobs
        result_tune = await session.execute(
            select(AdvancedTuningJob)
            .order_by(AdvancedTuningJob.started_at.desc())
            .limit(limit)
            .offset(skip)
        )
        tune_rows = result_tune.scalars().all()

        # 3. Map to Info
        jobs = []
        for job in tune_rows:
            ds_id = str(job.dataset_source_id) if job.dataset_source_id else None
            ds_name = ds_map.get(ds_id) if ds_id else None
            
            # Fallback
            if not ds_name and ds_id:
                 ds_name = f"Dataset {ds_id}"

            jobs.append(AdvancedTuningManager.map_tuning_job_to_info(job, ds_name))
            
        return jobs

    @staticmethod
    async def get_latest_tuning_job_for_node(
        session: AsyncSession, node_id: str
    ) -> Optional[JobInfo]:
        result = await session.execute(
            select(AdvancedTuningJob)
            .where(AdvancedTuningJob.node_id == node_id)
            .where(AdvancedTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(AdvancedTuningJob.finished_at.desc())
            .limit(1)
        )
        job = result.scalars().first()

        if job:
            return AdvancedTuningManager.map_tuning_job_to_info(job, None)
        return None

    @staticmethod
    async def get_best_tuning_job_for_model(
        session: AsyncSession, model_type: str
    ) -> Optional[JobInfo]:
        result = await session.execute(
            select(AdvancedTuningJob)
            .where(AdvancedTuningJob.model_type == model_type)
            .where(AdvancedTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(AdvancedTuningJob.finished_at.desc())
            .limit(1)
        )
        job = result.scalars().first()

        if job:
            return AdvancedTuningManager.map_tuning_job_to_info(job, None)
        return None

    @staticmethod
    async def get_tuning_jobs_for_model(
        session: AsyncSession, model_type: str, limit: int = 20
    ) -> List[JobInfo]:
        result = await session.execute(
            select(AdvancedTuningJob)
            .where(AdvancedTuningJob.model_type == model_type)
            .where(AdvancedTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(AdvancedTuningJob.finished_at.desc())
            .limit(limit)
        )
        jobs = result.scalars().all()

        return [
            AdvancedTuningManager.map_tuning_job_to_info(job, None)
            for job in jobs
        ]
