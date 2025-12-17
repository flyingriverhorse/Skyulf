"""
Job Management for V2 Pipeline.
Handles persistence of Training and Tuning jobs to the database.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.ml_pipeline.execution.schemas import JobInfo, JobStatus
from backend.ml_pipeline.execution.training_manager import TrainingManager
from backend.ml_pipeline.execution.tuning_manager import TuningJobManager


class JobManager:
    """
    Facade for managing training and tuning jobs.
    Delegates to TrainingManager and TuningJobManager.
    """

    @staticmethod
    async def create_job(
        session: AsyncSession,
        pipeline_id: str,
        node_id: str,
        job_type: Literal["training", "tuning", "preview"],
        dataset_id: str = "unknown",
        user_id: Optional[int] = None,
        model_type: str = "unknown",
        graph: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Creates a new job in the database (Async)."""
        if job_type == "training":
            return await TrainingManager.create_training_job(
                session,
                pipeline_id,
                node_id,
                dataset_id,
                user_id,
                model_type,
                graph,
            )
        elif job_type == "tuning":
            return await TuningJobManager.create_tuning_job(
                session,
                pipeline_id,
                node_id,
                dataset_id,
                user_id,
                model_type,
                graph,
            )
        elif job_type == "preview":
            return await TrainingManager.create_training_job(
                session,
                pipeline_id,
                node_id,
                dataset_id,
                user_id,
                model_type,
                graph,
                is_preview=True,
            )
        else:
            raise ValueError(f"Unknown job_type: {job_type}")

    @staticmethod
    async def cancel_job(session: AsyncSession, job_id: str) -> bool:
        """Cancels a job if it is running or queued."""
        # Try TrainingJob first
        if await TrainingManager.cancel_training_job(session, job_id):
            return True
        # Then TuningJob
        return await TuningJobManager.cancel_tuning_job(session, job_id)

    @staticmethod
    def update_status_sync(
        session: Session,
        job_id: str,
        status: Optional[JobStatus] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        logs: Optional[List[str]] = None,
    ):
        """Updates job status (Sync - for Background Tasks)."""
        # Try TrainingJob first
        if TrainingManager.update_status_sync(
            session, job_id, status, error, result, logs
        ):
            return

        # Then TuningJob
        TuningJobManager.update_status_sync(
            session, job_id, status, error, result, logs
        )

    @staticmethod
    async def get_job(session: AsyncSession, job_id: str) -> Optional[JobInfo]:
        """Retrieves job info (Async)."""
        # Try TrainingJob
        job = await TrainingManager.get_training_job(session, job_id)
        if job:
            return job

        # Then TuningJob
        return await TuningJobManager.get_tuning_job(session, job_id)

    @staticmethod
    async def list_jobs(
        session: AsyncSession,
        limit: int = 50,
        skip: int = 0,
        job_type: Optional[Literal["training", "tuning"]] = None,
    ) -> List[JobInfo]:
        """Lists recent jobs (Async)."""
        jobs = []

        if job_type == "training":
            jobs = await TrainingManager.list_training_jobs(session, limit, skip)
        elif job_type == "tuning":
            jobs = await TuningJobManager.list_tuning_jobs(session, limit, skip)
        else:
            # Combine both
            train_jobs = await TrainingManager.list_training_jobs(
                session, limit + skip, 0
            )
            tune_jobs = await TuningJobManager.list_tuning_jobs(
                session, limit + skip, 0
            )

            all_jobs = train_jobs + tune_jobs
            # Sort by start_time desc
            all_jobs.sort(key=lambda x: x.start_time or datetime.min, reverse=True)

            # Apply skip and limit
            jobs = all_jobs[skip : skip + limit]

        return jobs

    @staticmethod
    async def get_latest_tuning_job_for_node(
        session: AsyncSession, node_id: str
    ) -> Optional[JobInfo]:
        return await TuningJobManager.get_latest_tuning_job_for_node(session, node_id)

    @staticmethod
    async def get_best_tuning_job_for_model(
        session: AsyncSession, model_type: str
    ) -> Optional[JobInfo]:
        return await TuningJobManager.get_best_tuning_job_for_model(session, model_type)

    @staticmethod
    async def get_tuning_jobs_for_model(
        session: AsyncSession, model_type: str, limit: int = 20
    ) -> List[JobInfo]:
        return await TuningJobManager.get_tuning_jobs_for_model(
            session, model_type, limit
        )
