"""
Job Service
-----------
Centralized service for retrieving job entities (TrainingJob, HyperparameterTuningJob)
from the database. This eliminates the pattern of "Try TrainingJob, then TuningJob"
scattered across the codebase.
"""

from typing import Optional, Union

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.database.models import HyperparameterTuningJob, TrainingJob


class JobService:
    """Service for managing and retrieving Job entities."""

    @staticmethod
    def get_job_by_id_sync(
        session: Session, job_id: str
    ) -> Optional[Union[TrainingJob, HyperparameterTuningJob]]:
        """
        Retrieves a job by ID (Synchronous), checking both TrainingJob and HyperparameterTuningJob tables.
        """
        # 1. Try TrainingJob
        job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            return job

        # 2. Try HyperparameterTuningJob
        job = (
            session.query(HyperparameterTuningJob)
            .filter(HyperparameterTuningJob.id == job_id)
            .first()
        )
        return job

    @staticmethod
    async def get_job_by_id(
        session: AsyncSession, job_id: str
    ) -> Optional[Union[TrainingJob, HyperparameterTuningJob]]:
        """
        Retrieves a job by ID, checking both TrainingJob and HyperparameterTuningJob tables.

        Args:
            session: The async database session.
            job_id: The unique identifier of the job.

        Returns:
            The job entity (TrainingJob or HyperparameterTuningJob) if found, else None.
        """
        # 1. Try TrainingJob
        stmt_train = select(TrainingJob).where(TrainingJob.id == job_id)
        result_train = await session.execute(stmt_train)
        job = result_train.scalar_one_or_none()
        if job:
            return job

        # 2. Try HyperparameterTuningJob
        stmt_tune = select(HyperparameterTuningJob).where(
            HyperparameterTuningJob.id == job_id
        )
        result_tune = await session.execute(stmt_tune)
        job = result_tune.scalar_one_or_none()
        
        return job
