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

from backend.database.models import AdvancedTuningJob, BasicTrainingJob


class JobService:
    """Service for managing and retrieving Job entities."""

    @staticmethod
    def get_job_by_id_sync(
        session: Session, job_id: str
    ) -> Optional[Union[BasicTrainingJob, AdvancedTuningJob]]:
        """
        Retrieves a job by ID (Synchronous), checking both BasicTrainingJob and AdvancedTuningJob tables.
        """
        # 1. Try BasicTrainingJob
        job = session.query(BasicTrainingJob).filter(BasicTrainingJob.id == job_id).first()
        if job:
            return job

        # 2. Try AdvancedTuningJob
        job = (
            session.query(AdvancedTuningJob)
            .filter(AdvancedTuningJob.id == job_id)
            .first()
        )
        return job

    @staticmethod
    async def get_job_by_id(
        session: AsyncSession, job_id: str
    ) -> Optional[Union[BasicTrainingJob, AdvancedTuningJob]]:
        """
        Retrieves a job by ID, checking both BasicTrainingJob and AdvancedTuningJob tables.

        Args:
            session: The async database session.
            job_id: The unique identifier of the job.

        Returns:
            The job entity (BasicTrainingJob or AdvancedTuningJob) if found, else None.
        """
        # 1. Try BasicTrainingJob
        stmt_train = select(BasicTrainingJob).where(BasicTrainingJob.id == job_id)
        result_train = await session.execute(stmt_train)
        job = result_train.scalar_one_or_none()
        if job:
            return job

        # 2. Try AdvancedTuningJob
        stmt_tune = select(AdvancedTuningJob).where(
            AdvancedTuningJob.id == job_id
        )
        result_tune = await session.execute(stmt_tune)
        job = result_tune.scalar_one_or_none()
        
        return job
