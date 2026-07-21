"""
Job Service
-----------
Centralized service for retrieving job entities (TrainingJob) from the
database. TrainingJob is discriminated by `run_mode` ("fixed" | "tuned"),
so a single query by id is always unambiguous.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from backend.database.models import TrainingJob


class JobService:
    """Service for managing and retrieving Job entities."""

    @staticmethod
    def get_job_by_id_sync(session: Session, job_id: str) -> TrainingJob | None:
        """Retrieves a job by ID (Synchronous)."""
        return session.query(TrainingJob).filter(TrainingJob.id == job_id).first()

    @staticmethod
    async def get_job_by_id(session: AsyncSession, job_id: str) -> TrainingJob | None:
        """
        Retrieves a job by ID.

        Args:
            session: The async database session.
            job_id: The unique identifier of the job.

        Returns:
            The TrainingJob entity if found, else None.
        """
        stmt = select(TrainingJob).where(TrainingJob.id == job_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
