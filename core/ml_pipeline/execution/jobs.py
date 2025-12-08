"""
Job Management for V2 Pipeline.
Handles persistence of Training and Tuning jobs to the database.
"""

from typing import Dict, Any, Optional, List, Union, Literal
from enum import Enum
from datetime import datetime
import uuid
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy import select, update
from core.database.models import TrainingJob, HyperparameterTuningJob, DataSource

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

class JobInfo(BaseModel):
    job_id: str
    pipeline_id: str
    node_id: str
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    job_type: Literal["training", "tuning"]
    status: JobStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class JobManager:
    
    @staticmethod
    async def create_job(
        session: AsyncSession, 
        pipeline_id: str, 
        node_id: str, 
        job_type: Literal["training", "tuning"],
        dataset_id: str = "unknown",
        user_id: Optional[int] = None,
        model_type: str = "unknown",
        graph: Dict[str, Any] = None
    ) -> str:
        """Creates a new job in the database (Async)."""
        job_id = str(uuid.uuid4())
        graph = graph or {}
        
        if job_type == "training":
            job = TrainingJob(
                id=job_id,
                pipeline_id=pipeline_id,
                node_id=node_id,
                dataset_source_id=dataset_id,
                user_id=user_id,
                status=JobStatus.QUEUED.value,
                model_type=model_type, 
                graph=graph,
                started_at=datetime.now()
            )
        else:
            job = HyperparameterTuningJob(
                id=job_id,
                pipeline_id=pipeline_id,
                node_id=node_id,
                dataset_source_id=dataset_id,
                user_id=user_id,
                status=JobStatus.QUEUED.value,
                model_type=model_type,
                graph=graph,
                started_at=datetime.now()
            )
            
        session.add(job)
        await session.commit()
        return job_id

    @staticmethod
    def update_status_sync(
        session: Session, 
        job_id: str, 
        status: JobStatus, 
        error: Optional[str] = None, 
        result: Optional[Dict[str, Any]] = None
    ):
        """Updates job status (Sync - for Background Tasks)."""
        # Try finding in TrainingJob first
        job = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            job = session.query(HyperparameterTuningJob).filter(HyperparameterTuningJob.id == job_id).first()
            
        if job:
            job.status = status.value
            if error:
                job.error_message = error
                # Also update specific error fields if they exist
            
            if result:
                # Map result fields to model columns
                if isinstance(job, TrainingJob):
                    if "metrics" in result:
                        job.metrics = result["metrics"]
                    if "artifact_uri" in result:
                        job.artifact_uri = result["artifact_uri"]
                elif isinstance(job, HyperparameterTuningJob):
                    if "best_params" in result:
                        job.best_params = result["best_params"]
                    if "best_score" in result:
                        job.best_score = result["best_score"]
            
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                job.finished_at = datetime.now()
                # job.progress = 100
            
            session.commit()

    @staticmethod
    async def get_job(session: AsyncSession, job_id: str) -> Optional[JobInfo]:
        """Retrieves job info (Async)."""
        # Try TrainingJob
        stmt = select(TrainingJob, DataSource.name).outerjoin(DataSource, TrainingJob.dataset_source_id == DataSource.source_id).where(TrainingJob.id == job_id)
        result = await session.execute(stmt)
        row = result.first()
        
        job = None
        dataset_name = None
        job_type = "training"
        
        if row:
            job, dataset_name = row
        else:
            stmt = select(HyperparameterTuningJob, DataSource.name).outerjoin(DataSource, HyperparameterTuningJob.dataset_source_id == DataSource.source_id).where(HyperparameterTuningJob.id == job_id)
            result = await session.execute(stmt)
            row = result.first()
            if row:
                job, dataset_name = row
                job_type = "tuning"
            
        if job:
            return JobInfo(
                job_id=job.id,
                pipeline_id=job.pipeline_id,
                node_id=job.node_id,
                dataset_id=job.dataset_source_id,
                dataset_name=dataset_name,
                job_type=job_type,
                status=JobStatus(job.status),
                start_time=job.started_at,
                end_time=job.finished_at,
                error=job.error_message,
                result={"metrics": job.metrics} if job_type == "training" else {"best_params": job.best_params}
            )
        return None

    @staticmethod
    async def list_jobs(session: AsyncSession, limit: int = 50) -> List[JobInfo]:
        """Lists recent jobs (Async)."""
        # Fetch both and merge? Or just return separate lists?
        # For now, let's fetch TrainingJobs
        result_train = await session.execute(
            select(TrainingJob, DataSource.name)
            .outerjoin(DataSource, TrainingJob.dataset_source_id == DataSource.source_id)
            .order_by(TrainingJob.started_at.desc())
            .limit(limit)
        )
        train_rows = result_train.all()
        
        result_tune = await session.execute(
            select(HyperparameterTuningJob, DataSource.name)
            .outerjoin(DataSource, HyperparameterTuningJob.dataset_source_id == DataSource.source_id)
            .order_by(HyperparameterTuningJob.started_at.desc())
            .limit(limit)
        )
        tune_rows = result_tune.all()
        
        combined = []
        for j, d_name in train_rows:
            combined.append(JobInfo(
                job_id=j.id,
                pipeline_id=j.pipeline_id,
                node_id=j.node_id,
                dataset_id=j.dataset_source_id,
                dataset_name=d_name,
                job_type="training",
                status=JobStatus(j.status),
                start_time=j.started_at,
                end_time=j.finished_at,
                error=j.error_message,
                result={"metrics": j.metrics}
            ))
        for j, d_name in tune_rows:
            combined.append(JobInfo(
                job_id=j.id,
                pipeline_id=j.pipeline_id,
                node_id=j.node_id,
                dataset_id=j.dataset_source_id,
                dataset_name=d_name,
                job_type="tuning",
                status=JobStatus(j.status),
                start_time=j.started_at,
                end_time=j.finished_at,
                error=j.error_message,
                result={"best_params": j.best_params}
            ))
            
        # Sort by start time
        combined.sort(key=lambda x: x.start_time or datetime.min, reverse=True)
        return combined[:limit]

    @staticmethod
    async def get_latest_tuning_job_for_node(session: AsyncSession, node_id: str) -> Optional[JobInfo]:
        result = await session.execute(
            select(HyperparameterTuningJob)
            .where(HyperparameterTuningJob.node_id == node_id)
            .where(HyperparameterTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(HyperparameterTuningJob.finished_at.desc())
            .limit(1)
        )
        job = result.scalars().first()
        
        if job:
            return JobInfo(
                job_id=job.id,
                pipeline_id=job.pipeline_id,
                node_id=job.node_id,
                dataset_id=job.dataset_source_id,
                job_type="tuning",
                status=JobStatus(job.status),
                start_time=job.started_at,
                end_time=job.finished_at,
                error=job.error_message,
                result={"best_params": job.best_params, "best_score": job.best_score}
            )
        return None

    @staticmethod
    async def get_best_tuning_job_for_model(session: AsyncSession, model_type: str) -> Optional[JobInfo]:
        """
        Finds the best completed tuning job for a given model type.
        Orders by best_score descending (assuming higher is better for now, or just latest).
        Ideally we should check the metric direction, but for now let's take the latest successful one.
        """
        result = await session.execute(
            select(HyperparameterTuningJob)
            .where(HyperparameterTuningJob.model_type == model_type)
            .where(HyperparameterTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(HyperparameterTuningJob.finished_at.desc()) # Get latest
            .limit(1)
        )
        job = result.scalars().first()
        
        if job:
            return JobInfo(
                job_id=job.id,
                pipeline_id=job.pipeline_id,
                node_id=job.node_id,
                dataset_id=job.dataset_source_id,
                job_type="tuning",
                status=JobStatus(job.status),
                start_time=job.started_at,
                end_time=job.finished_at,
                error=job.error_message,
                result={"best_params": job.best_params, "best_score": job.best_score}
            )
        return None

    @staticmethod
    async def get_tuning_jobs_for_model(session: AsyncSession, model_type: str, limit: int = 20) -> List[JobInfo]:
        """
        Returns a list of completed tuning jobs for a specific model type.
        """
        result = await session.execute(
            select(HyperparameterTuningJob)
            .where(HyperparameterTuningJob.model_type == model_type)
            .where(HyperparameterTuningJob.status == JobStatus.COMPLETED.value)
            .order_by(HyperparameterTuningJob.finished_at.desc())
            .limit(limit)
        )
        jobs = result.scalars().all()
        
        return [
            JobInfo(
                job_id=job.id,
                pipeline_id=job.pipeline_id,
                node_id=job.node_id,
                dataset_id=job.dataset_source_id,
                job_type="tuning",
                status=JobStatus(job.status),
                start_time=job.started_at,
                end_time=job.finished_at,
                error=job.error_message,
                result={"best_params": job.best_params, "best_score": job.best_score}
            )
            for job in jobs
        ]
