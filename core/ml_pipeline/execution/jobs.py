"""
Job Management for V2 Pipeline.
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import uuid
from pydantic import BaseModel

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class JobInfo(BaseModel):
    job_id: str
    pipeline_id: str
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class JobManager:
    _jobs: Dict[str, JobInfo] = {}

    @classmethod
    def create_job(cls, pipeline_id: str) -> str:
        job_id = str(uuid.uuid4())
        cls._jobs[job_id] = JobInfo(
            job_id=job_id,
            pipeline_id=pipeline_id,
            status=JobStatus.QUEUED,
            start_time=datetime.now()
        )
        return job_id

    @classmethod
    def get_job(cls, job_id: str) -> Optional[JobInfo]:
        return cls._jobs.get(job_id)

    @classmethod
    def update_status(cls, job_id: str, status: JobStatus, error: Optional[str] = None, result: Optional[Dict[str, Any]] = None):
        if job_id in cls._jobs:
            job = cls._jobs[job_id]
            job.status = status
            if error:
                job.error = error
            if result:
                job.result = result
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                job.end_time = datetime.now()

    @classmethod
    def list_jobs(cls) -> List[JobInfo]:
        return list(cls._jobs.values())
