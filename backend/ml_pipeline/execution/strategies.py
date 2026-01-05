from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from datetime import datetime
from sqlalchemy.orm import Session

from backend.database.models import MLJob, BasicTrainingJob, AdvancedTuningJob
from backend.ml_pipeline.execution.schemas import PipelineExecutionResult
from backend.ml_pipeline.constants import StepType

class JobStrategy(ABC):
    """
    Abstract base class for job execution strategies.
    Encapsulates logic specific to different job types (Training, Tuning, etc.).
    """

    @abstractmethod
    def get_job_model(self) -> Type[MLJob]:
        """Returns the SQLAlchemy model class for this job type."""
        pass

    def get_job(self, session: Session, job_id: str) -> Optional[MLJob]:
        """Fetches the job from the database."""
        return session.query(self.get_job_model()).filter(self.get_job_model().id == job_id).first()

    @abstractmethod
    def get_initial_log(self, job: MLJob) -> str:
        """Returns the initial log message for the job."""
        pass

    def handle_success(self, job: MLJob, result: PipelineExecutionResult) -> None:
        """
        Updates the job with results from a successful pipeline execution.
        Base implementation handles common metrics.
        """
        # Extract metrics from the last node if available
        if result.node_results:
            last_node_id = list(result.node_results.keys())[-1]
            last_result = result.node_results[last_node_id]

            final_metrics = (
                last_result.metrics.copy() if last_result.metrics else {}
            )

            # Collect dropped columns from all nodes
            all_dropped_columns = []
            for node_res in result.node_results.values():
                if node_res.metrics and "dropped_columns" in node_res.metrics:
                    cols = node_res.metrics["dropped_columns"]
                    if isinstance(cols, list):
                        all_dropped_columns.extend(cols)

            if all_dropped_columns:
                all_dropped_columns = list(set(all_dropped_columns))
                final_metrics["dropped_columns"] = all_dropped_columns

            job.metrics = final_metrics

    def handle_failure(self, job: MLJob, error_msg: str) -> None:
        """Updates the job with failure information."""
        job.status = "failed"
        job.error_message = error_msg
        job.finished_at = datetime.now()


class BasicTrainingStrategy(JobStrategy):
    def get_job_model(self) -> Type[MLJob]:
        return BasicTrainingJob

    def get_initial_log(self, job: MLJob) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Cast to BasicTrainingJob to access specific fields if needed, 
        # though 'version' is on the model
        version = getattr(job, "version", "unknown")
        return f"[{timestamp}] Training Job Version: {version}"


class AdvancedTuningStrategy(JobStrategy):
    def get_job_model(self) -> Type[MLJob]:
        return AdvancedTuningJob

    def get_initial_log(self, job: MLJob) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        run_number = getattr(job, "run_number", "unknown")
        return f"[{timestamp}] Tuning Job Run: {run_number}"

    def handle_success(self, job: MLJob, result: PipelineExecutionResult) -> None:
        # Call base to set standard metrics
        super().handle_success(job, result)
        
        # Add tuning-specific fields
        if job.metrics:
            if "best_params" in job.metrics:
                job.best_params = job.metrics["best_params"]
            if "best_score" in job.metrics:
                job.best_score = job.metrics["best_score"]
            if "trials" in job.metrics:
                job.results = job.metrics["trials"]


class JobStrategyFactory:
    _strategies: Dict[str, JobStrategy] = {
        StepType.BASIC_TRAINING: BasicTrainingStrategy(),
        StepType.ADVANCED_TUNING: AdvancedTuningStrategy(),
        # Add more strategies here as needed
    }

    @classmethod
    def get_strategy(cls, job_type: str) -> JobStrategy:
        """Returns the strategy for the given job type."""
        if job_type in cls._strategies:
            return cls._strategies[job_type]
        raise ValueError(f"Unknown job type: {job_type}")
    
    @classmethod
    def get_strategy_by_job(cls, job: MLJob) -> JobStrategy:
        if isinstance(job, BasicTrainingJob):
            return cls._strategies[StepType.BASIC_TRAINING]
        elif isinstance(job, AdvancedTuningJob):
            return cls._strategies[StepType.ADVANCED_TUNING]
        else:
            raise ValueError(f"Unknown job type: {type(job)}")

    @classmethod
    def find_job(cls, session: Session, job_id: str) -> tuple[Optional[MLJob], Optional[JobStrategy]]:
        """
        Tries to find the job in all known tables.
        Returns (job, strategy) or (None, None).
        """
        for name, strategy in cls._strategies.items():
            job = strategy.get_job(session, job_id)
            if job:
                return job, strategy
        return None, None
