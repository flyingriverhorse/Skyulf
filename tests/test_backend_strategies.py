import unittest
from unittest.mock import MagicMock

from backend.database.models import TrainingJob
from backend.ml_pipeline._execution.schemas import (
    JobStatus,
    NodeExecutionResult,
    PipelineExecutionResult,
)
from backend.ml_pipeline._execution.strategies import (
    AdvancedTuningStrategy,
    BasicTrainingStrategy,
)
from backend.ml_pipeline.constants import StepType


class TestEnumStrSemantics(unittest.TestCase):
    """StepType/JobStatus are StrEnum: str()/f-string must yield the bare
    value, not `ClassName.MEMBER` (regression guard against reverting to
    `(str, Enum)`)."""

    def test_step_type_str_returns_value(self):
        self.assertEqual(str(StepType.BASIC_TRAINING), "basic_training")
        self.assertEqual(f"{StepType.ADVANCED_TUNING}", "advanced_tuning")

    def test_job_status_str_returns_value(self):
        self.assertEqual(str(JobStatus.QUEUED), "queued")
        self.assertEqual(f"{JobStatus.COMPLETED}", "completed")


class TestBasicTrainingStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = BasicTrainingStrategy()
        # Mock a Job object
        self.job = MagicMock(spec=TrainingJob)
        self.job.run_mode = "fixed"
        self.job.metrics = {}  # Start with empty metrics

    def test_get_job_model(self):
        self.assertEqual(self.strategy.get_job_model(), TrainingJob)

    def test_get_initial_log(self):
        self.job.version = "1.0.0"
        log = self.strategy.get_initial_log(self.job)
        self.assertIn("Training Job Version: 1.0.0", log)

    def test_handle_success(self):
        # Create a mock execution result
        node_res = NodeExecutionResult(
            node_id="node_1",
            status="success",
            output_artifact_id="path/to/artifact",
            metrics={"accuracy": 0.95, "dropped_columns": ["col_A"]},
        )
        pipeline_res = PipelineExecutionResult(
            pipeline_id="pipe_123", status="success", node_results={"node_1": node_res}
        )

        self.strategy.handle_success(self.job, pipeline_res)

        # Verify job was updated
        self.assertEqual(self.job.metrics["accuracy"], 0.95)
        self.assertEqual(self.job.metrics["dropped_columns"], ["col_A"])

    def test_handle_failure(self):
        error_msg = "Out of Memory"
        self.strategy.handle_failure(self.job, error_msg)

        self.assertEqual(self.job.status, "failed")
        self.assertEqual(self.job.error_message, error_msg)
        self.assertIsNotNone(self.job.finished_at)


class TestAdvancedTuningStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = AdvancedTuningStrategy()
        self.job = MagicMock(spec=TrainingJob)
        self.job.run_mode = "tuned"
        self.job.metrics = {}
        # Mock specific tuning fields
        self.job.best_params = {}
        self.job.best_score = 0.0

    def test_get_job_model(self):
        self.assertEqual(self.strategy.get_job_model(), TrainingJob)

    def test_handle_success_tuning_fields(self):
        # Result simulates a Tuning node output
        metrics = {
            "best_params": {"max_depth": 5},
            "best_score": 0.88,
            "trials": [{"id": 1, "score": 0.85}, {"id": 2, "score": 0.88}],
        }

        node_res = NodeExecutionResult(
            node_id="tuner_node", status="success", output_artifact_id="path", metrics=metrics
        )
        pipeline_res = PipelineExecutionResult(
            pipeline_id="tune_123", status="success", node_results={"tuner_node": node_res}
        )

        self.strategy.handle_success(self.job, pipeline_res)

        # Verify tuning specific fields were extracted from metrics to the job model
        self.assertEqual(self.job.best_params, {"max_depth": 5})
        self.assertEqual(self.job.best_score, 0.88)
