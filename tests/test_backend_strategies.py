import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

from backend.ml_pipeline.execution.strategies import (
    JobStrategyFactory, 
    BasicTrainingStrategy, 
    AdvancedTuningStrategy
)
from backend.ml_pipeline.constants import StepType
from backend.database.models import BasicTrainingJob, AdvancedTuningJob
from backend.ml_pipeline.execution.schemas import PipelineExecutionResult, NodeExecutionResult

class TestJobStrategyFactory(unittest.TestCase):
    def test_get_basic_strategy(self):
        strategy = JobStrategyFactory.get_strategy(StepType.BASIC_TRAINING)
        self.assertIsInstance(strategy, BasicTrainingStrategy)

    def test_get_tuning_strategy(self):
        strategy = JobStrategyFactory.get_strategy(StepType.ADVANCED_TUNING)
        self.assertIsInstance(strategy, AdvancedTuningStrategy)

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            JobStrategyFactory.get_strategy("INVALID_TYPE")

class TestBasicTrainingStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = BasicTrainingStrategy()
        # Mock a Job object
        self.job = MagicMock(spec=BasicTrainingJob)
        self.job.metrics = {}  # Start with empty metrics
        
    def test_get_job_model(self):
        self.assertEqual(self.strategy.get_job_model(), BasicTrainingJob)

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
            metrics={"accuracy": 0.95, "dropped_columns": ["col_A"]}
        )
        pipeline_res = PipelineExecutionResult(
            pipeline_id="pipe_123",
            status="success",
            node_results={"node_1": node_res}
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
        self.job = MagicMock(spec=AdvancedTuningJob)
        self.job.metrics = {}
        # Mock specific tuning fields
        self.job.best_params = {}
        self.job.best_score = 0.0

    def test_get_job_model(self):
        self.assertEqual(self.strategy.get_job_model(), AdvancedTuningJob)

    def test_handle_success_tuning_fields(self):
        # Result simulates a Tuning node output
        metrics = {
            "best_params": {"max_depth": 5},
            "best_score": 0.88,
            "trials": [{"id": 1, "score": 0.85}, {"id": 2, "score": 0.88}]
        }
        
        node_res = NodeExecutionResult(
            node_id="tuner_node",
            status="success",
            output_artifact_id="path",
            metrics=metrics
        )
        pipeline_res = PipelineExecutionResult(
            pipeline_id="tune_123",
            status="success",
            node_results={"tuner_node": node_res}
        )

        self.strategy.handle_success(self.job, pipeline_res)
        
        # Verify tuning specific fields were extracted from metrics to the job model
        self.assertEqual(self.job.best_params, {"max_depth": 5})
        self.assertEqual(self.job.best_score, 0.88)

