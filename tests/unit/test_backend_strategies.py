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
        self.assertEqual(str(StepType.TRAINING), "training")
        self.assertEqual(f"{StepType.DATA_LOADER}", "data_loader")

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
        self.job.tuned_thresholds = None
        self.job.tuned_thresholds_enabled = False

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

    def test_handle_success_stamps_leakage_gate_verdict(self):
        """The pre-execution leakage verdict is persisted on the job so the
        Job Details UI can show it as factual per-job information."""
        node_res = NodeExecutionResult(
            node_id="node_1",
            status="success",
            output_artifact_id="path/to/artifact",
            metrics={"accuracy": 0.9},
        )
        verdict = {"status": "passed", "messages": []}
        pipeline_res = PipelineExecutionResult(
            pipeline_id="pipe_123",
            status="success",
            node_results={"node_1": node_res},
            leakage_verdict=verdict,
        )

        self.strategy.handle_success(self.job, pipeline_res)

        self.assertEqual(self.job.metrics["leakage_gate"], verdict)

    def test_handle_success_omits_leakage_gate_for_legacy_results(self):
        """Results without a verdict (e.g. previews) leave metrics untouched."""
        node_res = NodeExecutionResult(
            node_id="node_1",
            status="success",
            output_artifact_id="path/to/artifact",
            metrics={"accuracy": 0.9},
        )
        pipeline_res = PipelineExecutionResult(
            pipeline_id="pipe_123", status="success", node_results={"node_1": node_res}
        )

        self.strategy.handle_success(self.job, pipeline_res)

        self.assertNotIn("leakage_gate", self.job.metrics)

    def test_handle_success_collects_nested_step_dropped_columns(self):
        """Nested preprocessing step details must still feed dropped_columns rollups."""
        node_res = NodeExecutionResult(
            node_id="node_1",
            status="success",
            output_artifact_id="path/to/artifact",
            metrics={
                "accuracy": 0.95,
                "steps": {
                    "0:select": {
                        "details": {
                            "dropped_columns": ["col_A", "col_B"],
                        }
                    }
                },
            },
        )
        pipeline_res = PipelineExecutionResult(
            pipeline_id="pipe_123", status="success", node_results={"node_1": node_res}
        )

        self.strategy.handle_success(self.job, pipeline_res)

        self.assertEqual(self.job.metrics["accuracy"], 0.95)
        self.assertCountEqual(self.job.metrics["dropped_columns"], ["col_A", "col_B"])

    def test_handle_success_seeds_tuned_thresholds_from_training_time_tuning(self):
        """F-13 bridge: thresholds selected during training are copied into
        the per-job tuned-thresholds store (enabled), so Experiments and
        deployment reuse the existing save/toggle lifecycle."""
        node_res = NodeExecutionResult(
            node_id="node_1",
            status="success",
            output_artifact_id="path/to/artifact",
            metrics={
                "accuracy": 0.9,
                "decision_thresholds": {"no": 0.62, "yes": 0.38},
                "decision_threshold_metric": "f1",
            },
        )
        pipeline_res = PipelineExecutionResult(
            pipeline_id="pipe_123", status="success", node_results={"node_1": node_res}
        )

        self.strategy.handle_success(self.job, pipeline_res)

        self.assertEqual(self.job.tuned_thresholds["thresholds"], {"no": 0.62, "yes": 0.38})
        # classes preserve the model's classes_ order, not sorted order
        self.assertEqual(self.job.tuned_thresholds["classes"], ["no", "yes"])
        self.assertEqual(self.job.tuned_thresholds["metric"], "f1")
        self.assertEqual(self.job.tuned_thresholds["split_used"], "validation")
        self.assertIsNotNone(self.job.tuned_thresholds["computed_at"])
        self.assertEqual(self.job.tuned_thresholds["source"], "training")
        self.assertTrue(self.job.tuned_thresholds_enabled)

    def test_handle_success_leaves_threshold_store_alone_without_training_thresholds(self):
        node_res = NodeExecutionResult(
            node_id="node_1",
            status="success",
            output_artifact_id="path/to/artifact",
            metrics={"accuracy": 0.9},
        )
        pipeline_res = PipelineExecutionResult(
            pipeline_id="pipe_123", status="success", node_results={"node_1": node_res}
        )

        self.strategy.handle_success(self.job, pipeline_res)

        self.assertIsNone(self.job.tuned_thresholds)
        self.assertFalse(self.job.tuned_thresholds_enabled)

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
        self.job.tuned_thresholds = None
        self.job.tuned_thresholds_enabled = False

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

    def test_handle_success_seeds_tuned_thresholds(self):
        """The seeding bridge lives on the base strategy and must fire for
        tuning jobs too (their handle_success delegates to super first)."""
        metrics = {
            "best_params": {"max_depth": 5},
            "best_score": 0.88,
            "decision_thresholds": {"0": 0.58, "1": 0.42},
            "decision_threshold_metric": "balanced_accuracy",
        }
        node_res = NodeExecutionResult(
            node_id="tuner_node", status="success", output_artifact_id="path", metrics=metrics
        )
        pipeline_res = PipelineExecutionResult(
            pipeline_id="tune_123", status="success", node_results={"tuner_node": node_res}
        )

        self.strategy.handle_success(self.job, pipeline_res)

        self.assertEqual(self.job.tuned_thresholds["thresholds"], {"0": 0.58, "1": 0.42})
        self.assertEqual(self.job.tuned_thresholds["metric"], "balanced_accuracy")
        self.assertTrue(self.job.tuned_thresholds_enabled)

    def test_handle_success_ignores_empty_decision_thresholds(self):
        """An empty dict (e.g. a defensive emit) must not enable the store."""
        metrics = {"best_score": 0.88, "decision_thresholds": {}}
        node_res = NodeExecutionResult(
            node_id="tuner_node", status="success", output_artifact_id="path", metrics=metrics
        )
        pipeline_res = PipelineExecutionResult(
            pipeline_id="tune_123", status="success", node_results={"tuner_node": node_res}
        )

        self.strategy.handle_success(self.job, pipeline_res)

        self.assertIsNone(self.job.tuned_thresholds)
        self.assertFalse(self.job.tuned_thresholds_enabled)
