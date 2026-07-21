"""Focused unit tests for ``extract_job_details`` in
``backend.ml_pipeline._execution.graph_utils``.

Covers `target_column` extraction from a raw job `graph` dict for
training terminal node types, including the unified `StepType.TRAINING`
node with ``run_mode`` fixed and tuned.
"""

from backend.ml_pipeline._execution.graph_utils import extract_job_details
from backend.ml_pipeline.constants import StepType


def test_extract_job_details_basic_training_target_column():
    graph = {
        "nodes": [
            {"node_id": "n1", "step_type": StepType.DATA_LOADER.value, "params": {}},
            {
                "node_id": "n2",
                "step_type": StepType.TRAINING.value,
                "params": {"run_mode": "fixed", "target_column": "y", "model_type": "random_forest"},
            },
        ]
    }
    _, target_column, _ = extract_job_details(graph, "n2")
    assert target_column == "y"


def test_extract_job_details_advanced_tuning_target_column():
    graph = {
        "nodes": [
            {"node_id": "n1", "step_type": StepType.DATA_LOADER.value, "params": {}},
            {
                "node_id": "n2",
                "step_type": StepType.TRAINING.value,
                "params": {"run_mode": "tuned", "target_column": "y", "algorithm": "random_forest"},
            },
        ]
    }
    _, target_column, _ = extract_job_details(graph, "n2")
    assert target_column == "y"


def test_extract_job_details_unified_training_fixed_target_column():
    graph = {
        "nodes": [
            {"node_id": "n1", "step_type": StepType.DATA_LOADER.value, "params": {}},
            {
                "node_id": "n2",
                "step_type": StepType.TRAINING.value,
                "params": {
                    "target_column": "y",
                    "run_mode": "fixed",
                    "model_type": "random_forest",
                },
            },
        ]
    }
    _, target_column, _ = extract_job_details(graph, "n2")
    assert target_column == "y"


def test_extract_job_details_unified_training_tuned_target_column():
    graph = {
        "nodes": [
            {"node_id": "n1", "step_type": StepType.DATA_LOADER.value, "params": {}},
            {
                "node_id": "n2",
                "step_type": StepType.TRAINING.value,
                "params": {
                    "target_column": "y",
                    "run_mode": "tuned",
                    "algorithm": "random_forest",
                },
            },
        ]
    }
    _, target_column, _ = extract_job_details(graph, "n2")
    assert target_column == "y"
