"""Focused unit tests for ``backend.ml_pipeline._internal._helpers``.

Covers `_resolve_branch_context` (and its public wrapper `branch_label`),
which extracts a branch's terminal model_type for the "Path A · <model>"
label — including the unified `StepType.TRAINING` node case (fixed/tuned
run modes).
"""

from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline._internal import _helpers
from backend.ml_pipeline.constants import StepType


def test_resolve_branch_context_basic_training_uses_model_type():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"run_mode": "fixed", "model_type": "random_forest"},
                inputs=["n1"],
            ),
        ],
    )
    model_type, leaf_step, leaf_display = _helpers._resolve_branch_context(sub)
    assert model_type == "random_forest"
    assert leaf_step == ""
    assert leaf_display == ""


def test_resolve_branch_context_advanced_tuning_falls_back_to_algorithm():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"run_mode": "tuned", "algorithm": "xgboost"},
                inputs=["n1"],
            ),
        ],
    )
    model_type, _, _ = _helpers._resolve_branch_context(sub)
    assert model_type == "xgboost"


def test_resolve_branch_context_training_fixed_uses_model_type():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"run_mode": "fixed", "model_type": "random_forest"},
                inputs=["n1"],
            ),
        ],
    )
    model_type, leaf_step, leaf_display = _helpers._resolve_branch_context(sub)
    assert model_type == "random_forest"
    assert leaf_step == ""
    assert leaf_display == ""


def test_resolve_branch_context_training_tuned_falls_back_to_algorithm():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"run_mode": "tuned", "algorithm": "xgboost"},
                inputs=["n1"],
            ),
        ],
    )
    model_type, _, _ = _helpers._resolve_branch_context(sub)
    assert model_type == "xgboost"


def test_branch_label_training_node_matches_basic_training_format():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"run_mode": "fixed", "model_type": "random_forest_classifier"},
                inputs=["n1"],
            ),
        ],
    )
    assert _helpers.branch_label(0, sub) == "Path A · Random Forest"
