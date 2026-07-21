"""Focused unit tests for the internal helper functions in
``backend.ml_pipeline._internal._routers.run_pipeline``.

These target pure functions that classify a sub-pipeline's terminal node
into a `(model_type, job_type)` pair (or pick the terminal node id itself)
before job creation — logic that is otherwise only exercised indirectly
(and rarely) via the full `/run` endpoint integration tests.
"""

from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline._internal._routers import run_pipeline as run_pipeline_mod
from backend.ml_pipeline.constants import StepType

# --------------------------------------------------------------------------
# _resolve_branch_target_node_id
# --------------------------------------------------------------------------


def test_resolve_branch_target_node_id_basic_training_terminal():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
            NodeConfig(
                node_id="n2", step_type=StepType.TRAINING, params={}, inputs=["n1"]
            ),
        ],
    )
    assert run_pipeline_mod._resolve_branch_target_node_id(sub, None) == "n2"


def test_resolve_branch_target_node_id_training_terminal_fixed():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"run_mode": "fixed"},
                inputs=["n1"],
            ),
        ],
    )
    assert run_pipeline_mod._resolve_branch_target_node_id(sub, None) == "n2"


def test_resolve_branch_target_node_id_training_terminal_tuned():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="n1", step_type=StepType.DATA_LOADER, params={}, inputs=[]),
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"run_mode": "tuned"},
                inputs=["n1"],
            ),
        ],
    )
    assert run_pipeline_mod._resolve_branch_target_node_id(sub, None) == "n2"


# --------------------------------------------------------------------------
# _resolve_model_and_job_type
# --------------------------------------------------------------------------


def test_resolve_model_and_job_type_basic_training():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"run_mode": "fixed", "model_type": "random_forest"},
                inputs=["n1"],
            ),
        ],
    )
    model_type, job_type = run_pipeline_mod._resolve_model_and_job_type(sub, "n2", None)
    assert model_type == "random_forest"
    assert job_type == "training"


def test_resolve_model_and_job_type_advanced_tuning():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"run_mode": "tuned", "algorithm": "random_forest"},
                inputs=["n1"],
            ),
        ],
    )
    model_type, job_type = run_pipeline_mod._resolve_model_and_job_type(sub, "n2", None)
    assert model_type == "random_forest"
    assert job_type == "tuning"


def test_resolve_model_and_job_type_training_fixed_resolves_to_basic_training():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"run_mode": "fixed", "model_type": "random_forest"},
                inputs=["n1"],
            ),
        ],
    )
    model_type, job_type = run_pipeline_mod._resolve_model_and_job_type(sub, "n2", None)
    assert model_type == "random_forest"
    assert job_type == "training"


def test_resolve_model_and_job_type_training_tuned_resolves_to_advanced_tuning():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"run_mode": "tuned", "algorithm": "random_forest"},
                inputs=["n1"],
            ),
        ],
    )
    model_type, job_type = run_pipeline_mod._resolve_model_and_job_type(sub, "n2", None)
    assert model_type == "random_forest"
    assert job_type == "tuning"


def test_resolve_model_and_job_type_training_defaults_to_fixed_when_run_mode_missing():
    sub = PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(
                node_id="n2",
                step_type=StepType.TRAINING,
                params={"model_type": "logistic_regression"},
                inputs=["n1"],
            ),
        ],
    )
    model_type, job_type = run_pipeline_mod._resolve_model_and_job_type(sub, "n2", None)
    assert model_type == "logistic_regression"
    assert job_type == "training"
