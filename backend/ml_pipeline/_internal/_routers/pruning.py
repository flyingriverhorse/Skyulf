"""Inspect Optuna pruning eligibility without executing or loading a pipeline."""

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.ml_pipeline._execution._cycle_validation import validate_no_cycles
from backend.ml_pipeline._execution.engine._feature_eng import (
    FeatureEngMixin,
    partition_fold_steps,
)
from backend.ml_pipeline._execution.model_components import get_model_components
from backend.ml_pipeline._execution.schemas import NodeConfig
from backend.ml_pipeline._internal._schemas import PipelineConfigModel
from backend.ml_pipeline.constants import StepType
from skyulf.leakage import train_test_splitters
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling._tuning.splitters import select_cv_by_type

router = APIRouter(tags=["ML Pipeline"])


class PruningSupportRequest(BaseModel):
    """Current model/search settings and converted graph for pruning inspection."""

    model_type: str
    search_space: dict[str, list[Any]]
    pipeline: PipelineConfigModel
    node_id: str
    strategy_params: dict[str, Any] = Field(default_factory=dict)


class PruningSupportResponse(BaseModel):
    """Whether trials can stop during training or between CV folds, with a reason."""

    supported: bool
    mode: Literal["iterations", "folds", "none"]
    reason: str


def _validated_nodes(request: PruningSupportRequest) -> dict[str, NodeConfig]:
    """Reject malformed topology before the runtime's ancestor traversal can follow it."""
    nodes = [NodeConfig(**node.model_dump()) for node in request.pipeline.nodes]
    by_id = {node.node_id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Cannot confirm pruning support: duplicate pipeline node IDs")
    if request.node_id not in by_id:
        raise ValueError("Cannot confirm pruning support: the selected model node is missing")
    if any(source not in by_id for node in nodes for source in node.inputs):
        raise ValueError("Cannot confirm pruning support: an upstream pipeline node is missing")
    validate_no_cycles(nodes)
    if by_id[request.node_id].step_type != StepType.TRAINING:
        raise ValueError("Cannot confirm pruning support: the selected node is not a model")
    return by_id


def _has_merged_ancestors(node: NodeConfig, nodes: dict[str, NodeConfig]) -> bool:
    """Inspect only the selected model's ancestors after cycle and reference validation."""
    while node.inputs:
        if len(set(node.inputs)) > 1:
            return True
        node = nodes[node.inputs[0]]
    return False


def _graph_pruning_context(node: NodeConfig, nodes: dict[str, NodeConfig]) -> tuple[bool, bool]:
    """Mirror fold replay selection while leaving saved artifacts and data untouched."""
    if _has_merged_ancestors(node, nodes):
        raise ValueError("Cannot confirm pruning support for merged pipeline inputs")
    inspector = FeatureEngMixin()
    inspector._node_configs = nodes
    resolved = inspector._upstream_fe_chain(node)
    if resolved is None:
        raise ValueError("Connect a supported path from a data source")
    _loader_id, chain = resolved
    steps = [step for _node_id, node_steps in chain for step in node_steps]
    _first, unsafe, replay = partition_fold_steps(steps, node.params.get("target_column"))
    if unsafe:
        raise ValueError(
            "Data-dependent preprocessing before the first split cannot be refitted safely"
        )
    holdout = any(
        float((step.get("params") or {}).get("validation_size", 0.0)) > 0
        for step in steps
        if step.get("transformer") in train_test_splitters()
    )
    return bool(replay), holdout


def _model_pruning_support(
    request: PruningSupportRequest, node: NodeConfig, preprocessing: bool, holdout: bool
) -> PruningSupportResponse:
    """Use Advanced tuning defaults and the real Core estimator/wrapper preparation."""
    task_type = (
        node.params.get("task_type") or node.params.get("problem_type") or node.params.get("task")
    )
    calculator, _applier = get_model_components(
        request.model_type, task_type=task_type if isinstance(task_type, str) else None
    )
    tuning = dict(node.params.get("tuning_config") or {})
    calculator.prepare_tuning_params(tuning)
    search_space = request.search_space or calculator.build_tuning_search_space(tuning, "optuna")
    config = TuningConfig(
        strategy="optuna",
        search_space=search_space,
        strategy_params=request.strategy_params,
        cv_enabled=tuning.get("cv_enabled", True),
        cv_folds=tuning.get("cv_folds", 5),
        cv_type=tuning.get("cv_type", "k_fold"),
    )
    n_splits = 1 if holdout else select_cv_by_type(config, calculator.problem_type).get_n_splits()
    return PruningSupportResponse(
        **TuningCalculator(calculator).pruning_support(
            config, n_splits=n_splits, preprocessing=preprocessing
        )
    )


@router.post("/pruning-support", response_model=PruningSupportResponse)
def get_pruning_support(request: PruningSupportRequest) -> PruningSupportResponse:
    """Return pruning capability; the selected pruner does not disable itself.

    Model defaults match Advanced tuning, so Basic-mode hyperparameters are not
    applied. An upstream validation split overrides tuning CV just as it does
    during execution. Merged graphs conservatively require confirmation;
    this route never fits models or reads data, artifacts, files, or the database.
    """
    try:
        nodes = _validated_nodes(request)
        node = nodes[request.node_id]
        preprocessing, holdout = _graph_pruning_context(node, nodes)
        return _model_pruning_support(request, node, preprocessing, holdout)
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        reason = f"Cannot confirm pruning support: {exc}"
    return PruningSupportResponse(supported=False, mode="none", reason=reason)
