"""Inspect Optuna pruning eligibility without executing or loading a pipeline."""

from typing import Any

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
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig

router = APIRouter(tags=["ML Pipeline"])


class PruningSupportRequest(BaseModel):
    """Current model/search settings and converted graph for pruning inspection."""

    model_type: str
    search_space: dict[str, list[Any]]
    pipeline: PipelineConfigModel
    node_id: str
    strategy_params: dict[str, Any] = Field(default_factory=dict)


class PruningSupportResponse(BaseModel):
    """Whether incremental pruning is available, with its disabling reason."""

    supported: bool
    reason: str | None


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


def _graph_pruning_reason(node: NodeConfig, nodes: dict[str, NodeConfig]) -> str | None:
    """Mirror fold replay selection while leaving saved artifacts and data untouched."""
    if _has_merged_ancestors(node, nodes):
        return "Cannot confirm incremental pruning support for merged pipeline inputs"
    inspector = FeatureEngMixin()
    inspector._node_configs = nodes
    resolved = inspector._upstream_fe_chain(node)
    if resolved is None:
        return "Cannot confirm pruning support: connect a supported path from a data source"
    _loader_id, chain = resolved
    steps = [step for _node_id, node_steps in chain for step in node_steps]
    _first, unsafe, replay = partition_fold_steps(steps, node.params.get("target_column"))
    if unsafe:
        return "Data-dependent preprocessing before the first split cannot be refitted safely"
    if replay:
        return "Pruning is unavailable when preprocessing runs inside each validation fold"
    return None


def _model_pruning_reason(request: PruningSupportRequest, node: NodeConfig) -> str | None:
    """Use Advanced tuning defaults and the real Core estimator/wrapper preparation."""
    task_type = (
        node.params.get("task_type") or node.params.get("problem_type") or node.params.get("task")
    )
    calculator, _applier = get_model_components(
        request.model_type, task_type=task_type if isinstance(task_type, str) else None
    )
    config = TuningConfig(
        strategy="optuna",
        search_space=request.search_space,
        strategy_params=request.strategy_params,
    )
    return TuningCalculator(calculator).pruning_support_reason(config)


@router.post("/pruning-support", response_model=PruningSupportResponse)
def get_pruning_support(request: PruningSupportRequest) -> PruningSupportResponse:
    """Return configuration eligibility; the current selected pruner does not disable itself.

    Model defaults match Advanced tuning, so Basic-mode hyperparameters are not
    applied. Graph inspection shares runtime fold selection, independent of the
    post-tuning CV toggle. Merged graphs conservatively require confirmation;
    this route never fits models or reads data, artifacts, files, or the database.
    """
    try:
        nodes = _validated_nodes(request)
        node = nodes[request.node_id]
        reason = _model_pruning_reason(request, node) or _graph_pruning_reason(node, nodes)
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        reason = f"Cannot confirm pruning support: {exc}"
    return PruningSupportResponse(supported=reason is None, reason=reason)
