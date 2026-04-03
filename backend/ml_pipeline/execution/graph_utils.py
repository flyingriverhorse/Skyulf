import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, cast

from backend.ml_pipeline.execution.schemas import NodeConfig, PipelineConfig

logger = logging.getLogger(__name__)

TERMINAL_STEP_TYPES = {"basic_training", "advanced_tuning"}


def partition_parallel_pipeline(config: PipelineConfig) -> List[PipelineConfig]:
    """Split a pipeline into independent sub-pipelines for parallel execution.

    Returns a list of PipelineConfig objects.  When the graph contains only a
    single execution branch the list has exactly one element (the original
    config, unchanged).

    Two scenarios trigger partitioning:

    1. **Multiple terminal (training/tuning) nodes** on the canvas — each
       terminal becomes its own sub-pipeline.
    2. **Single terminal with ``execution_mode == "parallel"``** in its params —
       each incoming branch is treated as a separate experiment sharing the
       same model configuration.
    """
    node_map: Dict[str, NodeConfig] = {n.node_id: n for n in config.nodes}

    # Identify terminal nodes
    terminals = [
        n for n in config.nodes if n.step_type in TERMINAL_STEP_TYPES
    ]

    if not terminals:
        return [config]

    # --- Case 2: single terminal with parallel mode ---
    if len(terminals) == 1:
        term = terminals[0]
        if term.params.get("execution_mode") != "parallel" or len(term.inputs) <= 1:
            return [config]

        # Each input to the terminal is a separate experiment branch
        sub_configs: List[PipelineConfig] = []
        for idx, branch_root_id in enumerate(term.inputs):
            ancestors = _collect_ancestors(branch_root_id, node_map)
            # Build a copy of the terminal node with only this branch's input
            term_copy = NodeConfig(
                node_id=term.node_id,
                step_type=term.step_type,
                params={k: v for k, v in term.params.items() if k != "execution_mode"},
                inputs=[branch_root_id],
            )
            branch_nodes = [node_map[nid] for nid in ancestors if nid in node_map]
            branch_nodes.append(term_copy)
            sub_configs.append(
                PipelineConfig(
                    pipeline_id=f"{config.pipeline_id}__branch_{idx}",
                    nodes=branch_nodes,
                    metadata={**config.metadata, "branch_index": idx,
                              "parent_pipeline_id": config.pipeline_id},
                )
            )
        return sub_configs

    # --- Case 1: multiple terminal nodes ---
    sub_configs = []
    for idx, term in enumerate(terminals):
        # Check if this terminal also needs per-input splitting (parallel mode)
        if (
            term.params.get("execution_mode") == "parallel"
            and len(term.inputs) > 1
        ):
            for sub_idx, branch_root_id in enumerate(term.inputs):
                ancestors = _collect_ancestors(branch_root_id, node_map)
                term_copy = NodeConfig(
                    node_id=term.node_id,
                    step_type=term.step_type,
                    params={k: v for k, v in term.params.items()
                            if k != "execution_mode"},
                    inputs=[branch_root_id],
                )
                branch_nodes = [
                    node_map[nid] for nid in ancestors if nid in node_map
                ]
                branch_nodes.append(term_copy)
                sub_configs.append(
                    PipelineConfig(
                        pipeline_id=(
                            f"{config.pipeline_id}__branch_{idx}_{sub_idx}"
                        ),
                        nodes=branch_nodes,
                        metadata={
                            **config.metadata,
                            "branch_index": idx,
                            "sub_branch_index": sub_idx,
                            "parent_pipeline_id": config.pipeline_id,
                        },
                    )
                )
        else:
            ancestors = _collect_ancestors(term.node_id, node_map)
            branch_nodes = [
                node_map[nid] for nid in ancestors if nid in node_map
            ]
            sub_configs.append(
                PipelineConfig(
                    pipeline_id=f"{config.pipeline_id}__branch_{idx}",
                    nodes=branch_nodes,
                    metadata={
                        **config.metadata,
                        "branch_index": idx,
                        "parent_pipeline_id": config.pipeline_id,
                    },
                )
            )
    return sub_configs


def _collect_ancestors(node_id: str, node_map: Dict[str, NodeConfig]) -> List[str]:
    """BFS backwards to collect all ancestor node IDs in topological order."""
    visited: set[str] = set()
    queue = [node_id]
    result: List[str] = []

    while queue:
        nid = queue.pop(0)
        if nid in visited or nid not in node_map:
            continue
        visited.add(nid)
        result.append(nid)
        for parent_id in node_map[nid].inputs:
            queue.append(parent_id)

    # Reverse for topological order (parents first)
    result.reverse()
    return result


def _get_strategy_from_params(params: Dict[str, Any]) -> Optional[str]:
    if "tuning_config" in params and "strategy" in params["tuning_config"]:
        return cast(str, params["tuning_config"]["strategy"])
    if "tuning" in params and "strategy" in params["tuning"]:
        return cast(str, params["tuning"]["strategy"])
    if "search_strategy" in params:
        return cast(str, params["search_strategy"])
    if "strategy" in params:
        return cast(str, params["strategy"])
    return None


def extract_tuning_strategy(node_data: Dict[str, Any]) -> Optional[str]:
    """Extracts tuning strategy from node data."""
    # Check PipelineConfigModel params
    params = node_data.get("params", {})
    strategy = _get_strategy_from_params(params)
    if strategy:
        return strategy

    # Check React Flow data/config
    data = node_data.get("data", {})
    config = data.get("config", {})

    if "tuning" in config and "strategy" in config["tuning"]:
        return cast(str, config["tuning"]["strategy"])
    if "strategy" in config:
        return cast(str, config["strategy"])
    if "search_strategy" in data:
        return cast(str, data["search_strategy"])
    return None


def _find_strategy_in_nodes(
    nodes: List[Dict[str, Any]], node_id: Optional[str] = None
) -> Optional[str]:
    for node in nodes:
        if node_id:
            if node.get("node_id") == node_id or node.get("id") == node_id:
                found = extract_tuning_strategy(node)
                if found:
                    return found
        else:
            found = extract_tuning_strategy(node)
            if found and found != "random":
                return found
    return None


def determine_search_strategy(graph: Dict[str, Any], node_id: str) -> str:
    """Determines the search strategy for a tuning job from the graph."""
    if not graph or "nodes" not in graph:
        return "random"

    # 1. Try to find in target node
    strategy = _find_strategy_in_nodes(graph["nodes"], node_id)
    if strategy:
        return strategy

    # 2. If not found, look for ANY node with strategy
    strategy = _find_strategy_in_nodes(graph["nodes"])
    if strategy:
        return strategy

    return "random"


def _parse_node_info(node: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    if "step_type" in node:
        # PipelineConfigModel structure
        nid = node.get("node_id", "")
        ntype = node.get("step_type", "")
        params = node.get("params", {})
    else:
        # React Flow structure
        nid = node.get("id", "")
        ntype = node.get("type") or node.get("data", {}).get("catalogType") or ""
        # Try config, then parameters, then data itself
        params = (
            node.get("data", {}).get("config")
            or node.get("parameters")
            or node.get("data", {})
        )
    return nid, ntype, params


from backend.ml_pipeline.constants import StepType

def _extract_columns(ntype: str, params: Dict[str, Any]) -> List[str]:
    dropped: List[str] = []
    if ntype in [
        "drop_missing_columns",
        "DropMissingColumns",
        "drop_column_recommendations",
        "drop_columns",
    ]:
        cols = params.get("columns")
        if isinstance(cols, list):
            dropped.extend(cols)
    if ntype == "feature_selection":
        dropped_feats = params.get("dropped_features")
        if isinstance(dropped_feats, list):
            dropped.extend(dropped_feats)
    return dropped


def extract_job_details(
    graph: Dict[str, Any], node_id: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str]]:
    """Extracts hyperparameters, target_column, and dropped_columns from graph."""
    hyperparameters = None
    target_column = None
    dropped_columns: List[str] = []

    if not graph or "nodes" not in graph:
        return hyperparameters, target_column, dropped_columns

    for node in graph["nodes"]:
        nid, ntype, params = _parse_node_info(node)

        if nid == node_id:
            hyperparameters = params

        # Also look for target column
        if ntype in [
            "train_test_split",
            "TrainTestSplitter",
            "feature_target_split",
            StepType.BASIC_TRAINING,
            StepType.ADVANCED_TUNING,
            "hyperparameter_tuning",
        ] and params.get("target_column"):
            target_column = params.get("target_column")

        dropped_columns.extend(_extract_columns(ntype, params))

    return hyperparameters, target_column, dropped_columns
