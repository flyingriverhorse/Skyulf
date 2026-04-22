import copy
import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, cast

from backend.ml_pipeline.execution.schemas import NodeConfig, PipelineConfig

logger = logging.getLogger(__name__)

TERMINAL_STEP_TYPES = {"basic_training", "advanced_tuning"}


def partition_for_preview(config: PipelineConfig) -> List[PipelineConfig]:
    """Split a pipeline into one sub-pipeline per data leaf, for preview.

    Unlike :func:`partition_parallel_pipeline`, this does NOT require training
    nodes to detect branches. Training/tuning nodes are stripped, then every
    remaining node with no downstream consumer is treated as a preview leaf
    and gets its own sub-pipeline (containing the leaf and all its ancestors).

    This is what powers the multi-tab "Path A / Path B" preview view when the
    user has multiple parallel preprocessing chains but no training nodes
    (or a single training node where the standard partitioner would only
    return one branch).

    Returns a single-element list (the original config minus training nodes)
    when the graph has only one data leaf.
    """
    # Strip training/tuning nodes — preview never fits models.
    data_nodes = [n for n in config.nodes if n.step_type not in TERMINAL_STEP_TYPES]
    if not data_nodes:
        return [config]

    node_map: Dict[str, NodeConfig] = {n.node_id: n for n in data_nodes}

    # A node is a leaf when no other (data) node lists it as an input.
    has_consumer: set[str] = set()
    for n in data_nodes:
        for parent_id in n.inputs:
            if parent_id in node_map:
                has_consumer.add(parent_id)

    leaves = [n for n in data_nodes if n.node_id not in has_consumer]

    base = PipelineConfig(
        pipeline_id=config.pipeline_id,
        nodes=data_nodes,
        metadata=config.metadata,
    )

    if len(leaves) <= 1:
        return [base]

    sub_configs: List[PipelineConfig] = []
    for idx, leaf in enumerate(leaves):
        ancestors = _collect_ancestors(leaf.node_id, node_map)
        branch_nodes = [node_map[nid] for nid in ancestors if nid in node_map]
        sub_configs.append(
            PipelineConfig(
                pipeline_id=f"{config.pipeline_id}__preview_{idx}",
                nodes=branch_nodes,
                metadata={
                    **config.metadata,
                    "preview_branch_index": idx,
                    "preview_leaf_node_id": leaf.node_id,
                    "parent_pipeline_id": config.pipeline_id,
                },
            )
        )
    return sub_configs


def _split_connected_components(config: PipelineConfig) -> List[PipelineConfig]:
    """Split a pipeline into one PipelineConfig per connected subgraph.

    Disconnected parts of the canvas (e.g. two datasets with no shared nodes)
    become separate experiment groups, each with its own pipeline_id.
    Returns a single-element list when the graph is fully connected.
    """
    node_map: Dict[str, NodeConfig] = {n.node_id: n for n in config.nodes}
    if not node_map:
        return [config]

    # Build undirected adjacency
    adj: Dict[str, set[str]] = defaultdict(set)
    for node in config.nodes:
        for parent_id in node.inputs:
            if parent_id in node_map:
                adj[node.node_id].add(parent_id)
                adj[parent_id].add(node.node_id)

    # BFS to find connected components
    visited: set[str] = set()
    components: List[List[str]] = []
    for nid in node_map:
        if nid in visited:
            continue
        component: List[str] = []
        queue = [nid]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            component.append(cur)
            for neighbor in adj[cur]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    if len(components) <= 1:
        return [config]

    # Each component gets its own pipeline_id
    results: List[PipelineConfig] = []
    for component_ids in components:
        id_set = set(component_ids)
        comp_nodes = [node_map[nid] for nid in component_ids]
        # Preserve original ID for first component, generate new for rest
        comp_pipeline_id = config.pipeline_id if not results else f"preview_{uuid.uuid4().hex[:12]}"
        results.append(
            PipelineConfig(
                pipeline_id=comp_pipeline_id,
                nodes=comp_nodes,
                metadata={**config.metadata},
            )
        )

    return results


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
    terminals = [n for n in config.nodes if n.step_type in TERMINAL_STEP_TYPES]

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
                    metadata={
                        **config.metadata,
                        "branch_index": idx,
                        "parent_pipeline_id": config.pipeline_id,
                    },
                )
            )
        return sub_configs

    # --- Case 1: multiple terminal nodes ---
    # When a terminal has multiple inputs, always split it into per-input
    # sub-branches so each path is a separate experiment. This avoids the
    # confusing "2 terminals but 3 logical paths" scenario.
    sub_configs = []
    global_branch = 0
    for term in terminals:
        is_parallel = term.params.get("execution_mode") == "parallel" and len(term.inputs) > 1
        if is_parallel:
            # Parallel mode: each input becomes its own experiment branch
            for branch_root_id in term.inputs:
                ancestors = _collect_ancestors(branch_root_id, node_map)
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
                        pipeline_id=(f"{config.pipeline_id}__branch_{global_branch}"),
                        nodes=branch_nodes,
                        metadata={
                            **config.metadata,
                            "branch_index": global_branch,
                            "parent_pipeline_id": config.pipeline_id,
                        },
                    )
                )
                global_branch += 1
        else:
            # Merge mode (default) or single input: one branch per terminal
            ancestors = _collect_ancestors(term.node_id, node_map)
            branch_nodes = [node_map[nid] for nid in ancestors if nid in node_map]
            sub_configs.append(
                PipelineConfig(
                    pipeline_id=f"{config.pipeline_id}__branch_{global_branch}",
                    nodes=branch_nodes,
                    metadata={
                        **config.metadata,
                        "branch_index": global_branch,
                        "parent_pipeline_id": config.pipeline_id,
                    },
                )
            )
            global_branch += 1
    return sub_configs


def _collect_ancestors(node_id: str, node_map: Dict[str, NodeConfig]) -> List[str]:
    """Collect a node and all its ancestors in true topological order.

    Algorithm overview
    ------------------
    This runs in two passes:

    1. **Discovery (BFS — Breadth-First Search).** BFS is a graph-traversal
       strategy that explores neighbours level-by-level using a FIFO queue
       (as opposed to DFS which uses a stack and goes deep first). We BFS
       *backwards* through the ``inputs`` edges starting from ``node_id`` to
       collect every ancestor — this gives us the subgraph we care about
       without visiting unrelated nodes elsewhere on the canvas.

    2. **Ordering (Kahn's algorithm).** Once we have the subgraph we run
       Kahn's topological sort restricted to those nodes. Kahn's repeatedly
       emits nodes whose in-degree (number of unprocessed parents) has
       reached 0, then decrements the in-degree of their children. This
       guarantees every parent appears before its children in the output —
       even for diamond-shaped graphs where a node is reachable through
       multiple paths.

    Why both? BFS alone (the previous implementation, with a final
    ``list.reverse()``) is *not* a valid topological sort: when a child
    has multiple parents the visit order can interleave them in a way that
    places the child before one of its parents. That triggered run-time
    ``FileNotFoundError: Artifact not found`` errors because the engine
    tried to load a parent's artifact that hadn't been produced yet.
    """
    if node_id not in node_map:
        return []

    # Pass 1 — BFS backwards from node_id to discover the ancestor subgraph.
    # We use a simple FIFO queue (list.pop(0)) because the graphs are tiny;
    # collections.deque would only matter at much larger scale.
    discovered: set[str] = set()
    queue = [node_id]
    while queue:
        nid = queue.pop(0)
        if nid in discovered or nid not in node_map:
            continue
        discovered.add(nid)
        for parent_id in node_map[nid].inputs:
            if parent_id in node_map and parent_id not in discovered:
                queue.append(parent_id)

    # Pass 2 — Kahn's algorithm: build in-degree map for the subgraph.
    in_degree: Dict[str, int] = {nid: 0 for nid in discovered}
    children: Dict[str, List[str]] = {nid: [] for nid in discovered}
    for nid in discovered:
        for parent_id in node_map[nid].inputs:
            if parent_id in discovered:
                in_degree[nid] += 1
                children[parent_id].append(nid)

    # Pass 2 — repeatedly emit zero-in-degree nodes.
    result: List[str] = []
    ready = [nid for nid, deg in in_degree.items() if deg == 0]
    while ready:
        nid = ready.pop(0)
        result.append(nid)
        for child in children[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)

    if len(result) != len(discovered):
        # Cycle in the ancestor subgraph — cannot topologically sort. Fall
        # back to BFS discovery order so the caller gets *something*; the
        # engine will surface a clearer error when execution actually fails.
        logger.warning(
            "Cycle detected while collecting ancestors of %s; "
            "falling back to BFS-reverse order.",
            node_id,
        )
        return list(reversed(list(discovered)))

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
            node.get("data", {}).get("config") or node.get("parameters") or node.get("data", {})
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
