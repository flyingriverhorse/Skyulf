import logging
import uuid
from collections import defaultdict
from functools import lru_cache
from typing import Any, cast

from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig

logger = logging.getLogger(__name__)

# Model families for the training-node model dropdowns that combine several
# underlying classifier/regressor models into one prediction (voting/stacking).
# These always resolve to "ensemble" regardless of the underlying model's own
# classification/regression tags — kept here (rather than duplicated in both
# `monitoring/router.py`'s Slow Nodes display and the training job managers)
# so there is exactly one place model-family resolution happens.
ENSEMBLE_MODEL_TYPES = frozenset(
    {"voting_classifier", "stacking_classifier", "voting_regressor", "stacking_regressor"}
)

TERMINAL_STEP_TYPES = {"training"}
# Step types that should be treated as terminal sinks when splitting a graph
# into parallel sub-pipelines for execution. Includes ``data_preview`` so a
# preview leaf coexisting with a training/tuning terminal gets its own
# executable sub-pipeline (otherwise it would be dropped because it isn't an
# ancestor of any training terminal).
PARTITION_TERMINAL_STEP_TYPES = TERMINAL_STEP_TYPES | {"data_preview"}

# Terminals that auto-parallelize when wired to multiple inputs, even without
# an explicit ``execution_mode == 'parallel'`` flag. A Data Preview node with
# two upstream paths almost always means "show me each path side-by-side",
# not "merge them and show one merged view" -- the latter is rarely useful
# and was a frequent source of confusion.
AUTO_PARALLEL_STEP_TYPES = {"data_preview"}


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Return ``items`` with duplicates removed, preserving original order."""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _unique_inputs(node: NodeConfig) -> list[str]:
    """Multi-handle splitters (TrainTestSplitter, FeatureTargetSplitter) emit
    several edges from the same source into one downstream node. The frontend
    converter dedupes ``inputs`` already, but older saved pipelines may still
    contain duplicates — so the partition logic must also be defensive: count
    branches by *unique source*, never by raw edge count.
    """
    return _dedupe_preserve_order(list(node.inputs))


def _is_parallel_terminal(term: NodeConfig) -> bool:
    """Return True when this terminal should split into per-input branches."""
    if len(_unique_inputs(term)) <= 1:
        return False
    if term.params.get("execution_mode") == "parallel":
        return True
    return term.step_type in AUTO_PARALLEL_STEP_TYPES


def _preview_data_leaves(data_nodes: list[NodeConfig]) -> list[NodeConfig]:
    """Return the nodes among ``data_nodes`` that have no downstream consumer."""
    node_map: dict[str, NodeConfig] = {n.node_id: n for n in data_nodes}
    has_consumer: set[str] = set()
    for n in data_nodes:
        for parent_id in n.inputs:
            if parent_id in node_map:
                has_consumer.add(parent_id)
    return [n for n in data_nodes if n.node_id not in has_consumer]


def _build_preview_leaf_sub_configs(
    config: PipelineConfig, node_map: dict[str, NodeConfig], leaves: list[NodeConfig]
) -> list[PipelineConfig]:
    """Build one sub-pipeline (leaf + its ancestors) per preview data leaf."""
    sub_configs: list[PipelineConfig] = []
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


def partition_for_preview(config: PipelineConfig) -> list[PipelineConfig]:
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

    node_map: dict[str, NodeConfig] = {n.node_id: n for n in data_nodes}

    # A node is a leaf when no other (data) node lists it as an input.
    leaves = _preview_data_leaves(data_nodes)

    base = PipelineConfig(
        pipeline_id=config.pipeline_id,
        nodes=data_nodes,
        metadata=config.metadata,
    )

    if len(leaves) <= 1:
        return [base]

    return _build_preview_leaf_sub_configs(config, node_map, leaves)


def _build_undirected_adjacency(config: PipelineConfig) -> dict[str, set[str]]:
    """Build an undirected adjacency map from each node's ``inputs`` edges."""
    node_map: dict[str, NodeConfig] = {n.node_id: n for n in config.nodes}
    adj: dict[str, set[str]] = defaultdict(set)
    for node in config.nodes:
        for parent_id in node.inputs:
            if parent_id in node_map:
                adj[node.node_id].add(parent_id)
                adj[parent_id].add(node.node_id)
    return adj


def _find_connected_components(
    node_map: dict[str, NodeConfig], adj: dict[str, set[str]]
) -> list[list[str]]:
    """BFS over ``adj`` to find each connected component of node ids in ``node_map``."""
    visited: set[str] = set()
    components: list[list[str]] = []
    for nid in node_map:
        if nid in visited:
            continue
        component: list[str] = []
        queue = [nid]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            component.append(cur)
            queue.extend(neighbor for neighbor in adj[cur] if neighbor not in visited)
        components.append(component)
    return components


def _build_component_configs(
    config: PipelineConfig, node_map: dict[str, NodeConfig], components: list[list[str]]
) -> list[PipelineConfig]:
    """Build one PipelineConfig per connected component, giving each its own pipeline_id."""
    results: list[PipelineConfig] = []
    for component_ids in components:
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


def _split_connected_components(config: PipelineConfig) -> list[PipelineConfig]:
    """Split a pipeline into one PipelineConfig per connected subgraph.

    Disconnected parts of the canvas (e.g. two datasets with no shared nodes)
    become separate experiment groups, each with its own pipeline_id.
    Returns a single-element list when the graph is fully connected.
    """
    node_map: dict[str, NodeConfig] = {n.node_id: n for n in config.nodes}
    if not node_map:
        return [config]

    adj = _build_undirected_adjacency(config)
    components = _find_connected_components(node_map, adj)

    if len(components) <= 1:
        return [config]

    return _build_component_configs(config, node_map, components)


def _terminal_copy_for_branch(term: NodeConfig, branch_root_id: str) -> NodeConfig:
    """Build a copy of a parallel terminal node scoped to a single branch input."""
    return NodeConfig(
        node_id=term.node_id,
        step_type=term.step_type,
        params={k: v for k, v in term.params.items() if k != "execution_mode"},
        inputs=[branch_root_id],
    )


def _build_parallel_branch_config(
    config: PipelineConfig,
    node_map: dict[str, NodeConfig],
    term: NodeConfig,
    branch_root_id: str,
    branch_index: int,
) -> PipelineConfig:
    """Build the sub-pipeline for one input branch of a parallel terminal."""
    ancestors = _collect_ancestors(branch_root_id, node_map)
    term_copy = _terminal_copy_for_branch(term, branch_root_id)
    branch_nodes = [node_map[nid] for nid in ancestors if nid in node_map]
    branch_nodes.append(term_copy)
    return PipelineConfig(
        pipeline_id=f"{config.pipeline_id}__branch_{branch_index}",
        nodes=branch_nodes,
        metadata={
            **config.metadata,
            "branch_index": branch_index,
            "parent_pipeline_id": config.pipeline_id,
        },
    )


def _build_merge_branch_config(
    config: PipelineConfig,
    node_map: dict[str, NodeConfig],
    term: NodeConfig,
    branch_index: int,
) -> PipelineConfig:
    """Build the sub-pipeline for a non-parallel (merge-mode) terminal."""
    ancestors = _collect_ancestors(term.node_id, node_map)
    branch_nodes = [node_map[nid] for nid in ancestors if nid in node_map]
    return PipelineConfig(
        pipeline_id=f"{config.pipeline_id}__branch_{branch_index}",
        nodes=branch_nodes,
        metadata={
            **config.metadata,
            "branch_index": branch_index,
            "parent_pipeline_id": config.pipeline_id,
        },
    )


def _partition_single_terminal(
    config: PipelineConfig, node_map: dict[str, NodeConfig], term: NodeConfig
) -> list[PipelineConfig]:
    """Partition a single-terminal pipeline, splitting per-input only if parallel mode is set."""
    if not _is_parallel_terminal(term):
        return [config]

    # Each input to the terminal is a separate experiment branch
    return [
        _build_parallel_branch_config(config, node_map, term, branch_root_id, idx)
        for idx, branch_root_id in enumerate(_unique_inputs(term))
    ]


def _partition_multiple_terminals(
    config: PipelineConfig,
    node_map: dict[str, NodeConfig],
    terminals: list[NodeConfig],
) -> list[PipelineConfig]:
    """Partition a pipeline with multiple terminal nodes into one sub-pipeline per branch.

    When a terminal has multiple inputs, always split it into per-input
    sub-branches so each path is a separate experiment. This avoids the
    confusing "2 terminals but 3 logical paths" scenario.
    """
    sub_configs: list[PipelineConfig] = []
    global_branch = 0
    for term in terminals:
        if _is_parallel_terminal(term):
            # Parallel mode: each input becomes its own experiment branch
            for branch_root_id in _unique_inputs(term):
                sub_configs.append(
                    _build_parallel_branch_config(
                        config, node_map, term, branch_root_id, global_branch
                    )
                )
                global_branch += 1
        else:
            # Merge mode (default) or single input: one branch per terminal
            sub_configs.append(_build_merge_branch_config(config, node_map, term, global_branch))
            global_branch += 1
    return sub_configs


def partition_parallel_pipeline(config: PipelineConfig) -> list[PipelineConfig]:
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
    node_map: dict[str, NodeConfig] = {n.node_id: n for n in config.nodes}

    # Identify terminal nodes (training, tuning, and data_preview leaves).
    terminals = [n for n in config.nodes if n.step_type in PARTITION_TERMINAL_STEP_TYPES]

    if not terminals:
        return [config]

    if len(terminals) == 1:
        return _partition_single_terminal(config, node_map, terminals[0])

    return _partition_multiple_terminals(config, node_map, terminals)


def _discover_ancestors_bfs(node_id: str, node_map: dict[str, NodeConfig]) -> set[str]:
    """BFS backwards through ``inputs`` edges from ``node_id`` to find the ancestor subgraph.

    We use a simple FIFO queue (list.pop(0)) because the graphs are tiny;
    collections.deque would only matter at much larger scale.
    """
    discovered: set[str] = set()
    queue = [node_id]
    while queue:
        nid = queue.pop(0)
        if nid in discovered or nid not in node_map:
            continue
        discovered.add(nid)
        queue.extend(
            parent_id
            for parent_id in node_map[nid].inputs
            if parent_id in node_map and parent_id not in discovered
        )
    return discovered


def _build_in_degree_and_children(
    discovered: set[str], node_map: dict[str, NodeConfig]
) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Build the in-degree map and children adjacency for Kahn's algorithm over ``discovered``."""
    in_degree: dict[str, int] = dict.fromkeys(discovered, 0)
    children: dict[str, list[str]] = {nid: [] for nid in discovered}
    for nid in discovered:
        for parent_id in node_map[nid].inputs:
            if parent_id in discovered:
                in_degree[nid] += 1
                children[parent_id].append(nid)
    return in_degree, children


def _kahn_topological_order(
    discovered: set[str], in_degree: dict[str, int], children: dict[str, list[str]]
) -> list[str]:
    """Repeatedly emit zero-in-degree nodes (Kahn's algorithm) to produce a topological order."""
    result: list[str] = []
    ready = [nid for nid, deg in in_degree.items() if deg == 0]
    while ready:
        nid = ready.pop(0)
        result.append(nid)
        for child in children[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
    return result


def _collect_ancestors(node_id: str, node_map: dict[str, NodeConfig]) -> list[str]:
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

    discovered = _discover_ancestors_bfs(node_id, node_map)
    in_degree, children = _build_in_degree_and_children(discovered, node_map)
    result = _kahn_topological_order(discovered, in_degree, children)

    if len(result) != len(discovered):
        # Cycle in the ancestor subgraph — cannot topologically sort. Fall
        # back to BFS discovery order so the caller gets *something*; the
        # engine will surface a clearer error when execution actually fails.
        logger.warning(
            "Cycle detected while collecting ancestors of %s; falling back to BFS-reverse order.",
            node_id,
        )
        return list(reversed(list(discovered)))

    return result


def _get_strategy_from_params(params: dict[str, Any]) -> str | None:
    if "tuning_config" in params and "strategy" in params["tuning_config"]:
        return cast(str, params["tuning_config"]["strategy"])
    if "tuning" in params and "strategy" in params["tuning"]:
        return cast(str, params["tuning"]["strategy"])
    if "search_strategy" in params:
        return cast(str, params["search_strategy"])
    if "strategy" in params:
        return cast(str, params["strategy"])
    return None


def extract_tuning_strategy(node_data: dict[str, Any]) -> str | None:
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


def _find_strategy_in_nodes(nodes: list[dict[str, Any]], node_id: str | None = None) -> str | None:
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


def determine_search_strategy(graph: dict[str, Any], node_id: str) -> str:
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


def _parse_node_info(node: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
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


from backend.ml_pipeline.constants import StepType  # noqa: E402


def _extract_columns(ntype: str, params: dict[str, Any]) -> list[str]:
    dropped: list[str] = []
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
    graph: dict[str, Any], node_id: str
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Extracts hyperparameters, target_column, and dropped_columns from graph."""
    hyperparameters = None
    target_column = None
    dropped_columns: list[str] = []

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
            StepType.TRAINING,
        ] and params.get("target_column"):
            target_column = params.get("target_column")

        dropped_columns.extend(_extract_columns(ntype, params))

    return hyperparameters, target_column, dropped_columns


@lru_cache(maxsize=1)
def model_registry_tags() -> dict[str, list[str]]:
    """Cache `{model_type_id: tags}` for every skyulf-core registered model node.

    Imported lazily to avoid a module-load cycle with the meta router.
    """
    from backend.ml_pipeline._internal._routers.meta import _build_node_registry

    return {item.id: item.tags for item in _build_node_registry()}


def resolve_model_family(
    model_type: str | None, registry_tags: dict[str, list[str]] | None = None
) -> str | None:
    """Resolve a `model_type` id to its display family (single source of truth).

    Mirrors the frontend's `getTaskForModelType` (jobMeta.ts) precedence, so
    Slow Nodes, the Jobs list, and any other surface resolving a model's
    family always agree. Returns `None` when `model_type` is missing or not
    resolvable via the registry's tags (e.g. an unregistered/unknown id).
    """
    if not model_type:
        return None
    if model_type in ENSEMBLE_MODEL_TYPES:
        return "ensemble"
    tags = (registry_tags if registry_tags is not None else model_registry_tags()).get(
        model_type, []
    )
    if "clustering" in tags:
        return "segmentation"
    if model_type == "logistic_regression":
        return "classification"
    if "text" in tags or "nlp" in tags:
        return "text_classification"
    if "classification" in tags:
        return "classification"
    if "regression" in tags:
        return "regression"
    return None


def resolve_training_model_type(
    graph: dict[str, Any] | None, node_id: str, fallback_model_type: str | None
) -> str | None:
    """Resolve the `algorithm`/`model_type` actually used by `node_id` in a stored graph.

    Prefers the matching node's own params (the graph reflects exactly what
    ran), falling back to `fallback_model_type` (typically the job row's own
    `model_type` column) when the node can't be found or carries no such
    param — covering both new runs and historical rows.
    """
    if isinstance(graph, dict):
        for node in graph.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            nid, _, params = _parse_node_info(node)
            if nid != node_id:
                continue
            candidate = params.get("algorithm") or params.get("model_type")
            if isinstance(candidate, str) and candidate:
                return candidate
            break
    return fallback_model_type if isinstance(fallback_model_type, str) else None


def resolve_training_model_family(
    graph: dict[str, Any] | None, node_id: str, fallback_model_type: str | None
) -> str | None:
    """Resolve a training job's real model family from its stored graph + fallback model_type.

    Combines `resolve_training_model_type` (which `model_type` actually
    ran) with `resolve_model_family` (which family that `model_type` belongs
    to) — the single code path every surface (Slow Nodes, Jobs list) should
    use so they can never disagree on a job's family.
    """
    model_type = resolve_training_model_type(graph, node_id, fallback_model_type)
    return resolve_model_family(model_type)
