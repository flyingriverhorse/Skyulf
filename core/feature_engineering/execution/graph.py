"""Graph traversal and manipulation utilities for feature engineering pipelines."""

import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

DATASET_NODE_ID = "dataset-source"


def generate_pipeline_id(dataset_source_id: str, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    """Generate a stable pipeline ID from dataset and graph structure.

    Creates a unique identifier that combines the dataset source with a hash
    of the pipeline graph. Same graph = same ID, different graph = different ID.
    This ensures transformers are isolated across different pipeline versions.

    Args:
        dataset_source_id: The source dataset identifier
        nodes: List of node dictionaries from the graph
        edges: List of edge dictionaries from the graph

    Returns:
        Pipeline ID in format: {dataset_source_id}_{hash}
    """
    # Create a stable representation of the graph for hashing
    # We want to capture: node IDs, types, configs, and connections
    graph_representation = {
        "nodes": [
            {
                "id": node.get("id"),
                "type": node.get("type"),
                "catalogType": node.get("data", {}).get("catalogType"),
                "config": node.get("data", {}).get("config"),
            }
            for node in sorted(nodes, key=lambda n: n.get("id", ""))
        ],
        "edges": [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "sourceHandle": edge.get("sourceHandle"),
                "targetHandle": edge.get("targetHandle"),
            }
            for edge in sorted(edges, key=lambda e: (e.get("source", ""), e.get("target", "")))
        ],
    }

    # Serialize to JSON with sorted keys for consistency
    graph_json = json.dumps(graph_representation, sort_keys=True, separators=(',', ':'))

    # Hash the graph structure
    graph_hash = hashlib.sha256(graph_json.encode('utf-8')).hexdigest()[:8]

    # Combine dataset ID with graph hash
    pipeline_id = f"{dataset_source_id}_{graph_hash}"

    return pipeline_id


def resolve_catalog_type(node: Dict[str, Any]) -> str:
    data = node.get("data") or {}
    if isinstance(data, dict):
        catalog_type = data.get("catalogType")
        if catalog_type:
            return str(catalog_type)
    node_type = node.get("type")
    return str(node_type) if node_type else ""


def resolve_node_label(node: Dict[str, Any]) -> str:
    data = node.get("data") or {}
    if isinstance(data, dict):
        label = data.get("label")
        if label:
            return str(label)
    label = node.get("label")
    if isinstance(label, str) and label.strip():
        return label
    return str(node.get("id") or "node")


def _build_predecessor_map(edges: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    predecessors: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source and target:
            predecessors[target].append(source)
    return predecessors


def _build_successor_map(edges: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    successors: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source and target:
            successors[source].append(target)
    return successors


def execution_order(
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, Any]],
    target_node_id: Optional[str] = None,
) -> List[str]:
    node_ids = set(nodes.keys())
    predecessors = _build_predecessor_map(edges)
    successors = _build_successor_map(edges)

    visited: Dict[str, bool] = {}
    order: List[str] = []

    def dfs(node_id: str) -> None:
        if node_id in visited:
            return
        visited[node_id] = True
        for pred in predecessors[node_id]:
            dfs(pred)
        order.append(node_id)

    if target_node_id:
        if target_node_id in node_ids:
            dfs(target_node_id)
    else:
        for node_id in node_ids:
            if not successors[node_id]:  # Leaf nodes
                dfs(node_id)

    reachable: Set[str] = set()
    stack: List[str] = [DATASET_NODE_ID]
    while stack:
        curr = stack.pop()
        if curr in reachable:
            continue
        reachable.add(curr)
        for succ in successors[curr]:
            stack.append(succ)

    ordered = [node_id for node_id in order if node_id in reachable]
    if target_node_id and target_node_id in ordered:
        # Ensure target is last if specified
        ordered.remove(target_node_id)
        ordered.append(target_node_id)
    return ordered


def sanitize_graph_nodes(raw_nodes: Any) -> Dict[str, Dict[str, Any]]:
    """Convert raw nodes list to a dictionary keyed by node ID."""
    node_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_nodes, list):
        for node in raw_nodes:
            if isinstance(node, dict):
                node_id = node.get("id")
                if node_id:
                    # Ensure data dict exists
                    if "data" not in node or not isinstance(node["data"], dict):
                        node["data"] = {}
                    node_map[str(node_id)] = node
    return node_map


def sanitize_graph_edges(raw_edges: Any) -> List[Dict[str, Any]]:
    """Ensure edges are a list of dictionaries."""
    if isinstance(raw_edges, list):
        return [e for e in raw_edges if isinstance(e, dict)]
    return []


def determine_node_split_type(
    node_id: str,
    edges: List[Dict[str, Any]],
    node_map: Dict[str, Dict[str, Any]],
    visited: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Determines which split type (train/test/validation) a node should process
    based on which output handle of a train_test_split node it's connected to.

    Returns: 'train', 'test', 'validation', or None
    """
    # Prevent infinite recursion by tracking visited nodes
    if visited is None:
        visited = set()

    if node_id in visited:
        return None

    visited.add(node_id)

    # Find incoming edges to this node
    incoming_edges = [e for e in edges if e.get("target") == node_id]

    for edge in incoming_edges:
        source_id = edge.get("source")
        source_handle = edge.get("sourceHandle")
        
        if not source_id:
            continue

        # Check if source is a split node
        source_node = node_map.get(source_id)
        if source_node:
            catalog_type = resolve_catalog_type(source_node)
            if catalog_type == "train_test_split":
                # Map handle IDs to split types
                if source_handle == "train":
                    return "train"
                elif source_handle == "test":
                    return "test"
                elif source_handle == "validation":
                    return "validation"
        
        # Recursively check upstream
        upstream_split = determine_node_split_type(source_id, edges, node_map, visited)
        if upstream_split:
            return upstream_split

    return None


def ensure_dataset_node(node_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if DATASET_NODE_ID not in node_map:
        node_map[DATASET_NODE_ID] = {
            "id": DATASET_NODE_ID,
            "type": "datasetNode",
            "data": {
                "label": "Dataset Source",
                "catalogType": "dataset-source",
            },
        }
    return node_map


def extract_graph_payload(graph: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    if not graph:
        return {}, []

    if not isinstance(graph, dict):
        return {}, []

    node_map = sanitize_graph_nodes(graph.get("nodes"))
    graph_edges = sanitize_graph_edges(graph.get("edges"))
    return node_map, graph_edges


def normalize_target_node(target_node_id: Optional[str]) -> Optional[str]:
    if isinstance(target_node_id, str):
        normalized = target_node_id.strip()
        if normalized:
            return normalized
    return None
