from typing import Any, Dict, Optional

from core.feature_engineering.execution.graph import (
    ensure_dataset_node,
    execution_order,
    extract_graph_payload,
    normalize_target_node,
    resolve_catalog_type,
)

def _collect_applied_skewness_methods(
    graph: Optional[Dict[str, Any]],
    target_node_id: Optional[str],
) -> Dict[str, str]:
    if not graph:
        return {}

    node_map, edges = extract_graph_payload(graph)
    if not node_map:
        return {}

    ensured_map = ensure_dataset_node(node_map)
    normalized_target = normalize_target_node(target_node_id)
    
    # Get execution order up to (and including) the target node
    # But apply_graph_transformations_before_node stops BEFORE the target.
    # So we want to collect methods from nodes that ARE executed.
    # If target is provided, execution_order returns path ending with target.
    # We should iterate up to target (exclusive).
    
    exec_order = execution_order(ensured_map, edges, normalized_target)
    
    applied_methods: Dict[str, str] = {}
    
    for node_id in exec_order:
        if node_id == normalized_target:
            break
            
        node = ensured_map.get(node_id)
        if not node:
            continue
            
        catalog_type = resolve_catalog_type(node)
        if catalog_type == "skewness_transform":
            data = node.get("data") or {}
            config = data.get("config") or {}
            transformations = config.get("transformations")
            
            # Parse transformations list
            if isinstance(transformations, list):
                for item in transformations:
                    if isinstance(item, dict):
                        col = item.get("column")
                        method = item.get("method")
                        if col and method:
                            applied_methods[str(col)] = str(method).lower()
                            
    return applied_methods


def _get_target_column_from_graph(graph: Optional[Dict[str, Any]]) -> Optional[str]:
    if not graph:
        return None
    
    node_map, _ = extract_graph_payload(graph)
    
    for node in node_map.values():
        if resolve_catalog_type(node) == "feature_target_split":
            config = node.get("data", {}).get("config", {})
            target = config.get("target_column") or config.get("target")
            if target and isinstance(target, str):
                return target
    return None
