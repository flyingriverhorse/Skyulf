"""Statistics recommendation logic."""

import json
from typing import Any, Dict, List, Optional

from core.feature_engineering.execution.graph import resolve_catalog_type


def parse_skewness_transformations(raw_payload: Optional[str]) -> Dict[str, str]:
    if not raw_payload:
        return {}

    try:
        payload = json.loads(raw_payload)
    except (TypeError, ValueError):
        return {}

    parsed: Dict[str, str] = {}

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                col = item.get("column")
                method = item.get("method")
                if col and method:
                    parsed[col] = method
    elif isinstance(payload, dict):
        for col, method in payload.items():
            if isinstance(col, str) and isinstance(method, str):
                parsed[col] = method
    else:
        pass

    return parsed


def collect_skewness_transformations_from_graph(
    execution_order: List[str],
    node_map: Dict[str, Dict[str, Any]],
    target_node_id: Optional[str],
) -> Dict[str, str]:
    selections: Dict[str, str] = {}

    for node_id in execution_order:
        if node_id == target_node_id:
            break

        node = node_map.get(node_id)
        if not node:
            continue

        catalog_type = resolve_catalog_type(node)
        if catalog_type == "skewness_transform":
            config = node.get("data", {}).get("config", {})
            method = config.get("method")
            columns = config.get("columns", [])
            
            # If method is 'auto', we can't know the specific transform without running it.
            # But if it's explicit (log, sqrt, box-cox), we can track it.
            if method and columns:
                for col in columns:
                    selections[col] = method

    return selections
