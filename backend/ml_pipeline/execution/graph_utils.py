from typing import Any, Dict, List, Optional, Tuple, cast


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
