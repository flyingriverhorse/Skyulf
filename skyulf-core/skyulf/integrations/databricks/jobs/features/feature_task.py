"""Run one visible feature job task with pinned cross-task source evidence."""

import json
import math
from pathlib import Path
from typing import Any

from ...features.config import parse_feature_config
from ...features.graph import plan_digest
from ...features.runtime import build_feature_group, initialize_features, merge_feature_groups
from ...projects.yaml_config import read_yaml_mapping


def _finite_number(value: str) -> float:
    """Reject nonstandard numbers and overflow before invoking any data operation."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Feature task JSON requires finite numbers.")
    return number


def _evidence(value: str) -> dict[str, Any]:
    """Require a bounded JSON object from an upstream task or repair parameter."""
    if len(value.encode("utf-8")) > 48 * 1024:
        raise ValueError("Feature input evidence exceeds the task-value limit.")
    result = json.loads(value, parse_float=_finite_number, parse_constant=_finite_number)
    if type(result) is not dict:
        raise ValueError("Feature task evidence must be a JSON object.")
    return result


def run_feature_notebook(spark: Any, dbutils: Any) -> dict[str, Any]:
    """Resolve declared widget inputs and publish bounded task evidence, not records."""
    get = dbutils.widgets.get
    config_path = Path(get("config_path"))
    plan = parse_feature_config(read_yaml_mapping(config_path))
    if plan is None:
        raise ValueError("Feature job is disabled; refresh the feature graph and redeploy.")
    if get("plan_sha256") != plan_digest(plan):
        raise ValueError("Feature graph is stale; run refresh_feature_graph.py and redeploy.")
    phase = get("phase")
    if phase == "initialize":
        result = initialize_features(spark, config_path.parent.parent, plan, get("selected_groups"))
        key = "snapshot_json"
    elif phase == "group":
        result = build_feature_group(
            spark, config_path.parent.parent, plan, _evidence(get("snapshot_json")), get("group")
        )
        key = "receipt_json"
    elif phase == "merge":
        receipts = {
            group.name: _evidence(get(f"receipt_{group.name}_json")) for group in plan.groups
        }
        result = merge_feature_groups(spark, plan, _evidence(get("snapshot_json")), receipts)
        key = "receipt_json"
    else:
        raise ValueError(f"Unknown feature task phase: {phase}.")
    payload = json.dumps(result, allow_nan=False)
    if len(payload.encode("utf-8")) > 48 * 1024:
        raise ValueError("Feature task evidence exceeds the Databricks task-value limit.")
    dbutils.jobs.taskValues.set(key=key, value=payload)
    print(json.dumps({"phase": phase, **result}, indent=2))
    return result
