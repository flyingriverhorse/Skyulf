"""Keep retraining decisions durable without rewriting immutable metric evidence."""

import re
from datetime import datetime
from typing import Any
from uuid import uuid4

from ....shared._contracts import table_name
from ..monitoring_config import json_digest, qualified_name
from ..monitoring_store import OWNER, PROPERTY, ensure_owned_object, merge_with_retry

ACTION_SCHEMA = (
    "action_id STRING, report_id STRING, recorded_at TIMESTAMP, action STRING, "
    "action_reason STRING, request_id STRING, run_id BIGINT, preview BOOLEAN"
)


def initialize_performance_actions(spark: Any, namespace: str) -> None:
    """Create an additive owned decision log without changing historical result schemas."""
    name = qualified_name(f"{namespace}.performance_actions")
    if ensure_owned_object(spark, name):
        expected = spark.createDataFrame([], schema=ACTION_SCHEMA).schema.simpleString()
        if spark.table(name).schema.simpleString() != expected:
            raise ValueError("Performance action table schema differs.")
        return
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {table_name(name)} ({ACTION_SCHEMA}) "
        f"USING DELTA TBLPROPERTIES ('{PROPERTY}' = '{OWNER}')"
    ).collect()
    ensure_owned_object(spark, name)


def _action(model: dict, result: dict, preview: bool) -> tuple[str, str]:
    """Distinguish actual submissions, advisory eligibility and precise guard skips."""
    if model["status"] == "ready":
        reason = result["status"]
    elif model.get("triggers"):
        reason = model["status"]
    else:
        reason = model.get("performance_reason", model["status"])
    if reason in {"submitted", "already_submitted"}:
        return "retraining_requested", reason
    if reason == "ready" and preview:
        return "request_eligible", reason
    if model.get("triggers"):
        return "retraining_skipped", reason
    return "none", reason


def record_performance_actions(
    spark: Any, namespace: str, result: dict, now: datetime, *, preview_only: bool = False
) -> None:
    """Record every opted-in model's exact guard outcome, including unchanged-data skips."""
    for model in result.get("models", []):
        if not model.get("performance_monitored"):
            continue
        action, reason = _action(model, result, preview_only)
        row = {
            "report_id": model["report_id"],
            "recorded_at": now,
            "action": action,
            "action_reason": reason,
            "request_id": result.get("request_id") if model["status"] == "ready" else None,
            "run_id": result.get("run_id") if model["status"] == "ready" else None,
            "preview": preview_only,
        }
        row["action_id"] = json_digest({**row, "recorded_at": now.isoformat()})
        persist_action(spark, namespace, row)


def persist_action(spark: Any, namespace: str, row: dict) -> None:
    """Append typed action evidence and preserve concurrent observations."""
    initialize_performance_actions(spark, namespace)
    name = table_name(qualified_name(f"{namespace}.performance_actions"))
    view = f"skyulf_performance_{uuid4().hex}"
    spark.createDataFrame([row], schema=ACTION_SCHEMA).createOrReplaceTempView(view)
    try:
        merge_with_retry(
            spark,
            f"MERGE INTO {name} t USING {view} s ON t.action_id = s.action_id "
            "WHEN NOT MATCHED THEN INSERT *",
        )
    finally:
        spark.catalog.dropTempView(view)


def load_performance_action(spark: Any, namespace: str, report_id: str) -> dict | None:
    """Enrich a report with its latest saved decision when the optional log exists."""
    name = qualified_name(f"{namespace}.performance_actions")
    if not re.fullmatch(r"[a-f0-9]{64}", report_id):
        raise ValueError("Performance action requires a report digest.")
    if not ensure_owned_object(spark, name):
        return None
    row = (
        spark.table(name)
        .where(f"report_id = '{report_id}'")
        .orderBy("preview", "recorded_at", "action_id", ascending=[True, False, False])
        .limit(1)
        .first()
    )
    return row.asDict() if row is not None else None


def performance_history_query(namespace: str) -> str:
    """Project saved metric and request evidence for the shared performance dashboard."""
    results = table_name(qualified_name(f"{namespace}.monitoring_results"))
    actions = table_name(qualified_name(f"{namespace}.performance_actions"))
    strings = (
        "reason",
        "metric",
        "direction",
        "baseline_kind",
        "baseline_reference",
        "tolerance_mode",
    )
    numbers = (
        "baseline_value",
        "current_value",
        "absolute_degradation",
        "relative_degradation",
        "tolerance",
        "threshold_value",
        "labeled_rows",
        "label_coverage",
        "consecutive_failures",
        "required_windows",
    )
    fields = [f"get_json_object(r.report_json, '$.performance.{key}') AS {key}" for key in strings]
    fields += [
        f"CAST(get_json_object(r.report_json, '$.performance.{key}') AS DOUBLE) AS {key}"
        for key in numbers
    ]
    # Dynamic table names are validated/quoted above; all projected keys are constants.
    return f"""
        WITH decisions AS (
            SELECT *, row_number() OVER (
                PARTITION BY report_id ORDER BY preview ASC, recorded_at DESC, action_id DESC
            ) AS rank FROM {actions}
        )
        SELECT r.report_id, r.monitor_id, r.config_digest, r.environment, r.project,
            r.model_name, r.model_version, r.model_catalog, r.model_schema,
            r.measured_at, r.observed_at,
            CAST(get_json_object(r.report_json, '$.performance.window_start') AS TIMESTAMP)
                AS window_start,
            CAST(get_json_object(r.report_json, '$.performance.window_end') AS TIMESTAMP)
                AS window_end,
            COALESCE(get_json_object(r.report_json, '$.performance.status'), 'disabled') AS status,
            {", ".join(fields)},
            COALESCE(a.action, get_json_object(r.report_json, '$.performance.action'), 'none') AS action,
            a.action_reason, a.request_id, a.run_id
        FROM {results} r LEFT JOIN decisions a ON r.report_id = a.report_id AND a.rank = 1
    """  # nosec B608 - validated identifiers and fixed projection keys only
