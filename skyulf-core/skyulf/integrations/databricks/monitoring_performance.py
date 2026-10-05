"""Measure independent, mature performance windows against a pinned model baseline."""

import json
from datetime import UTC, datetime
from typing import Any

from ...inference.local_scoring import score_local_pipeline
from ._contracts import table_name
from .monitoring_config import MonitorConfig, json_digest, qualified_name
from .monitoring_metrics import build_performance_report
from .monitoring_sources import read_current_observation, read_labels
from .performance_policy import (
    completed_performance_window,
    evaluate_performance,
    validate_performance_policy,
)


def performance_contract(artifact: Any, spec: Any, config: MonitorConfig) -> str:
    """Bind metric definitions, classes, fitted rules and eligible production populations."""
    policy = validate_performance_policy(config.performance_policy)
    return json_digest(
        {
            "definition": "skyulf.core.performance.v1.unweighted.saved_predictions",
            "task": artifact.manifest.task,
            "classes": list(artifact.manifest.classes),
            "model_digest": artifact.manifest.pipeline_sha256,
            "target": spec.target_column,
            "keys": list(spec.record_key_columns),
            "source": config.source_table,
            "labels": config.label_table,
            "available": config.result_available_at_column,
            "window_hours": policy.get("window_hours"),
            "label_delay_hours": policy.get("label_delay_hours"),
            "population": "predicted_rows_with_finite_available_keyed_labels",
        }
    )


def _measure(
    predictions: Any, labels: Any, artifact: Any, spec: Any, config: MonitorConfig, as_of: datetime
) -> dict:
    """Call the shared saved-outcome evaluator without recomputing production predictions."""
    return build_performance_report(
        predictions,
        labels,
        record_key_columns=spec.record_key_columns,
        target_column=spec.target_column,
        result_available_at_column=str(config.result_available_at_column),
        as_of=as_of,
        task=artifact.manifest.task,
        classes=artifact.manifest.classes,
    )


def population_contract(metric_digest: str, evidence: dict) -> str:
    """Separate a replaced physical table from another version of the same population."""
    identity = {
        key: evidence[key] for key in ("source_table_id", "prediction_table_id", "label_table_id")
    }
    if any(not isinstance(value, str) or not value for value in identity.values()):
        raise ValueError("Performance requires concrete source, prediction and label table IDs.")
    return json_digest({"metric_contract": metric_digest, "population": identity})


def measure_holdout_baseline(
    artifact: Any, spec: Any, holdout: Any, config: MonitorConfig, version: str, run_id: str
) -> dict:
    """Replay verified holdout membership with the same fitted scoring and metric conventions."""
    features = holdout.loc[:, list(artifact.manifest.input_columns)].copy()
    predictions = score_local_pipeline(features, artifact).reset_index(drop=True)
    labels = holdout.loc[:, [spec.target_column]].reset_index(drop=True).copy()
    # Verified model partitions omit key metadata. Pair this one replay locally;
    # production measurements always retain the original stable source keys.
    key = "__skyulf_holdout_row"
    while key in labels.columns:
        key += "_"
    predictions[key] = range(len(holdout))
    labels[key] = range(len(holdout))
    # These are saved training outcomes, not fabricated production availability.
    cutoff = datetime(1970, 1, 1, tzinfo=UTC)
    labels[str(config.result_available_at_column)] = cutoff
    report = build_performance_report(
        predictions,
        labels,
        record_key_columns=(key,),
        target_column=spec.target_column,
        result_available_at_column=str(config.result_available_at_column),
        as_of=cutoff,
        task=artifact.manifest.task,
        classes=artifact.manifest.classes,
    )
    policy = validate_performance_policy(config.performance_policy)
    return {
        "model_version": version,
        "metric": policy["metric"],
        "contract_digest": performance_contract(artifact, spec, config),
        "value": report["values"].get(policy["metric"]),
        "labeled_rows": report["labeled_rows"],
        "label_coverage": report["label_coverage"],
        "reference": f"runs:/{run_id}/training_filter_evidence.json",
        "kind": "training_holdout",
    }


def load_performance_history(
    spark: Any, namespace: str, config: MonitorConfig, end: datetime, count: int
) -> list[dict]:
    """Read at most one latest observation per completed window, bounded in Spark."""
    name = table_name(qualified_name(f"{namespace}.monitoring_results"))
    digest = json_digest(config.payload())
    rows = spark.sql(f"""
        WITH observations AS (
            SELECT report_id, measured_at,
                get_json_object(report_json, '$.performance') AS performance,
                CAST(get_json_object(report_json, '$.performance.window_end') AS TIMESTAMP) AS ending
            FROM {name}
            WHERE monitor_id = '{config.monitor_id}' AND config_digest = '{digest}'
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY ending ORDER BY measured_at DESC, report_id DESC
            ) AS rank FROM observations
            WHERE ending < TIMESTAMP '{end.isoformat()}'
        ) SELECT performance FROM ranked WHERE rank = 1 ORDER BY ending DESC LIMIT {int(count)}
    """).collect()
    return [json.loads(row["performance"]) for row in rows]


def _production_baseline(
    spark: Any, namespace: str, config: MonitorConfig, policy: dict
) -> dict | None:
    """Read a named immutable observation without resolving aliases or choosing a best window."""
    name = qualified_name(f"{namespace}.monitoring_results")
    report_id = policy["baseline"]["report_id"]
    rows = (
        spark.table(name)
        .where(f"report_id = '{report_id}' AND monitor_id = '{config.monitor_id}'")
        .limit(2)
        .collect()
    )
    if len(rows) != 1 or rows[0]["status"] == "failed":
        return None
    saved = json.loads(rows[0]["report_json"]).get("performance", {}).get("measurement")
    return {**saved, "report_id": report_id} if saved else None


def observe_performance(
    spark: Any,
    namespace: str,
    config: MonitorConfig,
    artifact: Any,
    spec: Any,
    reference: dict,
    now: datetime,
) -> dict:
    """Retain an independent verdict and all snapshot evidence for one mature window."""
    policy = validate_performance_policy(config.performance_policy)
    if policy["mode"] == "off":
        return {"status": "disabled", "reason": "policy_off", "action": "none"}
    start, end = completed_performance_window(now, policy)
    manifest = artifact.manifest
    _, predictions, evidence, _ = read_current_observation(
        spark,
        config,
        reference["model_version"],
        spec.record_key_columns,
        manifest.input_columns,
        probabilities=len(manifest.classes) if manifest.classification_probabilities else 0,
        as_of=now,
        start=start,
        end=end,
    )
    labels, label_evidence = read_labels(
        spark, config, spec.record_key_columns, spec.target_column, predictions, now
    )
    report = _measure(predictions, labels, artifact, spec, config, now)
    metric_digest = performance_contract(artifact, spec, config)
    current = {
        "model_version": reference["model_version"],
        "metric": policy["metric"],
        "contract_digest": population_contract(metric_digest, evidence | label_evidence),
        "value": report["values"].get(policy["metric"]),
        "labeled_rows": report["labeled_rows"],
        "label_coverage": report["label_coverage"],
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "as_of": now.isoformat(),
        "evidence": evidence | label_evidence,
    }
    baseline = reference.get("performance_baseline")
    if policy["baseline"]["kind"] == "production_window":
        baseline = _production_baseline(spark, namespace, config, policy)
    elif baseline and baseline["contract_digest"] == metric_digest:
        # A verified training reference deliberately compares to the current
        # production population; production baselines must retain their own IDs.
        baseline = baseline | {"contract_digest": current["contract_digest"]}
    history = load_performance_history(spark, namespace, config, end, policy["consecutive_windows"])
    verdict = evaluate_performance(policy, current, baseline, history, now=now)
    return verdict | {
        "measurement": current,
        "model_version": reference["model_version"],
        "baseline_kind": policy["baseline"]["kind"],
        "baseline_reference": (baseline or {}).get(
            "reference", policy["baseline"].get("report_id")
        ),
        "required_windows": policy["consecutive_windows"],
    }


def observe_performance_safely(
    spark: Any,
    namespace: str | None,
    config: MonitorConfig,
    artifact: Any,
    spec: Any,
    reference: dict,
    now: datetime,
) -> dict:
    """Keep independent drift evidence when a performance baseline or snapshot is unavailable."""
    policy = validate_performance_policy(config.performance_policy)
    if policy["mode"] == "off":
        return {"status": "disabled", "reason": "policy_off", "action": "none"}
    try:
        if namespace is None:
            raise ValueError("Performance policy requires a monitoring namespace.")
        return observe_performance(spark, namespace, config, artifact, spec, reference, now)
    except Exception as error:  # noqa: BLE001 - persist independent external-read failures
        return unavailable_performance(config, now, str(error)[:500], reference["model_version"])


def unavailable_performance(
    config: MonitorConfig, now: datetime, reason: str, version: str | None
) -> dict:
    """Keep failed reads as explicit broken windows, never as disabled or healthy policies."""
    policy = validate_performance_policy(config.performance_policy)
    if policy["mode"] == "off":
        return {"status": "disabled", "reason": "policy_off", "action": "none"}
    start, end = completed_performance_window(now, policy)
    return {
        "status": "unavailable",
        "reason": reason,
        "action": "none",
        "policy_digest": json_digest(policy),
        "model_version": version,
        "metric": policy["metric"],
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "as_of": now.isoformat(),
        "consecutive_failures": 0,
    }
