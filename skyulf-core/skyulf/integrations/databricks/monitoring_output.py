"""Render saved monitoring observations inside visible Databricks Job tasks."""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from html import escape
from typing import Any
from urllib.parse import urlsplit

from .job_output import output_table
from .monitoring_config import qualified_name


def _dashboard_link(url: str) -> str:
    """Link to a configured HTTPS dashboard without accepting active URL schemes."""
    try:
        parsed = urlsplit(url)
        valid = parsed.scheme == "https" and parsed.hostname and not parsed.username
    except ValueError:
        valid = False
    if not valid:
        return "<p>Set monitoring_dashboard_url to link to the shared dashboard.</p>"
    return (
        f'<p><a href="{escape(url, quote=True)}" target="_blank" '
        'rel="noopener noreferrer">Open monitoring dashboard</a></p>'
    )


def render_monitor_output(result: dict, dashboard_url: str = "") -> str:
    """Show execution status and navigation without treating drift as a task failure."""
    if result.get("status") == "queued":
        return (
            "<h2>Model monitoring queued</h2>"
            "<p>Monitoring runs in a separate job. Open its monitoring_report task "
            "(drift_report in older projects) "
            "for these results. The dashboard shows the latest completed calculations.</p>"
            + output_table(
                ("Monitoring job", "Monitoring run"),
                [(result.get("job_id", ""), result.get("run_id", ""))],
            )
            + _dashboard_link(dashboard_url)
        )
    rows = [(r.get("model_name", ""), r["status"]) for r in result.get("results", [])]
    summary = (
        output_table(("Model", "Monitoring result"), rows)
        if rows
        else (f"<p>{escape(result.get('status', 'completed').replace('_', ' '))}</p>")
    )
    return (
        "<h2>Model monitoring</h2>"
        "<p>Checks: feature drift, data quality and performance when actual targets are available. "
        "Open monitoring_report (drift_report in older projects) for this batch's "
        "drift and performance results.</p>" + summary + _dashboard_link(dashboard_url)
    )


def _window_literal(value: str) -> str:
    """Validate task-provided instants before placing them in a timestamp predicate."""
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None:
        raise ValueError("Monitoring report timestamps must include a timezone.")
    return instant.astimezone(UTC).isoformat()


def load_observation(spark: Any, reference: dict) -> dict:
    """Read the successful observation for exactly the monitor configuration and scored batch."""
    name = qualified_name(f"{reference['namespace']}.monitoring_results")
    for key in ("monitor_id", "config_digest"):
        if not re.fullmatch(r"[a-f0-9]{64}", reference.get(key, "")):
            raise ValueError("Monitoring report requires a validated observation reference.")
    start = _window_literal(reference["window_start"])
    end = _window_literal(reference["window_end"])
    predicate = (
        f"monitor_id = '{reference['monitor_id']}' "
        f"AND config_digest = '{reference['config_digest']}' "
        f"AND window_start = TIMESTAMP '{start}' AND window_end = TIMESTAMP '{end}' "
        "AND status <> 'failed'"
    )
    row = (
        spark.table(name)
        .where(predicate)
        .orderBy("measured_at", "report_id", ascending=False)
        .limit(1)
        .first()
    )
    if row is None:
        raise ValueError("The completed scoring batch has no saved monitoring observation.")
    result = row.asDict()
    if json.loads(result["report_json"]).get("performance"):
        from .performance_actions import load_performance_action  # noqa: PLC0415

        result["performance_action"] = (
            load_performance_action(spark, reference["namespace"], result["report_id"]) or {}
        )
    return result


def _metric_cell(metric: dict) -> str:
    """Display a metric with its own limit and retain unavailable values."""
    value, threshold = metric.get("value"), metric.get("threshold")
    if value is None:
        return "Unavailable"
    return f"{value:.4g} / {threshold:.4g}" if threshold is not None else f"{value:.4g}"


def _feature_result(metrics: dict[str, dict]) -> str:
    """Exclude diagnostic p-values from the feature's drift verdict."""
    checks = [m for key, m in metrics.items() if key != "ks_test_p_value"]
    if any(m.get("has_issue") for m in checks):
        return "Drift detected"
    if any(m.get("status") == "unavailable" for m in checks):
        return "Some checks unavailable"
    return "No drift detected"


def _reported_metric(metrics: dict[str, dict], name: str) -> dict:
    """Display categorical PSI in the shared PSI column using its recorded evidence."""
    if name == "psi":
        return metrics.get(name, metrics.get("psi_categorical", {}))
    return metrics.get(name, {})


def _performance_policy_section(report: dict, row: dict) -> str:
    """Show the persisted policy verdict without treating absent evidence as healthy."""
    evidence = report.get("performance")
    heading = "<h2>Performance policy evidence</h2>"
    if not isinstance(evidence, dict) or evidence.get("status") in {None, "off", "disabled"}:
        return heading + "<p>Disabled: no performance degradation policy was evaluated.</p>"
    status = evidence["status"]
    if status not in {"healthy", "degraded", "unavailable"}:
        status = "unavailable"
    status_label = status.capitalize()
    action = row.get("performance_action") or {}
    fields = (
        ("Report ID", row.get("report_id")),
        ("Model", row.get("model_name")),
        ("Version", row.get("model_version")),
        ("Status", status_label),
        ("Reason", evidence.get("reason")),
        ("Metric", evidence.get("metric")),
        ("Improvement direction", evidence.get("direction")),
        ("Baseline kind", evidence.get("baseline_kind")),
        ("Baseline reference", evidence.get("baseline_reference")),
        ("Baseline value", evidence.get("baseline_value")),
        ("Current value", evidence.get("current_value")),
        ("Absolute degradation", evidence.get("absolute_degradation")),
        ("Relative degradation", evidence.get("relative_degradation")),
        ("Tolerance", evidence.get("tolerance")),
        ("Tolerance mode", evidence.get("tolerance_mode")),
        ("Threshold value", evidence.get("threshold_value")),
        ("Labeled rows", evidence.get("labeled_rows")),
        ("Label coverage", evidence.get("label_coverage")),
        ("Window start", evidence.get("window_start")),
        ("Window end", evidence.get("window_end")),
        (
            "Failure streak",
            f"{evidence.get('consecutive_failures', 0)} / {evidence.get('required_windows', 0)}",
        ),
        ("Evaluated windows", json.dumps(evidence.get("evaluated_windows") or [], default=str)),
        ("Action", action.get("action", evidence.get("action", "No action recorded"))),
        ("Action reason", action.get("action_reason", evidence.get("action_reason"))),
        ("Request ID", action.get("request_id", evidence.get("request_id"))),
        ("Run ID", action.get("run_id", evidence.get("run_id"))),
    )
    return heading + output_table(("Evidence", "Saved value"), list(fields))


def _drift_status(report: dict) -> str:
    """Describe feature evidence independently of outcome metrics and policy eligibility."""
    checks = [
        metric
        for metric in report.get("metrics", [])
        if metric.get("category") == "drift" and metric.get("metric_name") != "ks_test_p_value"
    ]
    if any(metric.get("has_issue") for metric in checks):
        return "detected"
    if checks and all(metric.get("status") == "measured" for metric in checks):
        return "not_detected"
    return "unavailable"


def _performance_loss_status(report: dict) -> str:
    """Keep missing labels distinct from measured degradation and an absent policy."""
    evidence = report.get("performance") or {}
    status = evidence.get("status", "disabled")
    if status in {"off", "disabled"}:
        return "disabled"
    return status if status in {"healthy", "degraded", "unavailable"} else "unavailable"


def _monitoring_statuses(report: dict) -> dict[str, str]:
    """Expose independent observed drift, measured performance and policy loss states."""
    measured = any(
        metric.get("category") == "performance" and metric.get("value") is not None
        for metric in report.get("metrics", [])
    )
    return {
        "drift_status": _drift_status(report),
        "performance_status": "measured" if measured else "unavailable",
        "performance_loss_status": _performance_loss_status(report),
    }


def _observed_performance_section(report: dict) -> str:
    """Display saved target metrics even when no degradation policy is configured."""
    metrics = [
        metric for metric in report.get("metrics", []) if metric.get("category") == "performance"
    ]
    summary = output_table(
        ("Predictions", "Matched actual targets", "Label coverage"),
        [
            tuple(
                "Unavailable" if report.get(key) is None else report[key]
                for key in ("scored_rows", "labeled_rows", "label_coverage")
            )
        ],
    )
    detail = (
        output_table(
            ("Target", "Metric", "Observed value", "Measurement status"),
            [
                (
                    metric["column_name"],
                    metric["metric_name"],
                    _metric_cell(metric),
                    metric.get("status", "unavailable"),
                )
                for metric in metrics
            ],
        )
        if metrics
        else "<p>No saved performance metrics are available for this observation.</p>"
    )
    return (
        "<h2>Observed performance</h2>"
        "<p>Metrics use predictions matched to eligible actual targets. Missing labels are "
        "unavailable evidence, not performance loss. Policy windows below may cover a "
        "different population from this scoring observation.</p>" + summary + detail
    )


def render_drift_output(row: dict, dashboard_url: str = "") -> str:
    """Render saved drift and performance evidence through the compatible report API."""
    features: dict[str, dict] = {}
    report = json.loads(row["report_json"])
    for metric in report.get("metrics", []):
        if metric["category"] == "drift":
            features.setdefault(metric["column_name"], {})[metric["metric_name"]] = metric
    names = (
        "psi",
        "ks_statistic",
        "wasserstein_distance",
        "kl_divergence",
        "schema_missing",
        "type_drift",
    )
    rows = [
        (
            feature,
            _feature_result(metrics),
            *(_metric_cell(_reported_metric(metrics, name)) for name in names),
        )
        for feature, metrics in sorted(features.items())
    ]
    summary = output_table(
        ("Model", "Version", "Scoring time", "Monitoring time", "Features with drift"),
        [
            (
                row["model_name"],
                row["model_version"],
                row["observed_at"],
                row["measured_at"],
                row["drifted_columns"],
            )
        ],
    )
    detail = (
        output_table(
            (
                "Feature",
                "Result",
                "PSI / limit",
                "KS / limit",
                "Wasserstein / limit",
                "KL / limit",
                "Missing column",
                "Changed type",
            ),
            rows,
        )
        if rows
        else "<p>No feature drift metrics are available for this batch.</p>"
    )
    return (
        "<h2>Monitoring report</h2><p>Saved results for this scoring batch. Each metric has its own limit. "
        "KS p-value is diagnostic only. Timestamps use the Spark session timezone.</p>"
        + summary
        + output_table(
            ("Feature drift", "Observed performance", "Performance loss policy"),
            [tuple(_monitoring_statuses(report).values())],
        )
        + "<h2>Feature drift</h2>"
        + detail
        + _observed_performance_section(report)
        + _performance_policy_section(report, row)
        + _dashboard_link(dashboard_url)
    )


def _run_report_notebook(
    spark: Any,
    dbutils: Any,
    *,
    display_html: Callable[[str], Any] | None,
    independent_statuses: bool,
) -> dict:
    """Display the upstream monitor's saved batch and return a small task result."""
    reference = dbutils.jobs.taskValues.get(taskKey="monitor_model", key="monitoring_reference")
    if not isinstance(reference, dict) or reference.get("status") not in {
        "ready",
        "disabled",
        "no_new_predictions",
        "queued",
    }:
        raise ValueError("Drift report requires a completed monitoring task reference.")
    url = dbutils.widgets.getAll().get("monitoring_dashboard_url", "")
    if reference["status"] != "ready":
        result = dict(reference)
        html = render_monitor_output(result, url)
    else:
        rows = [
            load_observation(spark, item) for item in reference.get("observations", [reference])
        ]
        results = [_report_result(row, independent_statuses) for row in rows]
        result = results[0] if len(results) == 1 else {"results": results}
        html = "".join(render_drift_output(row, url) for row in rows)
    if display_html is not None:
        display_html(html)
    return result


def _report_result(row: dict, independent_statuses: bool) -> dict:
    """Extend new notebook results while preserving the legacy drift task result shape."""
    result = {key: row[key] for key in ("report_id", "status", "drifted_columns")}
    if independent_statuses:
        result.update(_monitoring_statuses(json.loads(row["report_json"])))
    return result


def run_monitoring_report_notebook(
    spark: Any, dbutils: Any, *, display_html: Callable[[str], Any] | None = None
) -> dict:
    """Display independent drift and performance states for the exact saved observation."""
    return _run_report_notebook(
        spark, dbutils, display_html=display_html, independent_statuses=True
    )


def run_drift_report_notebook(
    spark: Any, dbutils: Any, *, display_html: Callable[[str], Any] | None = None
) -> dict:
    """Keep the legacy notebook entrypoint and compact result compatible."""
    return _run_report_notebook(
        spark, dbutils, display_html=display_html, independent_statuses=False
    )
