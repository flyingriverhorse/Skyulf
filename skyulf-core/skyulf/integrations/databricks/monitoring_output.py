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
    rows = [(r.get("model_name", ""), r["status"]) for r in result.get("results", [])]
    summary = (
        output_table(("Model", "Monitoring result"), rows)
        if rows
        else (f"<p>{escape(result.get('status', 'completed').replace('_', ' '))}</p>")
    )
    return (
        "<h2>Model monitoring</h2>"
        "<p>Checks: feature drift, data quality and performance when actual targets are available. "
        "Open the drift_report task for this batch's feature results.</p>"
        + summary
        + _dashboard_link(dashboard_url)
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
    return row.asDict()


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


def render_drift_output(row: dict, dashboard_url: str = "") -> str:
    """Render a compact per-feature table from the persisted report, without recalculation."""
    features: dict[str, dict] = {}
    for metric in json.loads(row["report_json"]).get("metrics", []):
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
            *(_metric_cell(metrics.get(name, {})) for name in names),
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
        "<h2>Drift report</h2><p>Saved results for this scoring batch. Each metric has its own limit. "
        "KS p-value is diagnostic only. Timestamps use the Spark session timezone.</p>"
        + summary
        + detail
        + _dashboard_link(dashboard_url)
    )


def run_drift_report_notebook(
    spark: Any, dbutils: Any, *, display_html: Callable[[str], Any] | None = None
) -> dict:
    """Display the upstream monitor's saved batch and return a small task result."""
    reference = dbutils.jobs.taskValues.get(taskKey="monitor_model", key="monitoring_reference")
    if not isinstance(reference, dict) or reference.get("status") not in {
        "ready",
        "disabled",
        "no_new_predictions",
    }:
        raise ValueError("Drift report requires a completed monitoring task reference.")
    url = dbutils.widgets.getAll().get("monitoring_dashboard_url", "")
    if reference["status"] != "ready":
        result = {"status": reference["status"]}
        html = render_monitor_output(result, url)
    else:
        rows = [
            load_observation(spark, item) for item in reference.get("observations", [reference])
        ]
        results = [
            {key: row[key] for key in ("report_id", "status", "drifted_columns")} for row in rows
        ]
        result = results[0] if len(results) == 1 else {"results": results}
        html = "".join(render_drift_output(row, url) for row in rows)
    if display_html is not None:
        display_html(html)
    return result
