"""Shared SQL rules for dashboard identities, full observations and serving windows."""

from ...shared._contracts import table_name
from .monitoring_config import qualified_name

FULL_OBSERVATION = """NOT (status = 'no_data'
 AND get_json_object(report_json, '$.performance') IS NOT NULL
 AND get_json_object(report_json, '$.current_rows') IS NULL)"""


def complete_observations_query(namespace: str) -> str:
    """Exclude policy-only reports without changing SQL's three-valued NULL behavior."""
    results = table_name(qualified_name(f"{namespace}.monitoring_results"))
    return f"SELECT * FROM {results} WHERE {FULL_OBSERVATION}"


def dashboard_views(namespace: str) -> dict[str, str]:
    """Create identities after current health and serving windows after full observations."""
    return {
        "monitoring_identities": _identities_query(namespace),
        "serving_windows": _serving_windows_query(namespace),
    }


def _identities_query(namespace: str) -> str:
    """Keep unresolved and policy-only identities, but count only full measurement times."""
    health = table_name(qualified_name(f"{namespace}.current_health"))
    results = table_name(qualified_name(f"{namespace}.monitoring_results"))
    return f"""
WITH identities AS (
 SELECT monitor_id, model_catalog, model_schema, model_name, model_version,
 environment, project, measured_at FROM {health}
 UNION ALL
 SELECT monitor_id, model_catalog, model_schema, model_name, model_version,
 environment, project, CASE WHEN {FULL_OBSERVATION} THEN measured_at END AS measured_at FROM {results}
)
SELECT monitor_id, model_catalog, model_schema, model_name, model_version,
 concat(environment, ' / ', project, ' / ', substring(monitor_id, 1, 12)) AS monitor_context,
 max(measured_at) AS latest_measurement
FROM identities GROUP BY monitor_id, model_catalog, model_schema, model_name,
 model_version, environment, project
    """


def _serving_windows_query(namespace: str) -> str:
    """Read enabled current-config windows; rank independently per monitor and version."""
    inventory = table_name(qualified_name(f"{namespace}.model_inventory"))
    observations = table_name(qualified_name(f"{namespace}.complete_observations"))
    return f"""
WITH windows AS (
WITH serving_inventory AS (
 SELECT monitor_id, model_name, enabled, updated_at, config_digest,
 concat(environment, ' / ', project, ' / ', substring(monitor_id, 1, 12)) AS monitor_context,
 get_json_object(config_json, '$.model_version') AS model_version,
 get_json_object(config_json, '$.serving_endpoint') AS endpoint,
 COALESCE(get_json_object(config_json, '$.model_set_name'), model_name) AS served_model,
 COALESCE(get_json_object(config_json, '$.model_set_version'), get_json_object(config_json, '$.model_version')) AS served_version,
 get_json_object(config_json, '$.model_set_branch') AS served_branch
 FROM {inventory} WHERE enabled = true
 AND get_json_object(config_json, '$.serving_endpoint') IS NOT NULL
)
SELECT i.monitor_id, i.model_name, COALESCE(r.model_version, i.model_version) AS model_version,
 i.monitor_context, i.endpoint, i.served_model, COALESCE(i.served_version, r.model_version) AS served_version,
 i.served_branch, r.report_id, r.window_start, r.window_end, r.measured_at,
 COALESCE(r.status, 'awaiting_logs') AS status,
 CAST(get_json_object(r.report_json, '$.serving.request_count') AS BIGINT) AS request_count,
 CAST(get_json_object(r.report_json, '$.serving.success_count') AS BIGINT) AS success_count,
 CAST(get_json_object(r.report_json, '$.serving.error_count') AS BIGINT) AS error_count,
 CAST(get_json_object(r.report_json, '$.serving.logging_error_count') AS BIGINT) AS logging_error_count,
 CAST(get_json_object(r.report_json, '$.serving.status_4xx_count') AS BIGINT) AS status_4xx_count,
 CAST(get_json_object(r.report_json, '$.serving.status_5xx_count') AS BIGINT) AS status_5xx_count,
 CAST(get_json_object(r.report_json, '$.serving.status_other_count') AS BIGINT) AS status_other_count,
 CAST(get_json_object(r.report_json, '$.serving.latency_sample_count') AS BIGINT) AS latency_sample_count,
 CAST(get_json_object(r.report_json, '$.serving.latency_p50_ms') AS DOUBLE) AS latency_p50_ms,
 CAST(get_json_object(r.report_json, '$.serving.latency_p95_ms') AS DOUBLE) AS latency_p95_ms,
 CAST(get_json_object(r.report_json, '$.serving.latency_p99_ms') AS DOUBLE) AS latency_p99_ms,
 get_json_object(r.report_json, '$.serving.latency_bucket_edges_ms') AS latency_bucket_edges_ms,
 get_json_object(r.report_json, '$.serving.latency_bucket_counts') AS latency_bucket_counts,
 CAST(get_json_object(r.report_json, '$.serving.latency_mean_ms') AS DOUBLE) AS latency_mean_ms,
 CAST(get_json_object(r.report_json, '$.serving.latency_max_ms') AS DOUBLE) AS latency_max_ms,
 CAST(get_json_object(r.report_json, '$.serving.prediction_row_count') AS BIGINT) AS prediction_row_count
FROM {observations} r JOIN serving_inventory i ON r.monitor_id = i.monitor_id AND r.config_digest = i.config_digest
), classified AS (
 SELECT *, CASE WHEN status_4xx_count >= 0 AND status_5xx_count >= 0 AND status_other_count >= 0
 AND status_4xx_count + status_5xx_count + status_other_count = error_count THEN 1 ELSE 0 END AS classes_complete
 FROM windows
)
SELECT *,
 CASE WHEN status <> 'failed' AND request_count > 0 AND error_count BETWEEN 0 AND request_count
 THEN error_count * 1.0 / request_count END AS error_rate,
 CASE WHEN status <> 'failed' AND request_count > 0 AND prediction_row_count >= 0
 THEN prediction_row_count * 1.0 / request_count END AS rows_per_request
 , CASE WHEN status <> 'failed' AND request_count > 0 AND error_count BETWEEN 0 AND request_count AND classes_complete = 1 THEN status_4xx_count END AS plot_4xx,
 CASE WHEN status <> 'failed' AND request_count > 0 AND error_count BETWEEN 0 AND request_count AND classes_complete = 1 THEN status_5xx_count END AS plot_5xx,
 CASE WHEN status <> 'failed' AND request_count > 0 AND error_count BETWEEN 0 AND request_count THEN CASE WHEN classes_complete = 1 THEN status_other_count ELSE error_count END END AS plot_other,
 ROW_NUMBER() OVER (PARTITION BY monitor_id, model_version
 ORDER BY window_end, measured_at, report_id) AS observation_sequence,
 ROW_NUMBER() OVER (PARTITION BY monitor_id, model_version
 ORDER BY window_end DESC, measured_at DESC, report_id DESC) AS observation_recency
FROM classified
    """
