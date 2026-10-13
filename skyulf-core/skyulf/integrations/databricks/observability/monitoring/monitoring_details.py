"""Read-only investigation and concrete model lineage from existing stored evidence."""

from ...shared._contracts import table_name
from .monitoring_config import qualified_name


def observation_identity(evidence: dict) -> dict:
    """Keep optional metadata refreshes outside immutable observation/reference identity."""
    optional = {"model_metadata", "monitoring_experiment_id", "monitoring_link_status"}
    return {key: value for key, value in evidence.items() if key not in optional}


def _model_link() -> str:
    """Link a concrete UC version rather than a mutable alias selection."""
    return (
        "CASE WHEN model_version IS NOT NULL THEN concat('/explore/data/models/', "
        "model_catalog, '/', model_schema, '/', "
        "substr(model_name, length(model_catalog) + length(model_schema) + 3), "
        "'/version/', model_version) END"
    )


def investigation_views(namespace: str) -> dict[str, str]:
    """Reuse owned reports, enrollment and health without adding a metadata table."""
    objects = {
        name: table_name(qualified_name(f"{namespace}.{name}"))
        for name in (
            "current_health",
            "model_inventory",
            "monitoring_results",
            "metric_history",
            "performance_history",
        )
    }
    return {
        "monitoring_issues": _issues_query(objects),
        "model_version_details": _details_query(objects),
    }


def _issues_query(objects: dict[str, str]) -> str:
    """Keep latest configuration/version policy evidence distinct from feature health."""
    return f"""
        WITH policy_ranked AS (
            SELECT p.*, row_number() OVER (
                PARTITION BY p.monitor_id, p.config_digest, p.model_version
                ORDER BY p.window_end DESC, p.measured_at DESC, p.report_id DESC
            ) AS report_rank FROM {objects["performance_history"]} p
        ), policy_evidence AS (
            SELECT h.*, p.report_id AS policy_report_id, p.measured_at AS policy_measured_at,
                p.status AS policy_status, p.reason AS policy_reason,
                p.metric, p.current_value, p.threshold_value,
                CASE WHEN p.report_id IS NULL THEN 'no_matching_report'
                    WHEN p.window_end IS NULL THEN 'missing_performance_window'
                    WHEN p.measured_at IS NULL THEN 'missing_measurement_time'
                    WHEN unix_timestamp(p.measured_at) > unix_timestamp(current_timestamp())
                        THEN 'future_measurement'
                    WHEN p.window_end IS NOT NULL AND (
                        get_json_object(i.config_json, '$.performance_policy.window_hours') IS NULL
                        OR get_json_object(i.config_json, '$.performance_policy.label_delay_hours') IS NULL
                    ) THEN 'missing_policy_window_configuration'
                    WHEN unix_timestamp(p.window_end) >
                        floor((unix_timestamp(current_timestamp()) -
                            CAST(get_json_object(i.config_json, '$.performance_policy.label_delay_hours') AS DOUBLE) * 3600)
                            / NULLIF(CAST(get_json_object(i.config_json, '$.performance_policy.window_hours') AS DOUBLE) * 3600, 0))
                            * CAST(get_json_object(i.config_json, '$.performance_policy.window_hours') AS DOUBLE) * 3600
                        THEN 'performance_window_not_mature'
                    WHEN floor((unix_timestamp(current_timestamp()) -
                            CAST(get_json_object(i.config_json, '$.performance_policy.label_delay_hours') AS DOUBLE) * 3600)
                            / NULLIF(CAST(get_json_object(i.config_json, '$.performance_policy.window_hours') AS DOUBLE) * 3600, 0))
                            * CAST(get_json_object(i.config_json, '$.performance_policy.window_hours') AS DOUBLE) * 3600
                            - unix_timestamp(p.window_end) > i.expected_interval_hours * 3600
                        THEN 'stale_performance_window'
                    ELSE NULL END AS policy_evidence_issue
            FROM {objects["current_health"]} h JOIN {objects["model_inventory"]} i
                ON h.monitor_id = i.monitor_id
            LEFT JOIN policy_ranked p ON p.monitor_id = i.monitor_id
                AND p.config_digest = i.config_digest AND p.model_version = h.model_version
                AND p.report_rank = 1
            WHERE i.enabled AND COALESCE(
                get_json_object(i.config_json, '$.performance_policy.mode'), 'off') <> 'off'
        ), issues AS (
            SELECT h.monitor_id, h.environment, h.project, h.model_catalog, h.model_schema,
                h.model_name, h.model_version, 'health' AS source, h.health_status AS status,
                COALESCE(h.error_message, h.health_status) AS reason,
                h.measured_at, h.observed_at, h.report_id,
                CAST(NULL AS STRING) AS metric_name, CAST(NULL AS STRING) AS column_name,
                CAST(NULL AS DOUBLE) AS value, CAST(NULL AS DOUBLE) AS threshold,
                CASE WHEN h.health_status = 'failed' THEN 1
                    WHEN h.health_status IN ('drift', 'degraded') THEN 2 ELSE 3 END AS priority
            FROM {objects["current_health"]} h
            WHERE h.health_status NOT IN ('healthy', 'disabled')
            UNION ALL
            SELECT h.monitor_id, h.environment, h.project, h.model_catalog, h.model_schema,
                h.model_name, h.model_version, m.category, m.status,
                CASE WHEN m.status = 'unavailable' THEN 'measurement_unavailable'
                    ELSE 'recorded_threshold_issue' END,
                h.measured_at, h.observed_at, h.report_id,
                m.metric_name, m.column_name, m.value, m.threshold, 2
            FROM {objects["current_health"]} h JOIN {objects["metric_history"]} m
                ON h.report_id = m.report_id AND h.monitor_id = m.monitor_id
                AND h.model_version = m.model_version
            WHERE (m.has_issue OR m.status = 'unavailable') AND m.category IN ('quality', 'drift')
                AND h.health_status <> 'disabled'
            UNION ALL
            SELECT monitor_id, environment, project, model_catalog, model_schema,
                model_name, model_version, 'performance',
                CASE WHEN policy_evidence_issue IS NOT NULL THEN 'unavailable'
                    ELSE COALESCE(policy_status, 'unavailable') END,
                COALESCE(policy_evidence_issue, policy_reason, 'missing_policy_evidence'),
                policy_measured_at, CAST(NULL AS TIMESTAMP), policy_report_id,
                metric, CAST(NULL AS STRING), current_value, threshold_value,
                CASE WHEN policy_evidence_issue IS NULL AND policy_status = 'degraded'
                    THEN 2 ELSE 3 END
            FROM policy_evidence WHERE policy_evidence_issue IS NOT NULL
                OR COALESCE(policy_status, 'unavailable') NOT IN ('healthy', 'disabled')
        )
        SELECT issues.*, concat(issues.environment, ' / ', issues.project, ' / ',
            substr(issues.monitor_id, 1, 12)) AS monitor_context,
            r.mlflow_run_id,
            CASE WHEN r.mlflow_run_id IS NOT NULL AND
                get_json_object(r.evidence_json, '$.monitoring_experiment_id') IS NOT NULL
                THEN concat('/ml/experiments/',
                    get_json_object(r.evidence_json, '$.monitoring_experiment_id'),
                    '/runs/', r.mlflow_run_id) END AS evidence_run_link,
            {_model_link()} AS model_version_link
        FROM issues LEFT JOIN (
            SELECT report_id, mlflow_run_id, evidence_json FROM {objects["monitoring_results"]}
        ) r ON issues.report_id = r.report_id
    """


def _details_query(objects: dict[str, str]) -> str:
    """Never borrow metadata from another concrete version or monitoring context."""
    return f"""
        WITH identities AS (
            SELECT monitor_id, environment, project, model_catalog, model_schema,
                model_name, model_version FROM {objects["current_health"]}
            UNION
            SELECT monitor_id, environment, project, model_catalog, model_schema,
                model_name, model_version FROM {objects["monitoring_results"]}
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY monitor_id, model_name, model_version
                ORDER BY window_end DESC, measured_at DESC, report_id DESC
            ) AS report_rank FROM {objects["monitoring_results"]}
        ), evidence AS (
            SELECT i.*, r.report_id, r.measured_at,
                get_json_object(r.evidence_json, '$.training_run_id') AS training_run_id,
                get_json_object(r.evidence_json, '$.model_digest') AS model_digest,
                COALESCE(get_json_object(r.evidence_json, '$.model_metadata.status'),
                    CASE WHEN r.report_id IS NULL THEN 'never_observed'
                        ELSE 'not_recorded' END) AS metadata_status,
                get_json_object(r.evidence_json, '$.model_metadata.reason') AS metadata_reason,
                get_json_object(r.evidence_json, '$.model_metadata.captured_at') AS captured_at,
                get_json_object(r.evidence_json, '$.model_metadata.created_at') AS created_at,
                get_json_object(r.evidence_json, '$.model_metadata.aliases') AS aliases_at_capture,
                get_json_object(r.evidence_json, '$.model_metadata.algorithm') AS algorithm,
                get_json_object(r.evidence_json, '$.model_metadata.project_source_sha256')
                    AS project_source_sha256,
                get_json_object(r.evidence_json, '$.model_metadata.training_experiment_id')
                    AS training_experiment_id,
                get_json_object(r.evidence_json, '$.model_metadata.training_source_commit')
                    AS training_source_commit,
                get_json_object(r.evidence_json, '$.model_metadata.training') AS training_summary_json
            FROM identities i LEFT JOIN ranked r ON i.monitor_id = r.monitor_id
                AND i.model_name = r.model_name
                AND (i.model_version = r.model_version OR
                    (i.model_version IS NULL AND r.model_version IS NULL))
                AND r.report_rank = 1
        )
        SELECT *, concat(environment, ' / ', project, ' / ', substr(monitor_id, 1, 12))
            AS monitor_context, {_model_link()} AS model_version_link,
            CASE WHEN training_run_id IS NOT NULL AND training_experiment_id IS NOT NULL
                THEN concat('/ml/experiments/', training_experiment_id, '/runs/', training_run_id)
                END AS training_run_link
        FROM evidence
    """
