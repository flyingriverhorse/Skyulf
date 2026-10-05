"""Observe enrolled models independently and publish durable shared monitoring results."""

import json
from datetime import UTC, datetime
from typing import Any

from ..mlflow._client import make_tracking_client
from ..mlflow.tracking import TrackingConfig, track_run
from .monitoring_config import MonitorConfig, qualified_name
from .monitoring_reference import load_monitoring_reference
from .monitoring_sources import observation_window, read_current_observation, read_labels
from .monitoring_store import (
    enroll_monitor,
    initialize_monitoring_store,
    load_enrolled_models,
    persist_report,
    result_row,
)


def observe_model(
    spark: Any,
    config: MonitorConfig,
    *,
    as_of: datetime,
    window_start: datetime,
    window_end: datetime,
    tracking_uri: str | None,
    registry_uri: str | None,
    experiment_name: str | None,
    namespace: str | None = None,
    performance_only: bool = False,
) -> dict:
    """Compare one pinned model and scored population, preserving missing-label semantics."""
    from .monitoring_metrics import build_monitoring_report  # noqa: PLC0415 - separate computation

    artifact, spec, reference, evidence = load_monitoring_reference(
        spark,
        config,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    if performance_only:
        return _performance_only_row(
            spark,
            namespace,
            config,
            artifact,
            spec,
            evidence,
            as_of,
            window_start,
            window_end,
            tracking_uri,
            experiment_name,
        )
    manifest = artifact.manifest
    version = evidence["model_version"]
    probabilities = len(manifest.classes) if manifest.classification_probabilities else 0
    current, predictions, observed_evidence, observed_at = read_current_observation(
        spark,
        config,
        version,
        spec.record_key_columns,
        manifest.input_columns,
        probabilities=probabilities,
        as_of=as_of,
        start=window_start,
        end=window_end,
    )
    labels, label_evidence = read_labels(
        spark,
        config,
        spec.record_key_columns,
        spec.target_column,
        predictions,
        as_of,
    )
    evidence |= observed_evidence | label_evidence | {"as_of": as_of.isoformat()}
    report = build_monitoring_report(
        reference,
        current,
        predictions,
        labels,
        feature_columns=manifest.input_columns,
        record_key_columns=spec.record_key_columns,
        target_column=spec.target_column,
        result_available_at_column=config.result_available_at_column or "result_available_at",
        as_of=as_of,
        task=manifest.task,
        classes=manifest.classes,
        thresholds=config.thresholds,
    )
    if config.performance_policy:
        from .monitoring_performance import observe_performance_safely  # noqa: PLC0415

        report["performance"] = observe_performance_safely(
            spark, namespace, config, artifact, spec, evidence, as_of
        )
        evidence["performance"] = report["performance"]
    row = result_row(
        config, version, as_of, window_start, window_end, report, evidence, observed_at=observed_at
    )
    row["mlflow_run_id"] = _log_observation(
        config, row, report, evidence, tracking_uri, experiment_name
    )
    return row


def _performance_only_row(
    spark: Any,
    namespace: str | None,
    config: MonitorConfig,
    artifact: Any,
    spec: Any,
    evidence: dict,
    as_of: datetime,
    start: datetime,
    end: datetime,
    tracking_uri: str | None,
    experiment_name: str | None,
) -> dict:
    """Revisit delayed labels on no-op scores without inventing fresh feature observations."""
    from .monitoring_performance import observe_performance_safely  # noqa: PLC0415

    performance = observe_performance_safely(
        spark, namespace, config, artifact, spec, evidence, as_of
    )
    report = {
        "status": "no_data",
        "metrics": [],
        "performance": performance,
        "notes": ["No new scoring batch; evaluated the mature performance window only."],
    }
    evidence = evidence | {"performance": performance}
    row = result_row(config, evidence["model_version"], as_of, start, end, report, evidence)
    row["mlflow_run_id"] = _log_observation(
        config, row, report, evidence, tracking_uri, experiment_name
    )
    return row


def _log_observation(
    config: MonitorConfig,
    row: dict,
    report: dict,
    evidence: dict,
    tracking_uri: str | None,
    experiment_name: str | None,
) -> str | None:
    """Keep monitoring metrics in their own run rather than changing training evaluation."""
    existing = _finished_observation(row["report_id"], tracking_uri, experiment_name)
    if existing is not None:
        return existing
    tracking = TrackingConfig(
        enabled=experiment_name is not None,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )
    with track_run(tracking, run_name=f"monitor-{config.project}-{row['report_id'][:12]}") as run:
        run.set_tags(
            {
                "skyulf.monitoring.id": config.monitor_id,
                "skyulf.monitoring.report": row["report_id"],
                "skyulf.monitoring.status": report["status"],
                "skyulf.model.name": config.model_name,
                "skyulf.model.version": row["model_version"],
            }
        )
        run.log_config(
            {"report": report, "evidence": evidence}, artifact_file="monitoring/report.json"
        )
        run.log_metrics(
            {
                f"{m['category']}.{m['column_name']}.{m['metric_name']}": m["value"]
                for m in report["metrics"]
                if m["value"] is not None
            }
        )
        return run.run_id


def _finished_observation(
    report_id: str, tracking_uri: str | None, experiment_name: str | None
) -> str | None:
    """Reuse completed evidence on serialized job retries, including after a Delta write failure."""
    if experiment_name is None:
        return None
    client = make_tracking_client(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.`skyulf.monitoring.report` = '{report_id}' AND attributes.status = 'FINISHED'",
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def run_monitoring(
    spark: Any,
    namespace: str,
    models: list[dict] | None = None,
    *,
    as_of: datetime | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    tracking_uri: str | None = "databricks",
    registry_uri: str | None = "databricks-uc",
    experiment_name: str | None = None,
    enroll_models: bool = True,
    performance_only: bool = False,
) -> dict:
    """Observe the central inventory; optionally enroll explicit SDK records first."""
    qualified_name(f"{namespace}.model_inventory")
    as_of = datetime.now(UTC) if as_of is None else as_of
    start, end = observation_window(as_of, window_start, window_end)
    as_of = as_of.astimezone(UTC)
    if models is None:
        configs = _model_configs(load_enrolled_models(spark, namespace))
    else:
        configs = _model_configs(models)
        if enroll_models:
            for config in configs:
                enroll_monitor(spark, namespace, config)
    results = []
    for config in configs:
        if not config.enabled:
            continue
        row = _observe_or_failure(
            spark,
            config,
            as_of=as_of,
            window_start=start,
            window_end=end,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            experiment_name=experiment_name,
            namespace=namespace,
            performance_only=performance_only,
        )
        persist_report(spark, namespace, row)
        results.append({key: row[key] for key in ("report_id", "model_name", "status")})
    return {
        "results": results,
        "failed": sum(item["status"] == "failed" for item in results),
        "disabled": sum(not config.enabled for config in configs),
        "namespace": namespace,
    }


def _model_configs(models: list[dict]) -> list[MonitorConfig]:
    """Validate the whole inventory before its first registration side effect."""
    configs = [MonitorConfig.from_dict(model) for model in models]
    if len({config.monitor_id for config in configs}) != len(configs):
        raise ValueError("Monitoring configuration contains duplicate model identities.")
    return configs


def _observe_or_failure(spark: Any, config: MonitorConfig, **options: Any) -> dict:
    """Persist failure as an observation instead of hiding a model or fabricating health."""
    try:
        return observe_model(spark, config, **options)
    except Exception as exc:  # noqa: BLE001 - contain one model's external read/metric failure
        report = {"status": "failed", "metrics": [], "notes": [type(exc).__name__]}
        if config.performance_policy:
            from .monitoring_performance import unavailable_performance  # noqa: PLC0415

            report["performance"] = unavailable_performance(
                config,
                options["as_of"],
                f"{type(exc).__name__}: {str(exc)[:500]}",
                config.model_version,
            )
        return result_row(
            config,
            config.model_version,
            options["as_of"],
            options["window_start"],
            options["window_end"],
            report,
            {"failure_type": type(exc).__name__},
            error_message=f"{type(exc).__name__}: {str(exc)[:1000]}",
        )


def run_monitoring_notebook(spark: Any, dbutils: Any) -> dict:
    """Observe registered projects directly from the selected central Delta inventory."""
    values = dbutils.widgets.getAll()
    dates = {
        key: datetime.fromisoformat(values[key].replace("Z", "+00:00")) if values.get(key) else None
        for key in ("as_of", "window_start", "window_end")
    }
    observation_window(
        dates["as_of"] or datetime.now(UTC), dates["window_start"], dates["window_end"]
    )
    namespace = initialize_monitoring_store(
        spark, values["monitoring_catalog"], values["monitoring_schema"]
    )
    result = run_monitoring(
        spark,
        namespace,
        as_of=dates["as_of"],
        window_start=dates["window_start"],
        window_end=dates["window_end"],
        experiment_name=values.get("experiment_name") or None,
    )
    print(json.dumps(result, sort_keys=True))
    if result["failed"]:
        raise RuntimeError(
            f"{result['failed']} monitor(s) failed; results were saved in {namespace}."
        )
    return result
