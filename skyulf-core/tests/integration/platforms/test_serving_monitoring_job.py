"""Serving enrollment uses the independent Spark monitoring job."""

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.jobs.monitoring import spark_monitoring_job as job
from skyulf.integrations.databricks.observability.monitoring.monitoring_config import MonitorConfig
from skyulf.integrations.databricks.observability.monitoring.serving import serving_registration


def _config(**changes):
    """Return one valid concrete serving enrollment."""
    return {
        "environment": "production",
        "project": "risk",
        "model_name": "models.risk.churn",
        "model_version": "2",
        "source_table": "logs.serving.risk_payload",
        "prediction_table": "logs.serving.risk_payload",
        "execution_engine": "spark",
        "reference_namespace": "ops.monitoring",
        "serving_endpoint": "risk-v2",
    } | changes


def _widgets(items):
    """Expose the actual notebook widget and task-value boundary."""
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(getAll=lambda: items),
        jobs=SimpleNamespace(taskValues=SimpleNamespace(set=Mock())),
    )
    return dbutils


def _values(configs):
    """Bind enrollment to a concrete enabled project and central store."""
    return {
        "monitoring_action": "enroll_serving",
        "monitoring_enabled": "true",
        "monitoring_catalog": "ops",
        "monitoring_schema": "monitoring",
        "monitoring_environment": "production",
        "monitoring_project": "risk",
        "monitoring_serving_enrollments": json.dumps(configs),
    }


def _record_admission(monkeypatch, events):
    """Record the whole guarded interval while isolating the real Delta authority boundary."""

    @contextmanager
    def scope(*args, **kwargs):
        """Keep enrollment operations inside an observable admission context."""
        events.append("admission_enter")
        yield
        events.append("admission_exit")

    admission = Mock(side_effect=scope)
    monkeypatch.setattr(job, "reference_mutation_scope", admission)
    return admission


@pytest.mark.parametrize(
    "configs, error",
    [
        ([], "nonempty"),
        ([_config(), _config()], "distinct"),
        ([_config(project="other")], "project"),
        ([_config(environment="test")], "environment"),
        ([_config(reference_namespace="other.monitoring")], "namespace"),
        ([_config(serving_endpoint=None)], "serving_endpoint"),
        ([_config(unexpected="field")], "Unknown monitoring"),
        ([_config()] * 101, "100"),
    ],
)
def test_enrollment_rejects_invalid_batch_before_any_store_write(monkeypatch, configs, error):
    """One bad entry must not leave a partially enrolled project."""
    store = Mock(side_effect=AssertionError("store write"))
    admission = Mock(side_effect=AssertionError("admission before validation"))
    monkeypatch.setattr(serving_registration, "ensure_monitoring_store", store)
    monkeypatch.setattr(job, "reference_mutation_scope", admission)
    dbutils = _widgets(_values(configs))
    with pytest.raises(ValueError, match=error):
        job.run_project_monitoring_notebook(object(), dbutils)
    store.assert_not_called()
    admission.assert_not_called()
    dbutils.jobs.taskValues.set.assert_called_once_with(
        key="monitoring_reference", value={"status": "disabled"}
    )


@pytest.mark.parametrize("with_holder", [False, True])
def test_enrollment_prepares_all_references_then_enrolls_two_versions(monkeypatch, with_holder):
    """Prepared online versions must enter inventory without publishing an observation."""
    events = []
    admission = _record_admission(monkeypatch, events)
    monkeypatch.setattr(
        serving_registration, "ensure_monitoring_store", lambda *args: events.append("store")
    )
    monkeypatch.setattr(
        serving_registration,
        "prepare_spark_monitoring_reference",
        lambda spark, config, **kwargs: events.append(("prepare", config.model_version)),
    )
    monkeypatch.setattr(
        serving_registration,
        "enroll_monitor",
        lambda spark, namespace, config, **kwargs: events.append(
            ("enroll", namespace, config.model_version)
        ),
    )
    configs = [_config(), _config(model_version="3")]
    values = _values(configs)
    holder = {}
    if with_holder:
        values.update(
            job_run_id="123", task_key="monitoring", workspace_host="https://example.databricks.com"
        )
        holder = {
            "job_run_id": 123,
            "task_key": "monitoring",
            "workspace_host": "https://example.databricks.com",
        }
    dbutils = _widgets(values)
    spark = object()
    result = job.run_project_monitoring_notebook(spark, dbutils)
    assert events == [
        "admission_enter",
        "store",
        ("prepare", "2"),
        ("prepare", "3"),
        ("enroll", "ops.monitoring", "2"),
        ("enroll", "ops.monitoring", "3"),
        "admission_exit",
    ]
    admission.assert_called_once_with(spark, "ops.monitoring", **holder)
    assert result == {
        "status": "enrolled",
        "models": 2,
        "monitor_ids": [MonitorConfig.from_dict(config).monitor_id for config in configs],
        "inventory_table": "ops.monitoring.model_inventory",
    }
    dbutils.jobs.taskValues.set.assert_called_once_with(
        key="monitoring_reference", value={"status": "disabled"}
    )


def test_scheduled_project_selects_batch_and_serving(monkeypatch):
    """The ordinary schedule must retain batch enrollments alongside online entries."""
    batch = MonitorConfig.from_dict(
        _config(
            serving_endpoint=None,
            source_table="data.risk.features",
            prediction_table="data.risk.predictions",
        )
    )
    online = MonitorConfig.from_dict(_config())
    monkeypatch.setattr(
        job, "load_enrolled_models", lambda spark, namespace: [batch.payload(), online.payload()]
    )
    selected = job.project_configs(object(), "ops.monitoring", _values([]))
    assert [item.monitor_id for item in selected] == [batch.monitor_id, online.monitor_id]


def test_disabling_serving_monitor_does_not_require_available_reference(monkeypatch):
    """An unavailable model or reference must not block disabling its broken enrollment."""
    monkeypatch.setattr(serving_registration, "ensure_monitoring_store", Mock())
    events = []
    admission = _record_admission(monkeypatch, events)
    prepare = Mock(side_effect=AssertionError("unavailable registry or reference"))
    enroll = Mock()
    monkeypatch.setattr(serving_registration, "prepare_spark_monitoring_reference", prepare)
    monkeypatch.setattr(serving_registration, "enroll_monitor", enroll)
    config = _config(enabled=False)
    spark = object()
    result = job.run_project_monitoring_notebook(spark, _widgets(_values([config])))
    assert result["status"] == "enrolled"
    prepare.assert_not_called()
    enroll.assert_called_once()
    assert enroll.call_args.args[2].enabled is False
    assert enroll.call_args.kwargs == {"preserve_activation": True}
    admission.assert_called_once_with(spark, "ops.monitoring")
    assert events == ["admission_enter", "admission_exit"]


def test_observation_publishes_batch_and_serving_references():
    """Both selected identities must reach the existing report and retraining tasks."""
    batch = MonitorConfig.from_dict(
        _config(
            serving_endpoint=None,
            source_table="data.risk.features",
            prediction_table="data.risk.predictions",
        )
    )
    online = MonitorConfig.from_dict(_config())
    dbutils = _widgets({})
    start = datetime(2026, 10, 5, tzinfo=UTC)
    end = datetime(2026, 10, 6, tzinfo=UTC)
    job._publish_references(dbutils, "ops.monitoring", [batch, online], start, end)
    published = dbutils.jobs.taskValues.set.call_args.kwargs["value"]
    assert published["status"] == "ready"
    assert [item["monitor_id"] for item in published["observations"]] == [
        batch.monitor_id,
        online.monitor_id,
    ]


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, {"tracking_uri": "databricks", "registry_uri": "databricks-uc"}),
        (
            {"tracking_uri": "databricks://alternate", "registry_uri": "databricks-uc://alternate"},
            {"tracking_uri": "databricks://alternate", "registry_uri": "databricks-uc://alternate"},
        ),
    ],
)
def test_shared_enrollment_reference_registry_options(monkeypatch, options, expected):
    """Existing callers retain reference defaults while explicit registries survive enrollment."""
    monkeypatch.setattr(serving_registration, "ensure_monitoring_store", Mock())
    monkeypatch.setattr(serving_registration, "enroll_monitor", Mock())
    prepare = Mock()
    monkeypatch.setattr(serving_registration, "prepare_spark_monitoring_reference", prepare)
    result = serving_registration.enroll_serving_configs(
        object(), "ops.monitoring", _values([_config()]), **options
    )
    assert result["status"] == "enrolled"
    assert prepare.call_count == 1
    assert prepare.call_args.kwargs == expected
