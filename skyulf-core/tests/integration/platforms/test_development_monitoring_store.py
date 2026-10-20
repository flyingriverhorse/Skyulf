"""Development monitoring writes to its own store, which must differ from the shared one."""

import pytest

SHARED = {"monitoring_catalog": "main", "monitoring_schema": "skyulf_monitoring"}
DEV = {"monitoring_deployment_mode": "development", "monitoring_enabled": "true"}


def _destination(values):
    """Resolve the monitoring store exactly as every monitoring writer does."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        monitoring_destination,
    )

    return monitoring_destination(values)


@pytest.mark.parametrize(
    "values",
    [
        {"monitoring_deployment_mode": "development"},
        {"monitoring_deployment_mode": "development", **SHARED},
        {"monitoring_deployment_mode": "development", "monitoring_enabled": "false", **SHARED},
    ],
)
def test_development_without_opt_in_still_writes_nothing(values):
    """Old development invocations must keep the original guarantee: no monitoring writes."""
    assert _destination(values) is None


@pytest.mark.parametrize("name", ["development_monitoring", "mlops_dev", "gold_sandbox"])
def test_development_writes_to_its_own_store_with_any_name(name):
    """The development store may have any name; only its separation matters."""
    values = {**DEV, **SHARED, "development_monitoring_schema": name}
    assert _destination(values) == f"main.{name}"


@pytest.mark.parametrize(
    ("own", "message"),
    [
        ("", "Set development_monitoring_schema"),
        ("skyulf_monitoring", "must differ from the shared"),
    ],
)
def test_development_refuses_a_missing_or_shared_store(own, message):
    """An enabled development target without its own store must fail instead of mixing models."""
    with pytest.raises(ValueError, match=message):
        _destination({**DEV, **SHARED, "development_monitoring_schema": own})


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_shared_targets_ignore_the_development_store(enabled):
    """test, syst and prod always write to the shared store, whatever the dev setting says."""
    values = {
        "monitoring_deployment_mode": "production",
        "monitoring_enabled": enabled,
        "development_monitoring_schema": "mlops_dev",
        **SHARED,
    }
    assert _destination(values) == "main.skyulf_monitoring"


def test_enrollment_reference_store_follows_the_development_store():
    """Native references must live in the same store the development target writes to."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        build_monitor_enrollment_config,
    )

    values = {
        **DEV,
        **SHARED,
        "development_monitoring_schema": "mlops_dev",
        "monitoring_execution_engine": "spark",
        "monitoring_environment": "test_development",
        "monitoring_project": "risk",
    }
    workflow = {
        "model_name": "main.ml.model_dev",
        "score_source_table": "main.src.rows",
        "prediction_table": "main.out.predictions",
        "model_change_mode": "full_rebuild",
    }
    config = build_monitor_enrollment_config(workflow, values, "1")
    assert config.reference_namespace == "main.mlops_dev"


def test_development_retraining_stays_disabled():
    """Development monitoring only observes; drift never starts training there."""
    from skyulf.integrations.databricks.jobs.lifecycle.retraining_task import retraining_policy

    values = {"monitoring_deployment_mode": "development", "on_drift": "retrain"}
    assert retraining_policy(values)["mode"] == "disabled"


@pytest.mark.parametrize(
    ("mode", "own", "expected"),
    [
        ("development", "mlops_dev", "main.mlops_dev"),
        ("development", "skyulf_monitoring", None),
        ("production", "mlops_dev", "main.skyulf_monitoring"),
    ],
)
def test_serving_enrollment_uses_the_same_store_rule(mode, own, expected):
    """Endpoint monitoring follows the same separation rule as batch monitoring."""
    from skyulf.integrations.databricks.jobs import serving_enrollment

    values = {
        "bundle_mode": mode,
        "monitoring_enabled": "true",
        "development_monitoring_schema": own,
        "job_run_id": "1",
        "task_key": "serve",
        "workspace_host": "https://example.cloud.databricks.com",
        **SHARED,
    }
    if expected is None:
        with pytest.raises(ValueError, match="must differ"):
            serving_enrollment.serving_monitoring_namespace("telemetry", values)
    else:
        assert serving_enrollment.serving_monitoring_namespace("telemetry", values) == expected
