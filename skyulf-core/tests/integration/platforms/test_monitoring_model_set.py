"""Multi-target monitoring retains concrete component and parent version identities."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_monitoring_config import performance_policy
from test_monitoring_registration import settings, workflow


@pytest.fixture(autouse=True)
def existing_monitoring_store(monkeypatch):
    """Keep component binding tests independent of shared-store DDL."""
    from skyulf.integrations.databricks import monitoring_model_set as module

    monkeypatch.setattr(module, "ensure_monitoring_store", Mock())


def component(branch, version):
    """Represent the already validated immutable parent manifest."""
    return SimpleNamespace(
        branch=branch, reference=SimpleNamespace(name=f"models.risk.{branch}", version=version)
    )


@pytest.mark.parametrize("activation", [None, 1000])
def test_components_share_physical_table_and_keep_own_versions(monkeypatch, activation):
    """Parent version and output generation must never be confused with component versions."""
    from skyulf.integrations.databricks import monitoring_model_set as module

    enroll = Mock()
    monkeypatch.setattr(module, "enroll_monitor", enroll)
    config = {"model_name": "models.risk.set", "prediction_table": "outputs.risk.set_scores"}
    resolved = SimpleNamespace(name=config["model_name"], version="9")
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(components=[component("revenue", "2"), component("cost", "4")])
    )
    result = module.register_set_monitors(
        Mock(),
        workflow(),
        settings(monitoring_drift_thresholds='{"psi": 0.4}'),
        config,
        resolved,
        artifact,
        activation_started_ms=activation,
    )
    assert result is not None
    configs = result["configs"]
    assert all(item["thresholds"] == {"psi": 0.4} for item in configs)
    assert [item["model_version"] for item in configs] == ["2", "4"]
    assert all(
        item["model_set_version"] == "9" and item["prediction_table"] == "outputs.risk.set_scores"
        for item in configs
    )
    assert len(enroll.call_args_list) == 2
    assert all(
        call.kwargs["preserve_activation"] is (activation is None) for call in enroll.call_args_list
    )


def test_model_set_preflight_rejects_invalid_thresholds():
    """Direct model-set scoring validates custom policy before publishing component outputs."""
    from skyulf.integrations.databricks.monitoring_model_set import validate_set_monitoring

    with pytest.raises(ValueError, match="threshold"):
        validate_set_monitoring(
            settings(monitoring_drift_thresholds='{"psi": 0}'),
            {"publication": {"mode": "all"}},
        )


def test_model_set_preflight_rejects_invalid_performance_mapping():
    """Component policy errors are caught before scored outputs are published."""
    from skyulf.integrations.databricks.monitoring_model_set import validate_set_monitoring

    with pytest.raises(ValueError):
        validate_set_monitoring(
            settings(monitoring_performance_policies='{"models.risk.cost": {"mode": "retrain"}}'),
            {"publication": {"mode": "all"}},
        )


def test_model_set_rejects_unmatched_component_policy_before_enrollment(monkeypatch):
    """A typo in component identity must not be silently discarded after scoring."""
    from skyulf.integrations.databricks import monitoring_model_set as module

    enroll = Mock()
    monkeypatch.setattr(module, "enroll_monitor", enroll)
    parent = {"model_name": "models.risk.set", "prediction_table": "outputs.risk.set_scores"}
    resolved = SimpleNamespace(name=parent["model_name"], version="9")
    artifact = SimpleNamespace(manifest=SimpleNamespace(components=[component("cost", "2")]))
    values = settings(monitoring_performance_policies='{"models.risk.cosst": {"mode": "off"}}')
    with pytest.raises(ValueError, match="component"):
        module.register_set_monitors(Mock(), workflow(), values, parent, resolved, artifact)
    enroll.assert_not_called()


def test_model_set_selects_component_policy_independently(monkeypatch):
    """Each component gets only its own version-bound performance policy."""
    from skyulf.integrations.databricks import monitoring_model_set as module

    monkeypatch.setattr(module, "enroll_monitor", Mock())
    parent = {"model_name": "models.risk.set", "prediction_table": "outputs.risk.set_scores"}
    resolved = SimpleNamespace(name=parent["model_name"], version="9")
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(components=[component("revenue", "2"), component("cost", "4")])
    )
    values = settings(
        monitoring_label_table="labels.risk.actuals",
        monitoring_result_available_at_column="available_at",
        monitoring_performance_policies=json.dumps({"models.risk.revenue": performance_policy()}),
    )
    result = module.register_set_monitors(Mock(), workflow(), values, parent, resolved, artifact)
    assert result is not None
    assert result["configs"][0]["performance_policy"] == performance_policy()
    assert "performance_policy" not in result["configs"][1]


def test_multi_target_request_publishes_batch_and_components():
    """Independent monitoring receives the scored parent batch without resolving aliases."""
    from skyulf.integrations.databricks.monitoring_registration import publish_monitoring_request

    dbutils = Mock()
    payload = {
        "monitoring": {
            "inventory_table": "ops.monitoring.model_inventory",
            "configs": [{"model_version": "2"}, {"model_version": "4"}],
        },
        "commit_version": 3,
        "noop": False,
        "manifest": {"saved": True},
    }
    publish_monitoring_request(dbutils, payload)
    value = dbutils.jobs.taskValues.set.call_args.kwargs["value"]
    assert value["commit_version"] == 3 and value["has_saved_batch"] is True
    assert value["configs"] == payload["monitoring"]["configs"]


def test_component_batch_skips_only_completed_observations(monkeypatch):
    """One saved component cannot suppress another component's missing observation."""
    from skyulf.integrations.databricks import monitoring_tasks as tasks
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    configs = [
        MonitorConfig(
            environment="test",
            project="risk",
            model_name=f"models.risk.{branch}",
            model_version="2",
            source_table="features.risk.source",
            prediction_table="outputs.risk.set_scores",
            model_set_name="models.risk.set",
            model_set_version="9",
            model_set_branch=branch,
        )
        for branch in ("revenue", "cost")
    ]
    start = datetime(2026, 10, 1, tzinfo=UTC)
    end = start + timedelta(microseconds=1)
    monkeypatch.setattr(tasks, "scoring_observation_window", lambda *args: (start, end))
    monkeypatch.setattr(
        tasks,
        "completed_observation",
        lambda spark, namespace, config, *args: config.model_set_branch == "revenue",
    )
    run = Mock(return_value={"results": [{"status": "healthy"}], "failed": 0})
    monkeypatch.setattr(tasks, "run_monitoring", run)
    dbutils = Mock()
    tasks._observe_configs(
        Mock(),
        dbutils,
        "ops.monitoring",
        configs,
        {"commit_version": 3, "noop": False},
        workflow(),
        settings(),
    )
    assert run.call_args.args[2] == [configs[1].payload()]
    references = dbutils.jobs.taskValues.set.call_args.kwargs["value"]["observations"]
    assert [item["monitor_id"] for item in references] == [config.monitor_id for config in configs]
    assert all(item["window_start"] == start.isoformat() for item in references)


def test_set_activation_uses_frozen_settings_and_receipt(monkeypatch):
    """Enrollment before scoring must load the activated parent version, never its live alias."""
    from skyulf.integrations.databricks import monitoring_model_set as module
    from skyulf.integrations.databricks import monitoring_tasks
    from skyulf.integrations.mlflow import model_set, registry

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings()
    store = Mock()
    store.request = {
        "action": "train",
        "config": workflow(),
        "settings": {
            "model_name": "models.risk.set",
            "prediction_table": "outputs.risk.set_scores",
        },
    }
    store.receipt.return_value = {
        "output": {
            "alias_change": {
                "kind": "promotion",
                "model_name": "models.risk.set",
                "alias": "champion",
                "new_version": "9",
            }
        }
    }
    store.client.get_run.return_value.info.start_time = 1000
    bind = Mock(return_value=store)
    monkeypatch.setattr(monitoring_tasks, "deployment_store", bind)
    resolve = Mock(return_value=SimpleNamespace(name="models.risk.set", version="9"))
    monkeypatch.setattr(registry, "resolve_model", resolve)
    load = Mock()
    monkeypatch.setattr(model_set, "load_registered_model_set", load)
    enroll = Mock(return_value={"configs": []})
    monkeypatch.setattr(module, "register_set_monitors", enroll)
    result = module.run_set_monitor_enrollment_notebook(Mock(), dbutils)
    assert result["status"] == "enrolled"
    assert bind.call_args.kwargs == {"expected_phase": "prepare"}
    assert resolve.call_args.kwargs["version"] == "9"
    assert enroll.call_args.args[3] == store.request["settings"]
    assert enroll.call_args.kwargs["activation_started_ms"] == 1000
