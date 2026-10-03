"""Monitoring follows committed model deployments and scoring without repeating inference."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
from test_monitoring_registration import settings, workflow


def test_development_monitoring_tasks_exit_before_reading_models_or_tables(monkeypatch):
    """Direct notebook retries in dev must not access the central monitoring store."""
    from skyulf.integrations.databricks import monitoring_model_set as sets
    from skyulf.integrations.databricks import monitoring_tasks as tasks
    from skyulf.integrations.databricks.retraining_task import (
        run_retraining_check_notebook,
        run_retraining_notebook,
    )

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings(
        monitoring_deployment_mode="development", on_drift="retrain"
    )
    run = Mock(side_effect=AssertionError("Development must not run monitoring"))
    monkeypatch.setattr(tasks, "run_monitoring", run)
    spark = Mock()
    assert tasks.run_monitor_enrollment_notebook(spark, dbutils) == {"status": "disabled"}
    assert tasks.run_scoring_monitor_notebook(spark, dbutils) == {"status": "disabled"}
    assert sets.run_set_monitor_enrollment_notebook(spark, dbutils) == {"status": "disabled"}
    assert sets.register_set_monitors(spark, {}, dbutils.widgets.getAll(), {}, None, None) is None
    result = run_retraining_check_notebook(spark, dbutils)
    assert result["status"] == "disabled"
    assert run_retraining_notebook(spark, dbutils)["status"] == "disabled"
    assert spark.mock_calls == []
    run.assert_not_called()


@pytest.fixture(autouse=True)
def no_previous_observation(monkeypatch):
    """Keep task unit tests independent of a real Delta results table."""
    from skyulf.integrations.databricks import monitoring_tasks as tasks

    monkeypatch.setattr(tasks, "completed_observation", lambda *args: False)


@pytest.mark.parametrize(
    "action,kind", [("train", "initial"), ("approve", "promotion"), ("rollback", "rollback")]
)
def test_deployment_enrolls_actual_activated_version(monkeypatch, action, kind):
    """A promoted model must be visible before its first prediction table exists."""
    from skyulf.integrations.databricks import monitoring_registration as registration

    saved = []
    monkeypatch.setattr(
        registration, "enroll_monitor", lambda spark, ns, cfg, **kwargs: saved.append(cfg)
    )
    receipt = {"kind": kind, "model_name": "models.risk.model", "new_version": "7"}
    payload = {
        "action": action,
        "result": {"alias_change": receipt} if action == "train" else receipt,
    }
    registration.register_deployed_monitor(
        Mock(), workflow(), settings(), payload, activation_started_ms=1000
    )
    assert saved[0].model_version == "7"
    assert saved[0].prediction_table == "outputs.risk.predictions_v7"


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "train", "result": {"model_version": "8"}},
        {"action": "train", "result": {"alias_change": None}},
        {"action": "reject", "result": {"kind": "rejection", "new_version": "8"}},
    ],
)
def test_pending_or_rejected_candidates_do_not_replace_active_monitor(monkeypatch, payload):
    """Training and registration alone must not replace the deployed model's enrollment."""
    from skyulf.integrations.databricks import monitoring_registration as registration

    enroll = Mock()
    monkeypatch.setattr(registration, "enroll_monitor", enroll)
    assert (
        registration.register_deployed_monitor(
            Mock(), workflow(), settings(), payload, activation_started_ms=1000
        )
        is None
    )
    enroll.assert_not_called()


def request(**changes):
    """Represent the immutable handoff saved by the successful score task."""
    from skyulf.integrations.databricks.monitoring_registration import (
        build_monitor_enrollment_config,
    )

    return {
        "namespace": "ops.monitoring",
        "config": build_monitor_enrollment_config(workflow(), settings(), "7").payload(),
        "commit_version": 3,
        "noop": False,
        **changes,
    }


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("explicit_enabled", [False, True])
def test_monitor_task_uses_successful_branch_and_pinned_commit(
    monkeypatch, recovery, explicit_enabled
):
    """Monitoring observes just this batch and never resolves a moved alias or re-enrolls it."""
    from skyulf.integrations.databricks import monitoring_tasks as tasks

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings()
    if not explicit_enabled:
        dbutils.widgets.getAll.return_value.pop("monitoring_enabled")
    monkeypatch.setattr(
        tasks, "read_notebook_config", lambda _: workflow(auto_rebuild_on_cdf_expiry=recovery)
    )
    values = {
        ("score", "recovery_required"): recovery,
        ("recover_predictions" if recovery else "score", "monitoring_request"): request(),
    }
    dbutils.jobs.taskValues.get.side_effect = lambda taskKey, key: values[taskKey, key]
    start = datetime(2026, 10, 1, tzinfo=UTC)
    end = start + timedelta(microseconds=1)
    monkeypatch.setattr(tasks, "scoring_observation_window", lambda *args: (start, end))
    run = Mock(return_value={"results": [{"status": "drift"}], "failed": 0})
    monkeypatch.setattr(tasks, "run_monitoring", run)
    result = tasks.run_scoring_monitor_notebook(Mock(), dbutils)
    assert result["results"][0]["status"] == "drift"
    assert run.call_args.args[2][0]["model_version"] == "7"
    assert run.call_args.kwargs["enroll_models"] is False
    assert run.call_args.kwargs["window_start"] == start
    assert run.call_args.kwargs["window_end"] == end
    reference = dbutils.jobs.taskValues.set.call_args.kwargs["value"]
    assert reference["status"] == "ready"
    assert reference["window_start"] == start.isoformat()
    assert reference["window_end"] == end.isoformat()


def test_noop_monitor_task_preserves_previous_observation(monkeypatch):
    """An empty scoring retry must not replace prior health with an empty-window report."""
    from skyulf.integrations.databricks import monitoring_tasks as tasks

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings()
    dbutils.jobs.taskValues.get.return_value = request(noop=True, commit_version=None)
    monkeypatch.setattr(tasks, "read_notebook_config", lambda _: workflow())
    run = Mock()
    monkeypatch.setattr(tasks, "run_monitoring", run)
    assert tasks.run_scoring_monitor_notebook(Mock(), dbutils)["status"] == "no_new_predictions"
    run.assert_not_called()


def test_failed_measurement_fails_task_after_report_is_saved(monkeypatch):
    """Monitoring errors remain visible without asking scoring to publish predictions again."""
    from skyulf.integrations.databricks import monitoring_tasks as tasks

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings()
    dbutils.jobs.taskValues.get.return_value = request()
    monkeypatch.setattr(tasks, "read_notebook_config", lambda _: workflow())
    now = datetime.now(UTC)
    monkeypatch.setattr(
        tasks, "scoring_observation_window", lambda *args: (now, now + timedelta(microseconds=1))
    )
    run = Mock(return_value={"failed": 1, "results": [{"status": "failed"}]})
    monkeypatch.setattr(tasks, "run_monitoring", run)
    with pytest.raises(RuntimeError, match="saved"):
        tasks.run_scoring_monitor_notebook(Mock(), dbutils)
    run.assert_called_once()


def test_enrollment_notebook_uses_verified_result_and_frozen_workflow(monkeypatch):
    """A later edit to workflow.json cannot alter the completed deployment's enrollment."""
    from skyulf.integrations.databricks import monitoring_tasks as tasks

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings()
    frozen = workflow()
    payload = {
        "action": "approve",
        "result": {"kind": "promotion", "model_name": frozen["model_name"], "new_version": "7"},
    }
    store = Mock(request={"config": frozen})
    store.receipt.return_value = {"output": payload}
    monkeypatch.setattr(tasks, "deployment_store", lambda _: store)
    register = Mock(return_value={"monitor_id": "id"})
    monkeypatch.setattr(tasks, "register_deployed_monitor", register)
    tasks.run_monitor_enrollment_notebook(Mock(), dbutils)
    assert register.call_args.args[1] == frozen
    assert register.call_args.args[3] == payload


def test_successful_scoring_publishes_monitor_request_without_recovery(monkeypatch):
    """The visible monitor task needs a pinned handoff even when CDF recovery is disabled."""
    from skyulf.integrations.databricks import job_runtime, monitoring_registration
    from skyulf.integrations.databricks.local_incremental import IncrementalBatchResult
    from skyulf.integrations.databricks.local_workflow import BundleActionResult

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings()
    monkeypatch.setattr(job_runtime, "read_notebook_config", lambda _: workflow())
    monkeypatch.setattr(monitoring_registration, "enroll_monitor", Mock())
    outcome = IncrementalBatchResult(0, 1, 3, 3, 3, None, False, "models.risk.model", "7")
    monkeypatch.setattr(
        job_runtime,
        "run_bundle_action",
        lambda *args, **kwargs: BundleActionResult("score", outcome, False, {}),
    )
    json.loads(job_runtime.run_score_notebook(Mock(), dbutils, exit_notebook=False))
    handoffs = [
        call.kwargs["value"]
        for call in dbutils.jobs.taskValues.set.call_args_list
        if call.kwargs["key"] == "monitoring_request"
    ]
    assert handoffs[0]["config"]["model_version"] == "7"
    assert handoffs[0]["commit_version"] == 3


def test_scoring_window_uses_delta_commit_time_without_timezone_guessing():
    """A repaired task must find its old batch even after the daily monitoring window closes."""
    from unittest.mock import patch

    from skyulf.integrations.databricks import monitoring_tasks as tasks
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    spark = Mock()
    delta_history = Mock()
    delta_history.where.return_value.selectExpr.return_value.first.return_value = {
        "committed_us": 1000001
    }
    with patch.object(tasks, "history", return_value=delta_history):
        start, end = tasks.scoring_observation_window(
            spark, MonitorConfig.from_dict(request()["config"]), 3
        )
    assert start == datetime(1970, 1, 1, 0, 0, 1, 1, tzinfo=UTC)
    assert end - start == timedelta(microseconds=1)


def test_post_score_observation_does_not_reenroll_old_config(monkeypatch):
    """A delayed measurement cannot overwrite a newer deployment's inventory record."""
    from skyulf.integrations.databricks import monitoring

    enroll = Mock()
    monkeypatch.setattr(monitoring, "enroll_monitor", enroll)
    monkeypatch.setattr(monitoring, "persist_report", Mock())
    monkeypatch.setattr(
        monitoring,
        "observe_model",
        lambda *args, **kwargs: {
            "report_id": "report",
            "model_name": "models.risk.model",
            "status": "healthy",
        },
    )
    result = monitoring.run_monitoring(
        Mock(), "ops.monitoring", [request()["config"]], enroll_models=False
    )
    assert result["failed"] == 0
    enroll.assert_not_called()


@pytest.mark.parametrize("measured", [False, True])
def test_noop_repairs_missing_measurement_without_repeating_completed_report(monkeypatch, measured):
    """A retry after scoring commits but monitoring fails must still recover the observation."""
    from skyulf.integrations.databricks import monitoring_tasks as tasks

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings()
    dbutils.jobs.taskValues.get.return_value = request(noop=True, has_saved_batch=True)
    monkeypatch.setattr(tasks, "read_notebook_config", lambda _: workflow())
    now = datetime.now(UTC)
    monkeypatch.setattr(
        tasks, "scoring_observation_window", lambda *args: (now, now + timedelta(microseconds=1))
    )
    monkeypatch.setattr(tasks, "completed_observation", lambda *args: measured, raising=False)
    run = Mock(return_value={"failed": 0, "results": [{"status": "healthy"}]})
    monkeypatch.setattr(tasks, "run_monitoring", run)
    result = tasks.run_scoring_monitor_notebook(Mock(), dbutils)
    assert run.call_count == int(not measured)
    assert result.get("status") == ("already_observed" if measured else None)


def test_successful_monitor_only_repair_does_not_recalculate(monkeypatch):
    """An exact successful observation cannot become failed due to a later transient read error."""
    from skyulf.integrations.databricks import monitoring_tasks as tasks

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings()
    dbutils.jobs.taskValues.get.return_value = request()
    monkeypatch.setattr(tasks, "read_notebook_config", lambda _: workflow())
    now = datetime.now(UTC)
    monkeypatch.setattr(
        tasks, "scoring_observation_window", lambda *args: (now, now + timedelta(microseconds=1))
    )
    monkeypatch.setattr(tasks, "completed_observation", lambda *args: True)
    run = Mock(side_effect=AssertionError("Completed measurement was repeated"))
    monkeypatch.setattr(tasks, "run_monitoring", run)
    assert tasks.run_scoring_monitor_notebook(Mock(), dbutils)["status"] == "already_observed"
    run.assert_not_called()


def test_enrollment_repair_retains_original_lifecycle_identity(monkeypatch):
    """Repairing enrollment must not repeat activation or weaken cross-run receipt binding."""
    from skyulf.integrations.databricks import monitoring_tasks as tasks

    store = Mock()
    factory = Mock(return_value=store)
    monkeypatch.setattr(tasks, "PhaseStore", factory)
    reference = {"phase": "result", "run_id": "original"}
    values = {
        "workflow_contract": "3",
        "job_id": "12",
        "job_run_id": "34",
        "repair_count": "2",
        "execution_count": "3",
        "tracking_uri": "databricks",
        "reference_json": json.dumps(reference),
    }
    assert tasks.deployment_store(values) is store
    context = factory.call_args.args[1]
    assert context.identity() == {"job_id": "12", "job_run_id": "34"}
    store.bind.assert_called_once_with(reference)


def test_activation_metadata_preserves_legacy_config_identity():
    """Existing tables and report digests remain valid when activation ordering is added."""
    from skyulf.integrations.databricks import monitoring_store as store
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    config = MonitorConfig.from_dict(request()["config"])
    row = store.inventory_row(config, datetime.now(UTC))
    row["config_json"] = json.dumps(config.payload() | {"_activation_started_ms": 1000})
    assert store._inventory_config(row) == config
    row["config_json"] = json.dumps(config.payload() | {"_activation_started_ms": True})
    with pytest.raises(ValueError, match="Activation order"):
        store._inventory_config(row)


def test_scoring_merge_protects_active_selection_and_preserves_activation_order(monkeypatch):
    """The version guard must execute inside Delta MERGE, not through a racy read-before-write."""
    from skyulf.integrations.databricks import monitoring_store as store
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    monkeypatch.setattr(store, "ensure_owned_object", lambda *args: True)
    monkeypatch.setattr(store, "_inventory_isolation", lambda *args: "Serializable")
    merge = Mock()
    monkeypatch.setattr(store, "merge_with_retry", merge)
    store.enroll_monitor(
        Mock(),
        "ops.monitoring",
        MonitorConfig.from_dict(request()["config"]),
        preserve_activation=True,
    )
    sql = merge.call_args.args[1]
    assert "OR t.selection = s.selection) THEN UPDATE" in sql
    assert "_activation_started_ms" in sql and "substring(s.config_json, 2)" in sql


def test_old_activation_repair_cannot_replace_a_later_activation(monkeypatch):
    """Rollback versions remain supported while old invocation replays cannot regress ownership."""
    from skyulf.integrations.databricks import monitoring_store as store
    from skyulf.integrations.databricks.monitoring_config import MonitorConfig

    monkeypatch.setattr(store, "ensure_owned_object", lambda *args: True)
    monkeypatch.setattr(store, "_inventory_isolation", lambda *args: "Serializable")
    merge = Mock()
    monkeypatch.setattr(store, "merge_with_retry", merge)
    store.enroll_monitor(
        Mock(),
        "ops.monitoring",
        MonitorConfig.from_dict(request()["config"]),
        activation_started_ms=1000,
    )
    sql = merge.call_args.args[1]
    assert "AS BIGINT) < 1000" in sql
    assert "= 1000 AND t.selection = s.selection" in sql
