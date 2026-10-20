"""Generated mutation phases share reference admission without locking fitting."""

import json
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

mlflow = pytest.importorskip("mlflow")

from skyulf.integrations.databricks.jobs.lifecycle import lifecycle_tasks
from skyulf.integrations.databricks.jobs.shared import job_runtime
from skyulf.integrations.databricks.jobs.training import branch_tasks
from skyulf.integrations.databricks.model_sets import model_set_stages
from skyulf.integrations.databricks.observability.monitoring import reference_admission


@pytest.fixture
def admission(monkeypatch):
    """Record protected remote boundaries while preserving context-manager failure semantics."""
    state = SimpleNamespace(held=False, calls=[], exits=[])

    @contextmanager
    def hold(spark, namespace, **kwargs):
        """Make any mutation outside the acquired scope an observable test failure."""
        state.calls.append((spark, namespace, kwargs))
        state.held = True
        try:
            yield "owner"
        except BaseException:
            state.exits.append("failed")
            raise
        else:
            state.exits.append("released")
        finally:
            state.held = False

    monkeypatch.setattr(reference_admission, "hold_reference_mutation", hold)
    monkeypatch.setattr(reference_admission, "defer_reference_mutation", hold)

    def require(*args):
        """Require the write boundary to remain inside the recorded admission scope."""
        assert state.held

    monkeypatch.setattr(reference_admission, "require_reference_mutation", require)
    return state


def _store():
    """Represent an already integrity-checked frozen invocation at the phase boundary."""
    return SimpleNamespace(
        request={
            "action": "approve",
            "graph_version": 3,
            "monitoring_namespace": "catalog.monitoring",
            "workspace_host": "https://workspace.example",
            "config": {},
            "settings": {},
            "operator_values": {},
        },
        context=SimpleNamespace(job_run_id="123"),
        run_id="run",
        client=Mock(),
        begin=Mock(),
        complete=Mock(return_value="completed"),
    )


@pytest.mark.parametrize(
    "phase,task",
    [
        ("evaluate_register", "register_model"),
        ("compare", "evaluate_model"),
        ("decide", "model_decision"),
        ("operator", "model_decision"),
        ("finalize", "training_report"),
    ],
)
@pytest.mark.parametrize("fails", [False, True])
def test_lifecycle_mutation_and_failure_cleanup_share_admission(
    monkeypatch, admission, phase, task, fails
):
    """Cleanup must not escape the lock when a remote alias request raises."""
    store = _store()

    def mutation(*args):
        """Assert the actual phase callback runs after admission."""
        assert admission.held
        if fails:
            raise RuntimeError("lost remote reply")
        return {"status": "FINISHED"}

    def cleanup(*args):
        """An alias-restoration attempt remains protected after a phase failure."""
        assert admission.held

    monkeypatch.setattr(lifecycle_tasks, f"_{phase}", mutation)
    monkeypatch.setattr(lifecycle_tasks, "record_phase_failure", cleanup)
    if fails:
        with pytest.raises(RuntimeError, match="lost remote reply"):
            lifecycle_tasks._execute_phase("spark", store, phase, {})
    else:
        assert lifecycle_tasks._execute_phase("spark", store, phase, {}) == "completed"
    assert admission.calls == [
        (
            "spark",
            "catalog.monitoring",
            {"job_run_id": 123, "task_key": task, "workspace_host": "https://workspace.example"},
        )
    ]
    assert admission.exits == ["failed" if fails else "released"]


@pytest.mark.parametrize("phase", ["train", "load_data", "prepare_dataset"])
def test_training_data_and_fit_do_not_hold_reference_admission(monkeypatch, admission, phase):
    """The shared store remains available during long-running fitting and data reads."""
    store = _store()
    monkeypatch.setattr(lifecycle_tasks, f"_{phase}", Mock(return_value={}))
    assert lifecycle_tasks._execute_phase("spark", store, phase, {}) == "completed"
    assert admission.calls == []


def test_unconfigured_phase_does_not_touch_monitoring(monkeypatch, admission):
    """Existing local callers need no Spark session or monitoring store."""
    store = _store()
    store.request.pop("monitoring_namespace")
    monkeypatch.setattr(lifecycle_tasks, "_operator", Mock(return_value={}))
    assert lifecycle_tasks._execute_phase(None, store, "operator", {}) == "completed"
    assert admission.calls == []


def test_failed_acquisition_prevents_alias_network_mutation(monkeypatch, tmp_path):
    """Refused admission aborts the prepared receipt without sending an alias request."""
    from skyulf.integrations.mlflow.lifecycle import promotion

    monkeypatch.setattr(
        reference_admission, "hold_reference_mutation", Mock(side_effect=RuntimeError("busy"))
    )
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_registered_model("c.s.m")
    for _ in range(2):
        client.create_model_version("c.s.m", source=tmp_path.as_uri())
    client.set_registered_model_alias("c.s.m", "champion", "1")
    assign_alias = Mock(wraps=client.set_registered_model_alias)
    delete_alias = Mock(wraps=client.delete_registered_model_alias)
    monkeypatch.setattr(client, "set_registered_model_alias", assign_alias)
    monkeypatch.setattr(client, "delete_registered_model_alias", delete_alias)
    store = _store()
    record_failure = Mock()

    def mutation(*args):
        """Reach the actual alias transport after the phase's ordinary validation."""
        receipt = promotion.AliasChangeReceipt(
            "event", "promotion", "c.s.m", "champion", "1", "2", "digest", None
        )
        promotion.commit_change(client, receipt, [("champion", "2", "1")])

    monkeypatch.setattr(lifecycle_tasks, "_operator", mutation)
    monkeypatch.setattr(lifecycle_tasks, "record_phase_failure", record_failure)
    with pytest.raises(RuntimeError, match="busy"):
        lifecycle_tasks._execute_phase("spark", store, "operator", {})
    assign_alias.assert_not_called()
    delete_alias.assert_not_called()
    assert str(client.get_model_version_by_alias("c.s.m", "champion").version) == "1"
    assert "pending_alias_event" not in client.get_registered_model("c.s.m").tags
    event = json.loads(client.get_model_version("c.s.m", "2").tags["promotion_event"])
    assert event["state"] == "aborted"
    record_failure.assert_called_once_with(store, "operator")
    store.complete.assert_not_called()
    assert promotion._BEFORE_ALIAS_WRITE.get() is None


def test_notebook_initialization_pins_monitoring_destination(monkeypatch):
    """Later widgets cannot redirect a durable lifecycle invocation to another guard."""
    values = {
        "lifecycle_action": "train",
        "monitoring_catalog": "catalog",
        "monitoring_schema": "monitoring",
        "workspace_host": "https://workspace.example",
    }
    monkeypatch.setattr(job_runtime, "read_notebook_config", lambda values: {})
    monkeypatch.setattr(job_runtime, "validate_workflow_config", lambda config, **kwargs: config)
    monkeypatch.setattr(job_runtime, "validate_monitoring_settings", Mock())
    request = job_runtime._prepared_notebook_request(values, None)
    values["monitoring_schema"] = "changed"
    assert request["monitoring_namespace"] == "catalog.monitoring"
    assert request["workspace_host"] == "https://workspace.example"


@pytest.mark.parametrize("fails", [False, True])
def test_model_set_decision_uses_frozen_branch_destination(monkeypatch, admission, fails):
    """A model-set alias decision uses the original project destination and host."""
    store = _store()
    store.request.pop("monitoring_namespace")
    store.request.pop("workspace_host")
    store.request["operator_values"] = {
        "monitoring_catalog": "frozen",
        "monitoring_schema": "central",
        "workspace_host": "https://original.example",
    }

    def decide(*args):
        """The project-level alias change executes inside admission."""
        assert admission.held
        if fails:
            raise RuntimeError("remote reply")
        return {}

    monkeypatch.setattr(model_set_stages, "_decide", decide)
    monkeypatch.setattr(model_set_stages, "record_phase_failure", lambda *args: None)
    if fails:
        with pytest.raises(RuntimeError, match="remote reply"):
            model_set_stages._execute("spark", store, "model_decision", {})
    else:
        assert model_set_stages._execute("spark", store, "model_decision", {}) == "completed"
    assert admission.calls[0][1] == "frozen.central"
    assert admission.calls[0][2]["workspace_host"] == "https://original.example"
    assert admission.exits == ["failed" if fails else "released"]


@pytest.mark.parametrize("fails", [False, True])
def test_branch_operator_is_guarded(monkeypatch, admission, fails):
    """Manual model-set approval and rollback cannot race reference cleanup."""
    from skyulf.integrations.databricks.model_sets import model_set_project

    store = _store()
    monkeypatch.setattr(branch_tasks, "_bound_store", lambda *args: store)
    monkeypatch.setattr(branch_tasks, "validate_active_phase", Mock())
    monkeypatch.setattr(job_runtime, "operator_options", lambda *args: {})

    def operator(*args, **kwargs):
        """Keep the actual alias mutation inside its shared scope."""
        assert admission.held
        if fails:
            raise RuntimeError("remote reply")
        return {}

    monkeypatch.setattr(model_set_project, "run_model_set_operator", operator)
    if fails:
        with pytest.raises(RuntimeError, match="remote reply"):
            branch_tasks.run_branch_operator(
                "spark", context=store.context, tracking_uri="uri", reference={}
            )
    else:
        assert (
            branch_tasks.run_branch_operator(
                "spark", context=store.context, tracking_uri="uri", reference={}
            )
            == "completed"
        )
    assert admission.exits == ["failed" if fails else "released"]


@dataclass
class _Outcome:
    """Provide the public branch payload required before nomination."""

    components: dict


def test_branch_registration_nomination_is_guarded(monkeypatch, admission):
    """Challenger nomination protects the newly referenced model-set components."""
    store = _store()
    store.request["branch_plan"] = {}
    store.log = Mock()
    store.run = Mock()
    monkeypatch.setattr(branch_tasks, "_bound_store", lambda *args: store)
    monkeypatch.setattr(branch_tasks, "validate_active_phase", Mock())
    monkeypatch.setattr(branch_tasks, "restore_training_branches", lambda *args: ())
    monkeypatch.setattr(branch_tasks, "_completed_branches", lambda *args: _Outcome({}))
    monkeypatch.setattr(branch_tasks, "log_progress", Mock())

    def nominate(*args):
        """Assert the packaging and nomination boundary owns reference admission."""
        assert admission.held
        return {}

    monkeypatch.setattr(branch_tasks, "_register_set", nominate)
    assert (
        branch_tasks.register_branch_set(
            "spark", context=store.context, tracking_uri="uri", reference={}
        )
        == "completed"
    )
    assert admission.calls[0][2]["task_key"] == "register_model_set"


@pytest.mark.parametrize("fails", [False, True])
def test_central_enrollment_is_guarded(monkeypatch, admission, fails):
    """Every enrollment writer protects its published reference, including direct callers."""
    from skyulf.integrations.databricks.observability.monitoring import monitoring_store

    monkeypatch.setattr(monitoring_store, "inventory_row", lambda *args: {"monitor_id": "id"})

    def merge(*args, **kwargs):
        """The inventory cannot become protected after cleanup has admitted deletion."""
        assert admission.held
        if fails:
            raise RuntimeError("remote reply")

    monkeypatch.setattr(monitoring_store, "_merge_row", merge)
    if fails:
        with pytest.raises(RuntimeError, match="remote reply"):
            monitoring_store.enroll_monitor("spark", "catalog.monitoring", Mock())
    else:
        monitoring_store.enroll_monitor("spark", "catalog.monitoring", Mock())
    assert admission.exits == ["failed" if fails else "released"]


@pytest.mark.parametrize("fails", [False, True])
def test_rollout_promotion_is_guarded(monkeypatch, admission, fails):
    """A completed traffic rollout cannot activate references while cleanup is deleting them."""
    from skyulf.integrations.databricks.jobs import serving_job
    from skyulf.integrations.databricks.serving.rollout import RolloutResult
    from skyulf.integrations.databricks.serving.rollout_policy import RolloutState
    from skyulf.integrations.mlflow.lifecycle.promotion import AliasChangeReceipt

    state = RolloutState(
        "rollout",
        "endpoint",
        "c.s.m",
        "1",
        "c.s.m",
        "2",
        100,
        "2026-10-18T00:00:00+00:00",
        "2026-10-18T00:00:00+00:00",
        phase="COMPLETE",
    )
    result = RolloutResult(state, "COMMITTED", "receipt")

    settings = {
        "monitoring": {"reference_namespace": "catalog.monitoring"},
        "reference_holder": {
            "job_run_id": 123,
            "task_key": "daily_rollout",
            "workspace_host": "https://workspace.example",
        },
    }

    def promote(*args, **kwargs):
        """Assert alias activation occurs after acquiring the configured store owner."""
        assert admission.held
        if fails:
            raise RuntimeError("remote reply")
        return AliasChangeReceipt(
            "event", "promotion", "c.s.m", "champion", "1", "2", "digest", None
        )

    monkeypatch.setattr(serving_job, "promote_completed_rollout", promote)
    if fails:
        with pytest.raises(RuntimeError, match="remote reply"):
            serving_job._output(Mock(), "spark", {}, Mock(), result, settings)
    else:
        assert (
            serving_job._output(Mock(), "spark", {}, Mock(), result, settings)["promotion"][
                "new_version"
            ]
            == "2"
        )
    assert admission.calls[0][2] == settings["reference_holder"]
    assert admission.exits == ["failed" if fails else "released"]


def test_prepared_request_persists_monitoring_context(monkeypatch):
    """Request integrity covers the destination and workspace used by later alias phases."""
    store = _store()
    store.context.identity = lambda: {"job_id": "1", "job_run_id": "123"}
    store.context.job_id = "1"
    store.client.search_runs.return_value = []
    store.client.create_run.return_value.info.run_id = "new-run"
    store.log = Mock()
    monkeypatch.setattr(lifecycle_tasks, "get_or_create_experiment", lambda *args: "experiment")
    monkeypatch.setattr(
        lifecycle_tasks.workflow, "workflow_policies", lambda *args: (None, "manual")
    )
    monkeypatch.setattr(lifecycle_tasks, "_prepared_output", lambda *args: {})
    from skyulf.integrations.databricks.observability.monitoring import monitoring_store

    bootstrap = Mock()
    monkeypatch.setattr(monitoring_store, "ensure_monitoring_store", bootstrap)
    lifecycle_tasks._prepare(
        "spark",
        store,
        {},
        "approve",
        "experiment",
        {},
        None,
        monitoring_namespace="catalog.monitoring",
        workspace_host="https://workspace.example",
    )
    saved = store.log.call_args.args[1]
    assert saved["monitoring_namespace"] == "catalog.monitoring"
    assert saved["workspace_host"] == "https://workspace.example"
    bootstrap.assert_called_once_with("spark", "catalog", "monitoring")


@pytest.mark.parametrize("changed", [False, True])
def test_enrollment_uses_frozen_namespace_and_recovery_owner(monkeypatch, admission, changed):
    """A later notebook cannot enroll into a different namespace after guarded activation."""
    from skyulf.integrations.databricks.jobs.monitoring import monitoring_tasks

    store = _store()
    store.receipt = Mock(return_value={"output": {}})
    store.client.get_run.return_value.info.start_time = 123
    values = {
        "monitoring_catalog": "catalog",
        "monitoring_schema": "changed" if changed else "monitoring",
    }
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = values
    monkeypatch.setattr(monitoring_tasks, "deployment_store", lambda *args: store)

    def register(*args, **kwargs):
        """Propagate provenance; the service locks only the inventory/publication writes."""
        assert not admission.held
        assert kwargs["reference_holder"]["task_key"] == "register_monitor"
        return {"monitor_id": "monitor"}

    monkeypatch.setattr(monitoring_tasks, "register_deployed_monitor", register)
    if changed:
        with pytest.raises(ValueError, match="frozen lifecycle"):
            monitoring_tasks.run_monitor_enrollment_notebook("spark", dbutils)
        assert admission.calls == []
    else:
        assert (
            monitoring_tasks.run_monitor_enrollment_notebook("spark", dbutils)["status"]
            == "enrolled"
        )
        assert admission.calls == []


def test_project_prepare_carries_generated_holder(monkeypatch, admission):
    """A failed scheduled preparation retains enough provenance for verified recovery."""
    from skyulf.integrations.databricks.jobs.monitoring import spark_monitoring_job

    values = {
        "monitoring_catalog": "catalog",
        "monitoring_schema": "monitoring",
        "job_run_id": "123",
        "task_key": "monitor_model",
        "workspace_host": "https://workspace.example",
    }
    config = SimpleNamespace(monitor_id="id")
    monkeypatch.setattr(spark_monitoring_job, "project_configs", lambda *args: [config])

    def prepare(*args, **kwargs):
        """Reference publication receives owner metadata without locking its expensive reads."""
        assert not admission.held
        assert kwargs["reference_holder"]["task_key"] == "monitor_model"

    monkeypatch.setattr(spark_monitoring_job, "prepare_spark_monitoring_reference", prepare)
    assert spark_monitoring_job.prepare_project_references("spark", values)["models"] == 1
    assert admission.calls == []


def test_activation_enrollment_does_not_lock_expensive_reference_preparation(
    monkeypatch, admission
):
    """The enrolled version protects the gap while source preparation runs without the shared lock."""
    from skyulf.integrations.databricks.observability.monitoring import monitoring_registration
    from skyulf.integrations.databricks.observability.monitoring.spark import (
        spark_monitoring_reference,
    )

    config = SimpleNamespace(enabled=True, execution_engine="spark", monitor_id="monitor")
    monkeypatch.setattr(
        monitoring_registration, "build_monitor_enrollment_config", lambda *args: config
    )
    monkeypatch.setattr(monitoring_registration, "validate_monitoring_settings", Mock())
    monkeypatch.setattr(monitoring_registration, "ensure_monitoring_store", Mock())
    events = []

    def enroll(*args, **kwargs):
        """Inventory publication occurs under the job's recoverable owner."""
        with reference_admission.hold_reference_mutation(
            args[0], args[1], **kwargs["reference_holder"]
        ):
            assert admission.held
        events.append("enrolled")

    def prepare(*args, **kwargs):
        """Inner preparation receives provenance and acquires only at publication."""
        assert not admission.held
        assert kwargs["reference_holder"]["task_key"] == "register_monitor"
        events.append("prepare")

    monkeypatch.setattr(monitoring_registration, "enroll_monitor", enroll)
    monkeypatch.setattr(spark_monitoring_reference, "prepare_spark_monitoring_reference", prepare)
    monitoring_registration.register_deployed_monitor(
        "spark",
        {"model_name": "c.s.model"},
        {"monitoring_catalog": "catalog", "monitoring_schema": "monitoring"},
        {
            "action": "approve",
            "result": {"kind": "promotion", "model_name": "c.s.model", "new_version": "2"},
        },
        activation_started_ms=123,
        reference_holder={
            "job_run_id": 123,
            "task_key": "register_monitor",
            "workspace_host": "https://workspace.example",
        },
    )
    assert events == ["enrolled", "prepare"]
    assert admission.exits == ["released"]


def test_partial_legacy_widgets_use_manual_recovery_without_invented_task_identity():
    """Older jobs remain usable while only complete generated provenance enables automatic verification."""
    from skyulf.integrations.databricks.jobs.shared.reference_mutations import (
        notebook_reference_holder,
    )

    assert notebook_reference_holder({"job_run_id": "123"}) == {}
    with pytest.raises(ValueError, match="task_key"):
        notebook_reference_holder(
            {"job_run_id": "123", "task_key": "", "workspace_host": "https://workspace.example"}
        )


@pytest.mark.parametrize("task_key", ["score", "recover_predictions"])
def test_scoring_enrollment_forwards_generated_owner(monkeypatch, task_key):
    """Normal and recovered scoring leave a recoverable owner if their inventory write fails."""
    from skyulf.integrations.databricks.observability.monitoring import monitoring_registration

    config = SimpleNamespace(monitor_id="monitor", payload=lambda: {})
    monkeypatch.setattr(
        monitoring_registration, "build_monitor_enrollment_config", lambda *args: config
    )
    monkeypatch.setattr(monitoring_registration, "validate_monitoring_settings", Mock())
    monkeypatch.setattr(monitoring_registration, "ensure_monitoring_store", Mock())
    enroll = Mock()
    monkeypatch.setattr(monitoring_registration, "enroll_monitor", enroll)
    values = {
        "monitoring_catalog": "catalog",
        "monitoring_schema": "monitoring",
        "job_run_id": "123",
        "task_key": task_key,
        "workspace_host": "https://workspace.example",
    }
    result = monitoring_registration.register_scoring_monitor(
        "spark",
        {"model_name": "c.s.model"},
        values,
        {"result": {"selected_model_name": "c.s.model", "selected_model_version": "2"}},
    )
    assert result is not None and result["monitor_id"] == "monitor"
    assert enroll.call_args.kwargs["reference_holder"] == {
        "job_run_id": 123,
        "task_key": task_key,
        "workspace_host": "https://workspace.example",
    }
