"""Native child jobs receive only validated, exact producer monitoring receipts."""

import json
from unittest.mock import Mock

import pytest
from test_monitoring_registration import settings, workflow
from test_monitoring_tasks import request


def spark_request(**changes):
    """Build a real pinned Spark enrollment with an immutable scoring commit."""
    value = request()
    value["config"].update(execution_engine="spark", reference_namespace="ops.monitoring")
    return value | changes


def prepare(monkeypatch, receipt, *, values=None, recovery=False):
    """Expose notebook inputs and capture published task values without Databricks."""
    from skyulf.integrations.databricks import monitoring_tasks as tasks

    dbutils = Mock()
    dbutils.widgets.getAll.return_value = values or settings()
    monkeypatch.setattr(
        tasks, "read_notebook_config", lambda _: workflow(auto_rebuild_on_cdf_expiry=recovery)
    )
    inputs = {
        ("score", "recovery_required"): recovery,
        ("recover_predictions" if recovery else "score", "monitoring_request"): receipt,
    }
    dbutils.jobs.taskValues.get.side_effect = lambda taskKey, key: inputs[taskKey, key]
    output = tasks.prepare_scoring_monitoring_notebook(dbutils)
    published = {
        call.kwargs["key"]: call.kwargs["value"]
        for call in dbutils.jobs.taskValues.set.call_args_list
    }
    return output, published, dbutils


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("noop", [False, True])
def test_native_handoff_preserves_successful_receipt(monkeypatch, recovery, noop):
    """Recovery and repaired scoring must dispatch the exact saved commit and version."""
    receipt = spark_request(noop=noop, has_saved_batch=True)
    output, published, _ = prepare(monkeypatch, receipt, recovery=recovery)
    assert output == {"status": "ready"}
    assert published["monitoring_ready"] is True
    assert json.loads(published["monitoring_request_json"]) == receipt


@pytest.mark.parametrize(
    "values",
    [settings(monitoring_enabled="false"), settings(monitoring_deployment_mode="development")],
)
def test_native_handoff_disabled_does_not_read_receipt(monkeypatch, values):
    """Paused or ephemeral deployments must exclude the native child job entirely."""
    output, published, dbutils = prepare(monkeypatch, None, values=values)
    assert output == {"status": "disabled"}
    assert published == {"monitoring_ready": False, "monitoring_request_json": ""}
    dbutils.jobs.taskValues.get.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"execution_engine": "local", "reference_namespace": None},
        {"model_name": "models.risk.other"},
        {"project": "other"},
        {"environment": "other"},
        {"reference_namespace": "other.monitoring"},
        {"model_version": None, "model_alias": "champion"},
        {"prediction_table": "outputs.risk.other"},
    ],
)
def test_native_handoff_rejects_unbound_config(monkeypatch, change):
    """Native Spark dispatch cannot resolve mutable aliases or accept another producer."""
    receipt = spark_request()
    receipt["config"].update(change)
    with pytest.raises(ValueError):
        prepare(monkeypatch, receipt)


@pytest.mark.parametrize(
    "changes",
    [
        {"namespace": "other.monitoring"},
        {"commit_version": None},
        {"commit_version": True},
        {"commit_version": -1},
        {"noop": "false"},
        {"has_saved_batch": "false"},
        {"configs": []},
    ],
)
def test_native_handoff_rejects_invalid_receipt(monkeypatch, changes):
    """Invalid commit or batch metadata must fail before the ready condition is published."""
    with pytest.raises(ValueError):
        prepare(monkeypatch, spark_request(**changes))


def test_native_handoff_noop_without_predictions_is_skipped(monkeypatch):
    """No-op scoring without performance monitoring must preserve previous observation health."""
    output, published, _ = prepare(monkeypatch, spark_request(noop=True, commit_version=None))
    assert output == {"status": "no_new_predictions"}
    assert published["monitoring_ready"] is False


def test_native_handoff_rejects_ambiguous_single_and_set_bindings(monkeypatch):
    """Two competing configuration forms cannot silently choose another scoring binding."""
    receipt = spark_request()
    receipt["configs"] = [receipt["config"]]
    with pytest.raises(ValueError, match="configuration form"):
        prepare(monkeypatch, receipt)


def test_native_handoff_noop_keeps_late_label_observation(monkeypatch):
    """No new predictions must still dispatch enabled performance windows without an invocation ID."""
    from test_monitoring_config import performance_policy

    receipt = spark_request(noop=True, commit_version=None)
    receipt["config"].update(
        performance_policy=performance_policy(),
        label_table="labels.risk.actuals",
        result_available_at_column="available_at",
    )
    output, published, _ = prepare(monkeypatch, receipt)
    assert output == {"status": "ready"}
    assert json.loads(published["monitoring_request_json"]) == receipt


def test_native_handoff_keeps_pinned_model_set_components(monkeypatch):
    """A shared physical batch must retain each component and the original parent version."""
    from skyulf.integrations.databricks import model_set_project
    from skyulf.integrations.databricks import monitoring_tasks as tasks

    receipt = spark_request()
    base = receipt.pop("config")
    receipt["configs"] = [
        base
        | {
            "model_name": "models.risk." + branch,
            "model_version": version,
            "model_set_name": "models.risk.set",
            "model_set_version": "9",
            "model_set_branch": branch,
            "prediction_table": "outputs.risk.set_scores",
        }
        for branch, version in [("revenue", "2"), ("cost", "4")]
    ]
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings()
    dbutils.jobs.taskValues.get.return_value = receipt
    monkeypatch.setattr(
        tasks, "read_notebook_config", lambda _: workflow(training_layout="multi_target")
    )
    monkeypatch.setattr(
        model_set_project,
        "load_project_model_set",
        lambda *args: {
            "model_name": "models.risk.set",
            "prediction_table": "outputs.risk.set_scores",
        },
    )
    assert tasks.prepare_scoring_monitoring_notebook(dbutils) == {"status": "ready"}
    published = {
        call.kwargs["key"]: call.kwargs["value"]
        for call in dbutils.jobs.taskValues.set.call_args_list
    }
    assert json.loads(published["monitoring_request_json"]) == receipt
    receipt["configs"][1]["model_set_version"] = "10"
    with pytest.raises(ValueError, match="pinned model-set"):
        tasks.prepare_scoring_monitoring_notebook(dbutils)
