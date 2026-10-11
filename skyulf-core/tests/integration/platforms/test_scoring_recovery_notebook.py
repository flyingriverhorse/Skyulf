"""Recovery routing keeps CDF failures distinct from committed prediction output."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def recovery_request():
    """Return the pinned predecessor evidence needed by a separate recovery task."""
    return {
        "version": 1,
        "layout": "single_model",
        "source_table": "workspace.test.source",
        "source_table_id": "source-id",
        "target_table": "workspace.test.predictions",
        "target_table_id": "target-id",
        "target_version": 3,
        "source_start_version": 5,
        "source_end_version": 9,
        "model_name": "workspace.test.model",
        "model_version": "2",
        "model_digest": "a" * 64,
    }


def task_utils():
    """Capture task values without installing a Databricks execution context."""
    values = {}
    task = Mock()
    task.set.side_effect = lambda *, key, value: values.update({key: value})
    return SimpleNamespace(jobs=SimpleNamespace(taskValues=task)), values


@pytest.mark.parametrize("enabled", [False, True])
def test_only_opted_in_expiry_requests_a_separate_recovery_task(enabled):
    """Disabled projects must fail instead of silently replacing old predictions."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import CdfRecoveryRequired
    from skyulf.integrations.databricks.scoring.incremental.scoring_recovery import run_scoring_step

    dbutils, values = task_utils()
    callback = Mock(side_effect=CdfRecoveryRequired(recovery_request()))
    config = {"auto_rebuild_on_cdf_expiry": enabled}
    if not enabled:
        with pytest.raises(CdfRecoveryRequired):
            run_scoring_step(config, dbutils, callback)
        assert not values
    else:
        output = run_scoring_step(config, dbutils, callback)
        assert output["recovery_required"] is True
        assert values["recovery_required"] is True
        assert values["cdf_recovery_request"] == recovery_request()
        assert "scoring_result" not in values


@pytest.mark.parametrize(
    "error", [PermissionError("denied"), OSError("offline"), ValueError("schema")]
)
def test_ordinary_errors_never_request_recovery(error):
    """Automatic recovery must not turn configuration or infrastructure failures into writes."""
    from skyulf.integrations.databricks.scoring.incremental.scoring_recovery import run_scoring_step

    dbutils, values = task_utils()
    with pytest.raises(type(error), match=str(error)):
        run_scoring_step({"auto_rebuild_on_cdf_expiry": True}, dbutils, Mock(side_effect=error))
    assert not values


def test_normal_result_is_preserved_and_large_temporal_state_is_not_a_task_value():
    """Task-value limits must not fail an otherwise committed scoring batch."""
    from skyulf.integrations.databricks.scoring.incremental.scoring_recovery import run_scoring_step

    dbutils, values = task_utils()
    payload = {
        "action": "score",
        "result": {"noop": True, "manifest": {"temporal_history": "x" * 70000}},
    }
    output = run_scoring_step({"auto_rebuild_on_cdf_expiry": True}, dbutils, lambda: payload)
    assert output is payload
    assert values["recovery_required"] is False
    assert values["scoring_result"]["result"]["noop"] is True
    assert "temporal_history" not in values["scoring_result"]["result"]["manifest"]


@pytest.mark.parametrize("recovered", [False, True])
def test_report_reads_only_the_branch_that_ran(monkeypatch, recovered):
    """Skipped recovery tasks must not break successful normal score summaries."""
    from skyulf.integrations.databricks.scoring.incremental import scoring_recovery as module

    dbutils, _ = task_utils()
    dbutils.widgets = SimpleNamespace(getAll=lambda: {})
    monkeypatch.setattr(
        module, "read_notebook_config", lambda values: {"auto_rebuild_on_cdf_expiry": True}
    )
    payload = {"action": "score", "result": {"noop": False, "output_count": 3}}
    result_task = "recover_predictions" if recovered else "score"

    def get(*, taskKey, key):
        """Reject any attempt to read a value from an excluded task."""
        if taskKey == "score" and key == "recovery_required":
            return recovered
        assert (taskKey, key) == (result_task, "scoring_result")
        return payload

    dbutils.jobs.taskValues.get.side_effect = get
    output = module.run_scoring_report_notebook(None, dbutils, exit_notebook=False)
    assert '"output_count": 3' in output
    assert dbutils.jobs.taskValues.get.call_count == 2


def test_recovery_node_refuses_disabled_config_before_reading_task_values(monkeypatch):
    """A stale graph must not grant recovery permission when the project disables it."""
    from skyulf.integrations.databricks.scoring.incremental import scoring_recovery as module

    dbutils, _ = task_utils()
    dbutils.widgets = SimpleNamespace(getAll=lambda: {})
    monkeypatch.setattr(module, "read_notebook_config", lambda values: {})
    with pytest.raises(ValueError, match="enabled"):
        module.run_cdf_recovery_notebook(None, dbutils, exit_notebook=False)
    dbutils.jobs.taskValues.get.assert_not_called()


@pytest.mark.parametrize("active_generation", ["predictions_v2", "predictions_v1", None])
def test_empty_recovered_generation_remains_readable_without_new_view_activation(active_generation):
    """A recovered empty active generation is valid, but a new empty generation cannot replace it."""
    from skyulf.integrations.databricks.scoring.shared.prediction_output import (
        activate_prediction_view,
    )

    spark = Mock()
    spark.table.return_value.limit.return_value.count.return_value = 0
    spark.table.return_value.schema.fields = []
    spark.catalog.tableExists.return_value = active_generation is not None
    spark.catalog.getTable.return_value.tableType = "VIEW"
    spark.sql.return_value.first.side_effect = [
        {"value": "full_rebuild"},
        {
            "createtab_stmt": f"CREATE VIEW workspace.test.predictions AS SELECT * FROM workspace.test.{active_generation}"
        },
    ]
    if active_generation == "predictions_v2":
        activate_prediction_view(
            spark, "workspace.test.predictions", "workspace.test.predictions_v2"
        )
        assert all(
            not call.args[0].startswith(("ALTER", "CREATE")) for call in spark.sql.call_args_list
        )
    else:
        with pytest.raises(ValueError, match="contain predictions"):
            activate_prediction_view(
                spark, "workspace.test.predictions", "workspace.test.predictions_v2"
            )


def test_recovery_notebook_pins_version_and_publishes_only_completed_result(monkeypatch):
    """An alias change between tasks cannot redirect the saved recovery request."""
    from skyulf.integrations.databricks.lifecycle.workflow import BundleActionResult
    from skyulf.integrations.databricks.scoring.incremental import scoring_recovery as module

    dbutils, values = task_utils()
    dbutils.widgets = SimpleNamespace(getAll=lambda: {})
    config = {"auto_rebuild_on_cdf_expiry": True, "score_source_table": "workspace.test.source"}
    monkeypatch.setattr(module, "read_notebook_config", lambda values: config)
    dbutils.jobs.taskValues.get.side_effect = [True, recovery_request()]
    runner = Mock(
        return_value=BundleActionResult("score", {"noop": False, "output_count": 3}, False, {})
    )
    monkeypatch.setattr(module, "run_bundle_action", runner)
    result = module.run_cdf_recovery_notebook(None, dbutils, exit_notebook=False)
    assert runner.call_args.args[2] == {"score_model_version": "2"}
    assert runner.call_args.kwargs["recovery_request"] == recovery_request()
    assert values["scoring_result"]["result"]["output_count"] == 3
    assert '"output_count": 3' in result


def test_full_rebuild_empty_recovery_replay_and_normal_noop_reach_report(monkeypatch):
    """Workflow activation must not turn a valid empty committed snapshot into a failed job."""
    from tests.integration.platforms.test_databricks_lifecycle_workflow import _config

    from skyulf.integrations.databricks.lifecycle import workflow as workflow
    from skyulf.integrations.databricks.scoring.incremental.incremental_batch import (
        IncrementalBatchResult,
    )

    config = _config()
    config.update(model_change_mode="full_rebuild", model_version="2")
    spark = Mock()
    spark.table.return_value.limit.return_value.count.return_value = 0
    spark.table.return_value.schema.fields = []
    spark.catalog.tableExists.return_value = True
    spark.catalog.getTable.return_value.tableType = "VIEW"
    spark.sql.return_value.first.return_value = {
        "value": "full_rebuild",
        "createtab_stmt": f"CREATE VIEW {config['prediction_table']} AS SELECT * FROM {config['prediction_table']}_v2",
    }
    monkeypatch.setattr(workflow, "prepare_workflow", Mock())
    provision = Mock()
    monkeypatch.setattr(workflow, "provision_prediction_table", provision)
    batch = Mock(
        side_effect=[
            IncrementalBatchResult(None, 9, 0, 0, 4, {}, False, config["model_name"], "2"),
            IncrementalBatchResult(5, 9, 0, 0, 4, {}, True, config["model_name"], "2"),
            IncrementalBatchResult(9, 9, 0, 0, 4, {}, True, config["model_name"], "2"),
        ]
    )
    monkeypatch.setattr(workflow, "run_incremental_batch", batch)
    outcomes = [
        workflow.run_action(spark, config, "score", recovery_request=request)
        for request in (recovery_request(), recovery_request(), None)
    ]
    assert [result.noop for result in outcomes] == [False, True, True]
    assert all(result.commit_version == 4 for result in outcomes)
    assert provision.call_count == 1
    assert not any(
        call.args[0].startswith(("ALTER", "CREATE")) for call in spark.sql.call_args_list
    )


def test_model_set_recovery_uses_saved_version_without_resolving_champion(
    tmp_path,
    workflow_config,
    monkeypatch,
):
    """A changed alias or inherited version parameter cannot override the predecessor pin."""
    pytest.importorskip("mlflow")
    from tests.integration.platforms.test_model_set_project import _enable, _project

    from skyulf.integrations.databricks.model_sets import model_set_batch, model_set_project
    from skyulf.integrations.mlflow.models import model_set
    from skyulf.integrations.mlflow.registration.registry import ResolvedModel

    values, _ = _project(tmp_path, workflow_config)
    _enable(tmp_path)
    values["score_model_version"] = "99"
    config = model_set_project.read_notebook_config(values)
    request = {**recovery_request(), "layout": "model_set"}
    model = ResolvedModel("workspace.models.example_set_dev", "2", "models:/set/2", None, "a" * 64)
    champion = Mock(side_effect=AssertionError("must not resolve champion"))
    resolve = Mock(return_value=model)
    monkeypatch.setattr(model_set_project, "controlled_champion_version", champion)
    monkeypatch.setattr(model_set_project, "resolve_model", resolve)
    monkeypatch.setattr(model_set, "load_registered_model_set", Mock(return_value=object()))
    monkeypatch.setattr(model_set_project, "publication_views", lambda *args: ())
    batch = Mock(return_value=model_set_batch.ModelSetBatchResult(9, 3, 3, 4, {}, False))
    monkeypatch.setattr(model_set_batch, "run_model_set_batch", batch)
    result = model_set_project.score_model_set_payload(
        None, config, values, recovery_request=request
    )
    assert resolve.call_args.kwargs["version"] == "2"
    assert batch.call_args.kwargs["recovery_request"] == request
    assert result["output_count"] == 3
    champion.assert_not_called()
