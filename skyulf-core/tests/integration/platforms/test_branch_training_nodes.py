"""Separate branch tasks keep independent target populations and complete-only assembly."""

from typing import Any

import pytest
from test_local_branches import _configs, _data, tracked  # noqa: F401 - shared tracking fixture


def test_branch_tasks_join_only_when_every_model_finished(workflow_config, tracked, monkeypatch):
    """Regression, classification and ensemble branches must share one immutable parent plan."""
    from skyulf.integrations.databricks import branch_tasks, local_retraining
    from skyulf.integrations.databricks._lifecycle_state import LifecycleContext

    uri, client = tracked
    configs = _configs(workflow_config, store=uri)
    monkeypatch.setattr(
        local_retraining,
        "read_training_snapshot",
        lambda spark, spec: _data().loc[:, list(spec.source_columns)].copy(),
    )
    context = LifecycleContext("10", "20")
    initialized = branch_tasks.initialize_branch_training(
        None,
        configs=configs,
        settings=None,
        composition_source="",
        context=context,
        tracking_uri=uri,
        experiment_name="branches",
    )
    options: dict[str, Any] = {
        "reference": initialized.reference,
        "tracking_uri": uri,
        "context": context,
    }
    first = branch_tasks.run_branch_training(None, name="amount", **options)
    with pytest.raises(ValueError, match="already attempted"):
        branch_tasks.run_branch_training(None, name="amount", **options)
    with pytest.raises(ValueError, match="completed receipt"):
        branch_tasks.assemble_branch_training(None, **options)
    branch_tasks.run_branch_training(None, name="category", **options)
    branch_tasks.run_branch_training(None, name="ensemble", **options)
    result = branch_tasks.assemble_branch_training(None, **options)
    assert set(result.output["components"]) == {"amount", "category", "ensemble"}
    assert first.output["run_id"] != result.output["parent_run_id"]
    assert client.get_run(result.output["parent_run_id"]).info.status == "FINISHED"
    assert all(not model.aliases for model in client.search_registered_models())
    with pytest.raises(ValueError, match="active|attempted"):
        branch_tasks.assemble_branch_training(None, **options)


def test_branch_failure_finalizes_parent_without_partial_set(workflow_config, tracked, monkeypatch):
    """A failed child blocks assembly and the ALL_DONE report closes the incomplete parent."""
    import json
    from types import SimpleNamespace

    from skyulf.integrations.databricks import branch_tasks
    from skyulf.integrations.databricks._lifecycle_state import LifecycleContext
    from skyulf.integrations.databricks.training_node_notebook import run_models_report_notebook

    uri, client = tracked
    initialized = branch_tasks.initialize_branch_training(
        None,
        configs=_configs(workflow_config, store=uri),
        settings=None,
        composition_source="",
        context=LifecycleContext("10", "20"),
        tracking_uri=uri,
        experiment_name="branches",
    )

    def fail(*args, **kwargs):
        """Represent a failure before a branch has a complete registered candidate."""
        raise ValueError("deliberate failure")

    monkeypatch.setattr(branch_tasks, "train_branch", fail)
    with pytest.raises(ValueError, match="deliberate"):
        branch_tasks.run_branch_training(
            None,
            name="amount",
            context=LifecycleContext("10", "20"),
            tracking_uri=uri,
            reference=initialized.reference,
        )
    values = {
        "workflow_contract": "3",
        "job_id": "10",
        "job_run_id": "20",
        "repair_count": "0",
        "execution_count": "1",
        "tracking_uri": uri,
        "reference_json": json.dumps(initialized.reference),
        "decision_result_state": "upstream_failed",
    }
    with pytest.raises(ValueError, match="failed"):
        run_models_report_notebook(SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values)))
    assert client.get_run(initialized.reference["run_id"]).info.status == "FAILED"
    assert not client.search_registered_models()


@pytest.mark.parametrize("action", ["approve", "reject", "rollback"])
def test_branch_operator_uses_frozen_configuration(workflow_config, tracked, monkeypatch, action):
    """The operator branch must keep existing controls without reloading training recipes."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks import (
        branch_tasks,
        job_runtime,
        model_set_project,
        model_set_stages,
    )
    from skyulf.integrations.databricks._lifecycle_state import LifecycleContext

    uri, client = tracked
    configs = _configs(workflow_config, store=uri)
    initialized = branch_tasks.initialize_branch_training(
        None,
        configs=configs,
        settings={"model_name": "workspace.test.set"},
        composition_source="",
        context=LifecycleContext("10", "20"),
        tracking_uri=uri,
        experiment_name="branches",
        action=action,
        operator_values={"lifecycle_action": action},
    )
    configs["amount"]["score_source_table"] = "changed.table.name"
    operator = Mock(return_value={"action": action})
    monkeypatch.setattr(model_set_project, "run_model_set_operator", operator)
    monkeypatch.setattr(job_runtime, "operator_options", lambda *args: {})
    result = model_set_stages.run_model_set_phase(
        None,
        phase="model_decision",
        context=LifecycleContext("10", "20"),
        tracking_uri=uri,
        reference=initialized.reference,
    )
    assert result.output["action"] == action
    assert operator.call_args.kwargs["config"]["score_source_table"] != "changed.table.name"
    assert client.get_run(initialized.reference["run_id"]).info.status == "FINISHED"


def test_set_registration_evaluation_and_decision_are_separate(
    workflow_config, tracked, monkeypatch
):
    """The graph must expose real lifecycle work in each node without early promotion."""
    from types import SimpleNamespace

    from skyulf.integrations.databricks import branch_tasks, local_retraining, model_set_stages
    from skyulf.integrations.databricks._lifecycle_state import LifecycleContext

    uri, client = tracked
    configs = _configs(workflow_config, store=uri)
    configs = {"amount": configs["amount"], "category": configs["category"]}
    settings = {
        "model_name": "workspace.test.visible_set",
        "composition_config": {"outputs": []},
        "promotion_policy": "manual_approval",
    }
    monkeypatch.setattr(
        local_retraining,
        "read_training_snapshot",
        lambda spark, spec: _data().loc[:, list(spec.source_columns)].copy(),
    )
    from unittest.mock import Mock

    spark = Mock()
    spark.read.option.return_value.table.return_value = SimpleNamespace(dtypes=[("id", "bigint")])
    context = LifecycleContext("10", "20")
    initialized = branch_tasks.initialize_branch_training(
        spark,
        configs=configs,
        settings=settings,
        composition_source="",
        context=context,
        tracking_uri=uri,
        experiment_name="branches",
    )
    options: dict[str, Any] = {
        "context": context,
        "tracking_uri": uri,
        "reference": initialized.reference,
    }
    for name in configs:
        branch_tasks.run_branch_training(spark, name=name, **options)
    registered = model_set_stages.run_model_set_phase(spark, phase="register_model_set", **options)
    assert registered.output["model_set_candidate"]["version"] == "1"
    assert {
        key: str(value)
        for key, value in client.get_registered_model(settings["model_name"]).aliases.items()
    } == {"challenger": "1"}
    evaluated = model_set_stages.run_model_set_phase(
        spark,
        phase="evaluate_model_set",
        context=context,
        tracking_uri=uri,
        reference=registered.reference,
    )
    assert evaluated.output["quality"]["passed"] is False
    assert set(evaluated.output["quality"]["components"]) == set(configs)
    assert {
        key: str(value)
        for key, value in client.get_registered_model(settings["model_name"]).aliases.items()
    } == {"challenger": "1"}
    decided = model_set_stages.run_model_set_phase(spark, phase="model_decision", **options)
    assert decided.output["promotion_policy"] == "manual_approval"
    assert decided.output["alias_change"] is None
    assert {
        key: str(value)
        for key, value in client.get_registered_model(settings["model_name"]).aliases.items()
    } == {"challenger": "1"}
