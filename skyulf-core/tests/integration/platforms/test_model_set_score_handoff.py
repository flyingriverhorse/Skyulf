"""Whole-set scoring starts only after a verified successful alias transition."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_databricks_branch_template import _project
from test_local_branches import tracked  # noqa: F401 - shared tracking fixture
from test_model_set_project import _enable


def test_set_handoff_does_not_enable_component_handoff(tmp_path, workflow_config):
    """A parent handoff must retain manual activation and disabled scoring per component."""
    from skyulf.integrations.databricks.branch_notebook import load_training_branch_configs
    from skyulf.integrations.databricks.model_set_project import load_project_model_set

    values, _ = _project(tmp_path, workflow_config)
    path = Path(values["config_path"])
    path.write_text(
        json.dumps({**json.loads(path.read_text()), "score_handoff": "after_alias_change"})
    )
    values["deployed_score_handoff"] = "after_alias_change"
    _enable(tmp_path)
    assert load_project_model_set(values) is not None
    branches = load_training_branch_configs(values)
    assert all(config["score_handoff"] == "disabled" for config in branches.values())
    assert all(config["promotion_policy"] == "manual_approval" for config in branches.values())


@pytest.mark.parametrize("handoff", ["disabled", "after_alias_change"])
@pytest.mark.parametrize(
    "action,kind",
    [
        ("train", "initial"),
        ("train", "promotion"),
        ("approve", "promotion"),
        ("rollback", "rollback"),
        ("reject", "rejection"),
        ("train", None),
    ],
)
def test_final_report_publishes_score_condition(
    workflow_config, monkeypatch, handoff, action, kind
):
    """The child job must be gated by the saved transition, never by policy alone."""
    from skyulf.integrations.databricks import training_node_notebook as module
    from skyulf.integrations.mlflow.promotion import AliasChangeReceipt

    receipt = (
        asdict(
            AliasChangeReceipt(
                "event", kind, "workspace.models.set", "champion", None, "1", None, None
            )
        )
        if kind
        else None
    )
    output = {"alias_change" if action == "train" else "receipt": receipt}
    store = Mock()
    store.request = {
        "action": action,
        "config": workflow_config,
        "score_handoff": handoff,
        "settings": {"model_name": "workspace.models.set"},
    }
    store.receipt.return_value = {"output": output}
    monkeypatch.setattr(module, "_saved", lambda values: (store, {}))
    values = {"decision_result_state": "success"}
    dbutils = SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values), jobs=Mock())
    result = json.loads(module.run_models_report_notebook(dbutils))
    expected = handoff == "after_alias_change" and kind in {"initial", "promotion", "rollback"}
    assert result["score_requested"] is expected
    dbutils.jobs.taskValues.set.assert_called_once_with(key="score_requested", value=expected)


def test_schema_offers_multi_model_handoff():
    """Multi-model users must see the same opt-in handoff prompt as other layouts."""
    root = Path(__file__).resolve().parents[3] / "templates/databricks"
    schema = json.loads((root / "databricks_template_schema.json").read_text())
    assert "skip_prompt_if" not in schema["properties"]["score_handoff"]


def test_initialization_saves_parent_handoff_separately(workflow_config, tracked):
    """A frozen parent policy must survive independently of disabled component handoff."""
    from skyulf.integrations.databricks._lifecycle_state import LifecycleContext, PhaseStore
    from skyulf.integrations.databricks.branch_tasks import initialize_branch_training

    uri, _ = tracked
    context = LifecycleContext("20", "30")
    initialized = initialize_branch_training(
        None,
        configs={"model": {**workflow_config, "score_handoff": "disabled"}},
        settings={"model_name": "workspace.models.set"},
        composition_source="",
        context=context,
        tracking_uri=uri,
        experiment_name="handoff",
        action="approve",
        score_handoff="after_alias_change",
    )
    store = PhaseStore(uri, context)
    store.bind(initialized.reference)
    assert store.request["score_handoff"] == "after_alias_change"
    assert store.request["config"]["score_handoff"] == "disabled"
