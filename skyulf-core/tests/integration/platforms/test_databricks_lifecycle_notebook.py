"""Fixed lifecycle notebooks validate task metadata and use durable references."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "change",
    [
        {"repair_count": "1"},
        {"execution_count": "2"},
        {"job_run_id": "{{job.run_id}}"},
        {"workflow_contract": "1"},
        {"phase": "operator"},
        {"task_role": "score"},
    ],
)
def test_phase_notebook_rejects_unsafe_context_before_loading_config(change):
    """Repairs and role overrides cannot get as far as model or registry side effects."""
    from skyulf.integrations.databricks.jobs.shared import job_runtime

    execute = getattr(job_runtime, "run_lifecycle_notebook", None)
    assert callable(execute), "Fixed lifecycle notebook adapter is missing"
    values = {
        "job_id": "10",
        "job_run_id": "20",
        "repair_count": "0",
        "execution_count": "1",
        "workflow_contract": "2",
        "config_path": "must-not-open.json",
        **change,
    }
    dbutils = SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values))
    with pytest.raises(ValueError):
        execute(None, dbutils, phase="prepare")


def test_downstream_notebook_uses_saved_reference_without_editable_config(monkeypatch):
    """Later tasks must use the prepared invocation even if editable files disappear."""
    from skyulf.integrations.databricks.jobs.lifecycle import lifecycle_tasks
    from skyulf.integrations.databricks.jobs.shared import job_runtime

    reference = {
        "run_id": "abc",
        "request_sha256": "a" * 64,
        "phase": "prepare",
        "receipt_sha256": "b" * 64,
    }
    values = {
        "job_id": "10",
        "job_run_id": "20",
        "repair_count": "0",
        "execution_count": "1",
        "workflow_contract": "2",
        "reference_json": json.dumps(reference),
        "tracking_uri": "databricks",
        "config_path": "must-not-open.json",
        "lifecycle_action": "train",
    }
    result = SimpleNamespace(reference=reference, output={"training_rows": 16})
    execute = Mock(return_value=result)
    monkeypatch.setattr(lifecycle_tasks, "run_lifecycle_phase", execute)
    task_values = Mock()
    display = Mock(side_effect=RuntimeError("display unavailable"))
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(getAll=lambda: values),
        jobs=SimpleNamespace(taskValues=task_values),
        notebook=Mock(),
    )
    output = job_runtime.run_lifecycle_notebook(
        None,
        dbutils,
        phase="train",
        display_html=display,
        exit_notebook=False,
    )
    assert execute.call_args.kwargs["reference"] == reference
    assert "config" not in execute.call_args.kwargs
    assert json.loads(output)["training_rows"] == 16
    dbutils.notebook.exit.assert_not_called()
    execute.assert_called_once()


@pytest.mark.parametrize("phase", ["compare_decide", "complete"])
def test_first_candidate_notebook_renders_null_champion_metrics(monkeypatch, phase, caplog):
    """First-model reports must show quality and operator actions instead of JSON fallback."""
    from skyulf.integrations.databricks.jobs.lifecycle import lifecycle_tasks
    from skyulf.integrations.databricks.jobs.shared import job_runtime

    candidate = {
        "model_name": "workspace.test.model",
        "model_version": "1",
        "comparison": {
            "metric": "heldout_rmse",
            "eligible": False,
            "reason": "no_champion",
            "candidate_metrics": {"heldout_rmse": 2.7785},
            "champion_metrics": None,
        },
    }
    payload = (
        {
            "action": "train",
            "result": candidate,
            "score_requested": False,
            "next_actions": {
                action: {
                    "lifecycle_action": action,
                    "candidate_version": "1",
                    "expected_champion_version": "none",
                }
                for action in ("approve", "reject")
            },
        }
        if phase == "complete"
        else {"candidate": candidate, "alias_change": None, "promotion_policy": "manual_approval"}
    )
    monkeypatch.setattr(
        lifecycle_tasks,
        "run_lifecycle_phase",
        Mock(return_value=SimpleNamespace(reference={"phase": phase}, output=payload)),
    )
    values = {
        "job_id": "10",
        "job_run_id": "20",
        "repair_count": "0",
        "execution_count": "1",
        "workflow_contract": "2",
        "reference_json": '{"phase": "prepare"}',
        "tracking_uri": "databricks",
        "training_result_state": "success",
        "operator_result_state": "excluded",
    }
    reports = []
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(getAll=lambda: values),
        jobs=SimpleNamespace(taskValues=Mock()),
    )
    output = job_runtime.run_lifecycle_notebook(
        None, dbutils, phase=phase, display_html=reports.append, exit_notebook=False
    )
    assert json.loads(output) == payload
    assert len(reports) == 1
    visible = reports[0].split("<details>")[0]
    assert "heldout_rmse" in visible and "2.7785" in visible and "No champion" in visible
    if phase == "complete":
        assert "Available action: approve" in visible and "Available action: reject" in visible
        assert "If rollback is needed" not in visible
    else:
        assert "Awaiting manual review" in visible and "finalize_and_report" in visible
    assert "Readable output unavailable" not in caplog.text


@pytest.mark.parametrize("fails", [False, True])
def test_completion_publishes_score_request_only_after_verified_result(monkeypatch, fails):
    """The ALL_DONE join must never emit a score request when finalization fails."""
    from skyulf.integrations.databricks.jobs.lifecycle import lifecycle_tasks
    from skyulf.integrations.databricks.jobs.shared import job_runtime

    payload = {"action": "train", "result": {}, "score_requested": True}
    execute = Mock(return_value=SimpleNamespace(reference={"phase": "result"}, output=payload))
    if fails:
        execute.side_effect = ValueError("Lifecycle has no successful finalized result.")
    monkeypatch.setattr(lifecycle_tasks, "run_lifecycle_phase", execute)
    values = {
        "job_id": "10",
        "job_run_id": "20",
        "repair_count": "0",
        "execution_count": "1",
        "workflow_contract": "2",
        "reference_json": '{"phase": "prepare"}',
        "tracking_uri": "databricks",
        "training_result_state": "failed" if fails else "success",
        "operator_result_state": "excluded",
    }
    task_values = Mock()
    display = Mock()
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(getAll=lambda: values),
        jobs=SimpleNamespace(taskValues=task_values),
    )
    if fails:
        with pytest.raises(ValueError, match="successful finalized"):
            job_runtime.run_lifecycle_notebook(None, dbutils, phase="complete")
        task_values.set.assert_not_called()
    else:
        result = job_runtime.run_lifecycle_notebook(
            None, dbutils, phase="complete", display_html=display, exit_notebook=False
        )
        assert json.loads(result) == payload
        task_values.set.assert_any_call(key="score_requested", value=True)
        assert "Scoring requested" in display.call_args.args[0]
    assert execute.call_args.kwargs["task_states"] == {
        "training": "failed" if fails else "success",
        "operator": "excluded",
    }


@pytest.mark.parametrize(
    "filename,phase",
    [
        ("initialize_run", "initialize"),
        ("load_data", "load_data"),
        ("prepare_dataset", "prepare_dataset"),
        ("train_and_tune", "train"),
        ("select_best_model", "select_best_model"),
        ("register_model", "evaluate_register"),
        ("evaluate_model", "compare"),
        ("model_decision", "model_decision"),
        ("training_report", "complete"),
    ],
)
def test_generated_notebooks_bind_their_own_phase_and_defer_exit(monkeypatch, filename, phase):
    """Normal job nodes preserve their phase and JSON result without requesting HTML."""
    from skyulf.integrations.databricks.jobs.shared import job_runtime

    path = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/jobs"
        / f"{filename}.py"
    )
    execute = Mock(return_value='{"ok": true}')
    monkeypatch.setattr(job_runtime, "run_lifecycle_notebook", execute)
    notebook = Mock()
    dbutils = SimpleNamespace(notebook=notebook)
    display = Mock()
    runpy.run_path(
        str(path),
        run_name="__main__",
        init_globals={"spark": None, "dbutils": dbutils, "displayHTML": display},
    )
    assert execute.call_args.kwargs.get("display_html") is None
    assert execute.call_args.kwargs["phase"] == phase
    assert execute.call_args.kwargs["exit_notebook"] is False
    assert ("preprocessing_path" in execute.call_args.kwargs) == (phase == "initialize")
    assert "# COMMAND ----------" in path.read_text()
    notebook.exit.assert_called_once_with('{"ok": true}')


def test_phase_output_escapes_values_and_folds_technical_digests():
    """Useful counts stay visible while long proof fields remain in technical details."""
    from skyulf.integrations.databricks.jobs.shared.job_output import render_lifecycle_output

    output = render_lifecycle_output(
        "train",
        {"training_rows": 16, "model_digest": "a" * 64, "source_table": "<script>bad()</script>"},
    )
    visible = output.split("<details>")[0]
    assert "16" in visible
    assert "a" * 64 not in visible
    assert "<script>" not in output
    assert "&lt;script&gt;" in output


@pytest.mark.parametrize("phase", ["compare", "decide", "compare_decide"])
def test_phase_report_shows_comparison_and_actual_promotion_decision(phase):
    """Reviewers must see metrics and the alias decision without opening technical JSON."""
    from skyulf.integrations.databricks.jobs.shared.job_output import render_lifecycle_output

    candidate = {
        "model_version": "3",
        "comparison": {
            "metric": "heldout_rmse",
            "eligible": True,
            "reason": "candidate_improved",
            "candidate_metrics": {"heldout_rmse": 1.5},
            "champion_metrics": {"heldout_rmse": 2.0},
        },
    }
    payload = (
        candidate
        if phase == "compare"
        else {
            "candidate": candidate,
            "alias_change": {"kind": "promotion", "prior_version": "2", "new_version": "3"},
            "promotion_policy": "automatic",
        }
    )
    visible = render_lifecycle_output(phase, payload).split("<details>")[0]
    assert "heldout_rmse" in visible and "1.5" in visible and "2.0" in visible
    assert "candidate_improved" in visible
    assert "Eligible" in visible
    if phase in {"decide", "compare_decide"}:
        assert "Champion changed" in visible and "v2" in visible and "v3" in visible


@pytest.mark.parametrize("phase", ["decide", "compare_decide"])
def test_manual_review_report_does_not_imply_promotion(phase):
    """A candidate awaiting human approval must not look like a successful champion change."""
    from skyulf.integrations.databricks.jobs.shared.job_output import render_lifecycle_output

    visible = render_lifecycle_output(
        phase,
        {
            "candidate": {"model_version": "3"},
            "alias_change": None,
            "promotion_policy": "manual_approval",
        },
    ).split("<details>")[0]
    assert "Awaiting manual review" in visible
    assert "v3" in visible
    assert "Champion changed" not in visible


@pytest.mark.parametrize("action", ["train", "rollback"])
def test_prepare_freezes_project_code_only_for_training(
    tmp_path, monkeypatch, workflow_config, action
):
    """Operator actions must remain usable after a project's editable Python changes."""
    from skyulf.integrations.databricks.jobs.lifecycle import lifecycle_tasks
    from skyulf.integrations.databricks.jobs.shared import job_runtime

    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(workflow_config), encoding="utf-8")
    source = "def build_preprocessing():\n    return []\n"
    (tmp_path / "preprocessing.py").write_text(
        source if action == "train" else "raise RuntimeError('must not import')",
        encoding="utf-8",
        newline="\n",
    )
    values = {
        "job_id": "10",
        "job_run_id": "20",
        "repair_count": "0",
        "execution_count": "1",
        "workflow_contract": "2",
        "deployed_score_handoff": workflow_config["score_handoff"],
        "config_path": str(path),
        "lifecycle_action": action,
        "experiment_name": "/test/staged",
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
    }
    if action == "rollback":
        values.update(
            expected_champion_version="2",
            promotion_receipt_json=json.dumps(
                {
                    "event_id": "event",
                    "kind": "promotion",
                    "model_name": workflow_config["model_name"],
                    "alias": "champion",
                    "prior_version": "1",
                    "new_version": "2",
                    "comparison_sha256": "a" * 64,
                    "parent_event_id": None,
                }
            ),
        )
    execute = Mock(
        return_value=SimpleNamespace(
            reference={"run_id": "abc"},
            output={"action": action, "training_requested": action == "train"},
        )
    )
    monkeypatch.setattr(lifecycle_tasks, "run_lifecycle_phase", execute)
    task_values = Mock()
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(getAll=lambda: values), jobs=SimpleNamespace(taskValues=task_values)
    )
    job_runtime.run_lifecycle_notebook(
        None, dbutils, phase="prepare", preprocessing_path="preprocessing.py", exit_notebook=False
    )
    prepared = execute.call_args.kwargs["config"]
    assert ("project_python_source" in prepared["pipeline"]) == (action == "train")
    if action == "train":
        assert prepared["pipeline"]["project_python_source"] == source
    else:
        assert (
            execute.call_args.kwargs["operator_options"]["promotion_receipt"].prior_version == "1"
        )
    assert execute.call_args.kwargs["action"] == action
    assert any(
        call.kwargs == {"key": "training_requested", "value": action == "train"}
        for call in task_values.set.call_args_list
    )


@pytest.mark.parametrize(
    "phase,contract",
    [("initialize", "2"), ("model_decision", "2"), ("prepare", "3"), ("train_register", "3")],
)
def test_notebook_rejects_mixed_graph_contract_before_execution(monkeypatch, phase, contract):
    """Partially regenerated notebooks must not execute with another graph's contract."""
    from skyulf.integrations.databricks.jobs.lifecycle import lifecycle_tasks
    from skyulf.integrations.databricks.jobs.shared import job_runtime

    execute = Mock()
    monkeypatch.setattr(lifecycle_tasks, "run_lifecycle_phase", execute)
    values = {
        "job_id": "10",
        "job_run_id": "20",
        "repair_count": "0",
        "execution_count": "1",
        "workflow_contract": contract,
    }
    dbutils = SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values))
    with pytest.raises(ValueError, match="contract"):
        job_runtime.run_lifecycle_notebook(None, dbutils, phase=phase)
    execute.assert_not_called()
