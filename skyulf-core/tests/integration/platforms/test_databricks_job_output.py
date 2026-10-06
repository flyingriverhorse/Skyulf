"""Operator output must explain results without exposing technical copy requirements."""

import json

import pytest

from skyulf.integrations.databricks.jobs.shared import job_runtime


@pytest.mark.parametrize("noop", [False, True])
@pytest.mark.parametrize("model_set", [False, True])
def test_scoring_summary_shows_current_source_without_reusing_old_write(noop, model_set):
    """A no-op must show the current source watermark and keep old provenance clearly labeled."""
    from skyulf.integrations.databricks.jobs.shared.job_output import render_bundle_output
    from skyulf.integrations.databricks.model_sets.model_set_project import render_model_set_result

    result = {
        "source_end_version": 12,
        "input_count": 0 if noop else 4,
        "output_count": 0 if noop else 4,
        "commit_version": 8,
        "noop": noop,
        "manifest": {"model_name": "old_model", "model_version": "1", "source_end_version": 9},
    }
    context = {
        "source_table": "catalog.input.<source>",
        "prediction_table": "catalog.output.predictions",
    }
    if model_set:
        html = render_model_set_result(
            {**context, **result, "model_set_name": "catalog.models.set", "model_set_version": "3"}
        )
    else:
        html = render_bundle_output(
            {
                **context,
                "action": "score",
                "result": {
                    **result,
                    "selected_model_name": "catalog.models.one",
                    "selected_model_version": "3",
                },
            }
        )
    visible = html.split("<details>", 1)[0]
    assert "Source table" in visible and "catalog.input.&lt;source&gt;" in visible
    assert "Prediction table" in visible and "catalog.output.predictions" in visible
    assert "Source end version</td><td>12" in visible
    assert "Input rows" in visible and "Output rows" in visible
    assert "v3" in visible
    assert ("No new predictions written" if noop else "Prediction write completed") in visible


def test_comparison_output_explains_each_quality_gate():
    """Operators must see every failed bound even when the selected metric passes."""
    from skyulf.integrations.databricks.jobs.shared.job_output import render_lifecycle_output

    html = render_lifecycle_output(
        "compare_decide",
        {
            "candidate": {
                "comparison": {
                    "metric": "heldout_rmse",
                    "eligible": False,
                    "reason": "quality_gate_failed",
                    "min_improvement": 0.5,
                    "quality_threshold": 3,
                    "quality_gates": {"heldout_mae": 1, "heldout_r2": 0.9},
                    "candidate_metrics": {"heldout_rmse": 2, "heldout_mae": 2},
                    "champion_metrics": {"heldout_rmse": 3},
                }
            }
        },
    )
    visible = html.split("<details>", 1)[0]
    assert "Failed: threshold not met" in visible and "metric unavailable or non-finite" in visible
    assert "heldout_r2" in visible and "0.9" in visible
    assert "Minimum improvement (absolute)" in visible and "0.5" in visible


def test_promotion_summary_separates_handoff_request_from_score_completion():
    """A successful alias change must not claim that prediction already succeeded."""
    from skyulf.integrations.databricks.jobs.shared.job_output import render_bundle_output

    payload = {
        "action": "approve",
        "result": {
            "kind": "promotion",
            "model_name": "test.model",
            "prior_version": "1",
            "new_version": "5",
        },
        "score_requested": True,
        "next_actions": {
            "rollback": {
                "lifecycle_action": "rollback",
                "expected_champion_version": "5",
                "promotion_receipt_json": '{"event_id":"saved","comparison_sha256":"proof"}',
            }
        },
    }
    html = render_bundle_output(payload)
    assert "Champion changed" in html and "v1" in html and "v5" in html
    assert "requested" in html and "child score run" in html
    assert "promotion_receipt_json" in html and "Technical details" in html
    visible_summary = html.split("<details>", 1)[0]
    assert "If rollback is needed" in visible_summary
    assert "Required current champion</td><td>v5" in visible_summary
    assert "Restore version</td><td>v1" in visible_summary
    assert "does not run automatically" in visible_summary
    assert "comparison_sha256" not in visible_summary
    assert "promotion_receipt_json" not in visible_summary
    assert "Next action: rollback" not in html


def test_output_escapes_model_names_and_explains_noop_manifest():
    """Registry text cannot inject HTML and a no-op must not mislabel old write provenance."""
    from skyulf.integrations.databricks.jobs.shared.job_output import render_bundle_output

    html = render_bundle_output(
        {
            "action": "score",
            "score_requested": False,
            "next_actions": {},
            "result": {
                "noop": True,
                "selected_model_name": "workspace.test.model",
                "selected_model_version": "5",
                "input_count": 0,
                "output_count": 0,
                "manifest": {"model_name": "<script>alert(1)</script>", "model_version": "1"},
            },
        }
    )
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "No new predictions written" in html
    assert "previous write" in html
    assert "Selected model for this run" in html and "workspace.test.model v5" in html


def test_manual_training_output_shows_metrics_and_simple_action_fields():
    """Operators can review quality and see usable fields without finding a digest."""
    from skyulf.integrations.databricks.jobs.shared.job_output import render_bundle_output

    html = render_bundle_output(
        {
            "action": "train",
            "score_requested": False,
            "result": {
                "model_version": "2",
                "comparison": {
                    "metric": "heldout_rmse",
                    "eligible": True,
                    "reason": "candidate_improved",
                    "candidate_metrics": {"heldout_rmse": 1.5},
                    "champion_metrics": {"heldout_rmse": 3.0},
                },
            },
            "next_actions": {
                "approve": {
                    "lifecycle_action": "approve",
                    "candidate_version": "2",
                    "expected_champion_version": "1",
                }
            },
        }
    )
    assert "heldout_rmse" in html and "1.5" in html and "3.0" in html
    assert "candidate_version" in html and "expected_champion_version" in html
    assert "comparison_sha256" not in html


@pytest.mark.parametrize("display_fails", [False, True])
@pytest.mark.parametrize("entrypoint", ["run_notebook", "run_score_notebook"])
def test_notebook_renders_summary_and_preserves_machine_result(
    tmp_path, monkeypatch, workflow_config, capsys, display_fails, entrypoint
):
    """Readable notebook output must preserve the API exit JSON contract."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(workflow_config))
    values = {
        "config_path": str(path),
        "workflow_contract": "2",
        "deployed_score_handoff": "after_alias_change",
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
    }
    monkeypatch.setattr(job_runtime, "resolve_target_config", lambda config, bindings: config)
    outcome = job_runtime.BundleActionResult("score", {"noop": True}, False, {})
    monkeypatch.setattr(job_runtime, "run_bundle_action", lambda *args, **kwargs: outcome)
    notebook = Mock()
    display = Mock(side_effect=RuntimeError("display unavailable") if display_fails else None)
    warning = Mock()
    # Other suites install stdout log handlers; logging layout is not the exit JSON contract.
    monkeypatch.setattr(job_runtime.logging.getLogger(job_runtime.__name__), "warning", warning)
    getattr(job_runtime, entrypoint)(
        None,
        SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values), notebook=notebook),
        **({"task_role": "score"} if entrypoint == "run_notebook" else {}),
        display_html=display,
    )
    assert "No new predictions written" in display.call_args.args[0]
    if display_fails:
        warning.assert_called_once_with("Readable output unavailable; see JSON result.")
        assert "Noop: True" in capsys.readouterr().out
    else:
        warning.assert_not_called()
    assert json.loads(notebook.exit.call_args.args[0])["result"] == {"noop": True}
    if entrypoint == "run_score_notebook":
        payload = json.loads(notebook.exit.call_args.args[0])
        assert payload["source_table"] == workflow_config["score_source_table"]
        assert payload["prediction_table"] == workflow_config["prediction_table"]


@pytest.mark.parametrize("entrypoint", ["training_report.py", "score.py"])
def test_generated_notebook_keeps_report_and_exit_in_separate_cells(entrypoint):
    """Databricks replaces same-cell output on exit, so the report needs its own cell."""
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/jobs"
        / entrypoint
    )
    cells = path.read_text().split("# COMMAND ----------")
    assert len(cells) == 2
    expected_call = (
        "run_lifecycle_notebook(" if entrypoint == "training_report.py" else "run_score_notebook("
    )
    assert "exit_notebook=False" in cells[0] and expected_call in cells[0]
    assert ".notebook.exit(output)" in cells[1]


@pytest.mark.parametrize("entrypoint", ["score.py", "score_models.py"])
def test_generated_score_notebook_enables_visible_summary(entrypoint):
    """The notebook must pass Databricks' renderer to show HTML instead of only JSON."""
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/jobs"
        / entrypoint
    )
    source = path.read_text()
    assert 'display_html=globals().get("displayHTML")' in source


def test_legacy_notebook_converts_result_before_publishing_score_request(monkeypatch):
    """A failed result conversion must not publish a success/handoff task value."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from skyulf.integrations.databricks.jobs.shared import job_runtime

    class Uncopyable:
        """Represent a result that fails dataclass conversion before JSON rendering."""

        def __deepcopy__(self, memo):
            """Expose the ordering boundary without executing lifecycle mutations."""
            raise RuntimeError("cannot copy result")

    outcome = job_runtime.BundleActionResult("train", {"value": Uncopyable()}, True, {})
    monkeypatch.setattr(job_runtime, "read_notebook_config", lambda values: {})
    monkeypatch.setattr(job_runtime, "run_bundle_action", lambda *args, **kwargs: outcome)
    task_values = Mock()
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(getAll=lambda: {"lifecycle_action": "train"}),
        jobs=SimpleNamespace(taskValues=task_values),
    )
    with pytest.raises(RuntimeError, match="cannot copy result"):
        job_runtime.run_notebook(None, dbutils, task_role="lifecycle")
    task_values.set.assert_not_called()


def test_nested_search_output_explains_independent_outer_scores():
    """Notebook summaries must distinguish outer evaluation from final search scores."""
    from skyulf.integrations.databricks.jobs.shared.job_output import render_lifecycle_output

    report = {
        "status": "nested_cv",
        "outer_folds": 2,
        "inner_folds": 3,
        "mean_score": -0.25,
        "std_score": 0.05,
        "scoring_metric": "neg_mean_squared_error",
        "total_trials": 6,
        "folds": [
            {"fold": 1, "inner_best_score": -0.1, "outer_score": -0.2, "best_params": {"alpha": 2}}
        ],
    }
    html = render_lifecycle_output("train", {"tuning": {"nested_cv": report, "best_score": -0.1}})
    assert "Nested CV evaluation" in html and "Outer mean score" in html
    assert "Inner folds" in html and "neg_mean_squared_error" in html
    assert "separate final search" in html


def test_scoring_coverage_is_visible_only_for_current_write():
    """A no-op must not present prior prediction/exclusion counts as new work."""
    from skyulf.integrations.databricks.jobs.shared.job_output import render_bundle_output

    result = {
        "noop": False,
        "input_count": 3,
        "output_count": 3,
        "manifest": {
            "model_name": "workspace.test.model",
            "model_version": "1",
            "predicted_count": 2,
            "excluded_count": 1,
        },
    }
    html = render_bundle_output({"action": "score", "result": result})
    assert "Predicted rows" in html and "Excluded rows" in html
    noop_html = render_bundle_output({"action": "score", "result": result | {"noop": True}})
    assert "Predicted rows" not in noop_html and "Excluded rows" not in noop_html
