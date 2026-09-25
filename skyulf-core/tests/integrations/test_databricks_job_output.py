"""Operator output must explain results without exposing technical copy requirements."""

import json

import pytest

from skyulf.integrations.databricks import job_runtime


def test_promotion_summary_separates_handoff_request_from_score_completion():
    """A successful alias change must not claim that prediction already succeeded."""
    from skyulf.integrations.databricks.job_output import render_bundle_output

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
    from skyulf.integrations.databricks.job_output import render_bundle_output

    html = render_bundle_output(
        {
            "action": "score",
            "score_requested": False,
            "next_actions": {},
            "result": {
                "noop": True,
                "input_count": 0,
                "output_count": 0,
                "manifest": {"model_name": "<script>alert(1)</script>", "model_version": "1"},
            },
        }
    )
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "No new predictions written" in html
    assert "previous write" in html


def test_manual_training_output_shows_metrics_and_simple_action_fields():
    """Operators can review quality and see usable fields without finding a digest."""
    from skyulf.integrations.databricks.job_output import render_bundle_output

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
def test_notebook_renders_summary_and_preserves_machine_result(
    tmp_path, monkeypatch, capsys, display_fails
):
    """Readable notebook output must preserve the API exit JSON contract."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    path = tmp_path / "workflow.json"
    path.write_text("{}")
    values = {
        "config_path": str(path),
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
    job_runtime.run_notebook(
        None,
        SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values), notebook=notebook),
        task_role="score",
        display_html=display,
    )
    assert "No new predictions written" in display.call_args.args[0]
    if display_fails:
        assert json.loads(capsys.readouterr().out)["result"] == {"noop": True}
    assert json.loads(notebook.exit.call_args.args[0])["result"] == {"noop": True}


@pytest.mark.parametrize("entrypoint", ["workflow.py", "score.py"])
def test_generated_notebook_keeps_report_and_exit_in_separate_cells(entrypoint):
    """Databricks replaces same-cell output on exit, so the report needs its own cell."""
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "templates/databricks/template/{{.project_name}}/src"
        / entrypoint
    )
    cells = path.read_text().split("# COMMAND ----------")
    assert len(cells) == 2
    assert "exit_notebook=False" in cells[0] and "run_notebook(" in cells[0]
    assert ".notebook.exit(output)" in cells[1]
