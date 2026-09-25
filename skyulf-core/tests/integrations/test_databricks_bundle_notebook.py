"""A generated notebook must delegate behavior to the installed Core library."""

import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "action", ["train", "train_monthly", "score", "score_from_approve", "score_from_rollback"]
)
def test_notebook_delegates_bound_target_and_selected_action(tmp_path, monkeypatch, engine, action):
    """All widget actions must use the public library while preserving engine and target binding."""
    from skyulf.integrations.databricks import job_runtime

    inherited_action = (
        action.removeprefix("score_from_") if action.startswith("score_from_") else None
    )
    action = "score" if inherited_action else action
    path = (
        Path(__file__).resolve().parents[2]
        / "templates/databricks/template/{{.project_name}}/src"
        / ("score.py" if action == "score" else "workflow.py")
    )
    config = {
        "engine": engine,
        "score_model_selection": "pinned_version",
        "promotion_policy": "manual_approval",
        "score_handoff": "disabled",
        **{
            key: "{catalog}.{input_schema}." + key
            for key in ("training_table", "score_source_table", "prediction_table", "model_name")
        },
    }
    config_path = tmp_path / "workflow.json"
    config_path.write_text(json.dumps(config))
    values = {
        "config_path": str(config_path),
        "experiment_name": "/test/experiment",
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
    }
    if action != "score":
        values["lifecycle_action"] = action
    if inherited_action:
        # Run Job receives parent job parameters automatically, including evidence.
        values.update(
            lifecycle_action=inherited_action,
            candidate_version="2",
            comparison_sha256="b" * 64,
            expected_champion_version="1",
            rejection_reason="",
            promotion_receipt_json='{"kind":"promotion"}',
        )

    @dataclass
    class Result:
        """Represent the real service's serializable result boundary."""

        ok: bool = True

    run = Mock(return_value=Result())
    notebook = Mock()
    spark = object()
    monkeypatch.setattr(job_runtime, "run_action", run)
    task_values = Mock()
    runpy.run_path(
        str(path),
        run_name="__main__",
        init_globals={
            "spark": spark,
            "dbutils": SimpleNamespace(
                widgets=SimpleNamespace(getAll=lambda: values),
                notebook=notebook,
                jobs=SimpleNamespace(taskValues=task_values),
            ),
        },
    )
    assert run.call_args.args == (
        spark,
        {
            **config,
            **{
                key: "workspace.test." + key
                for key in (
                    "training_table",
                    "score_source_table",
                    "prediction_table",
                    "model_name",
                )
            },
        },
        action,
    )
    assert run.call_args.kwargs.get("experiment_name") == (
        "/test/experiment" if action.startswith("train") else None
    )
    assert (run.call_args.kwargs.get("artifact_path") is not None) == action.startswith("train")
    output = json.loads(notebook.exit.call_args.args[0])
    assert output["result"] == {"ok": True}
    assert output["score_requested"] is False
    if action == "score":
        task_values.set.assert_not_called()
    else:
        task_values.set.assert_called_once_with(key="score_requested", value=False)


@pytest.mark.parametrize("override", ["task_role", "action"])
def test_score_notebook_retains_override_guards_when_removing_parent_evidence(
    tmp_path, monkeypatch, override
):
    """Filtering inherited approval inputs must not hide attempts to override the fixed role."""
    from skyulf.integrations.databricks import job_runtime

    config_path = tmp_path / "workflow.json"
    config_path.write_text(
        json.dumps(
            {
                "score_model_selection": "champion",
                "promotion_policy": "manual_approval",
                "score_handoff": "after_alias_change",
                **{
                    key: "workspace.test." + key
                    for key in (
                        "training_table",
                        "score_source_table",
                        "prediction_table",
                        "model_name",
                    )
                },
            }
        )
    )
    values = {
        "config_path": str(config_path),
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
        "lifecycle_action": "approve",
        "candidate_version": "2",
        override: "approve",
    }
    run = Mock()
    monkeypatch.setattr(job_runtime, "run_action", run)
    with pytest.raises(ValueError, match="cannot be overridden"):
        job_runtime.run_notebook(
            None, SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values)), task_role="score"
        )
    run.assert_not_called()
