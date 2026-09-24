"""A generated notebook must delegate behavior to the installed Core library."""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("action", ["train", "train_monthly", "score"])
def test_notebook_delegates_bound_target_and_selected_action(tmp_path, monkeypatch, engine, action):
    """All widget actions must use the public library while preserving engine and target binding."""
    from skyulf.integrations.databricks import local_workflow

    path = (
        Path(__file__).resolve().parents[2]
        / "templates/databricks/template/{{.project_name}}/src/workflow.py"
    )
    spec = importlib.util.spec_from_file_location("thin_notebook", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.run_action is local_workflow.run_action
    assert module.resolve_target_config is local_workflow.resolve_target_config
    config = {
        "engine": engine,
        **{
            key: "{catalog}.{input_schema}." + key
            for key in ("training_table", "score_source_table", "prediction_table", "model_name")
        },
    }
    config_path = tmp_path / "workflow.json"
    config_path.write_text(json.dumps(config))
    values = {
        "config_path": str(config_path),
        "action": action,
        "experiment_name": "/test/experiment",
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
    }

    @dataclass
    class Result:
        """Represent the real service's serializable result boundary."""

        ok: bool = True

    run = Mock(return_value=Result())
    notebook = Mock()
    spark = object()
    monkeypatch.setattr(module, "run_action", run)
    monkeypatch.setattr(module, "spark", spark, raising=False)
    monkeypatch.setattr(
        module,
        "dbutils",
        SimpleNamespace(widgets=SimpleNamespace(get=values.__getitem__), notebook=notebook),
        raising=False,
    )
    module.main()
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
    assert run.call_args.kwargs["experiment_name"] == (
        "/test/experiment" if action.startswith("train") else None
    )
    assert (run.call_args.kwargs["artifact_path"] is not None) == action.startswith("train")
    assert json.loads(notebook.exit.call_args.args[0]) == {"ok": True}
