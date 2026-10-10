"""Generated named tasks are real training tasks with independent SHAP leaves."""

import json
import runpy

import pytest
import yaml
from test_databricks_bundle_generation import (  # noqa: F401
    _generate_project,
    _read_jobs,
    pytestmark,
)


@pytest.mark.parametrize("layout", ["model_competition", "multi_target"])
def test_named_training_and_shap_graph(tmp_path, layout):
    """SHAP must never become a dependency of model selection or complete-set assembly."""
    project = _generate_project(tmp_path, training_layout=layout, shap_enabled="true")
    tasks = {task["task_key"]: task for task in _read_jobs(project)["train"]["tasks"]}
    training = {key: task for key, task in tasks.items() if key.startswith("train_")}
    expected_count = 3 if layout == "model_competition" else 2
    assert len(training) == expected_count
    for key, task in training.items():
        assert task["notebook_task"]["notebook_path"] == "../src/jobs/train_model.py"
        shap = tasks[key.replace("train_", "shap_", 1)]
        assert shap["depends_on"] == [{"task_key": key}]
        assert shap["notebook_task"]["base_parameters"]["reference_json"] == (
            "{{tasks." + key + ".values.reference_json}}"
        )
    assert not any(
        dep["task_key"].startswith("shap_")
        for task in tasks.values()
        for dep in task.get("depends_on", [])
    )
    join = tasks["select_best_model" if layout == "model_competition" else "register_model_set"]
    assert {
        dep["task_key"] for dep in join["depends_on"] if dep["task_key"].startswith("train_")
    } == set(training)


@pytest.mark.parametrize("layout", ["model_competition", "multi_target"])
def test_refresh_graph_after_model_rename(tmp_path, layout):
    """Renamed YAML declarations must update tasks, references and the early mismatch guard."""
    project = _generate_project(tmp_path, training_layout=layout, shap_enabled="true")
    path = project / "config/training.yml"
    document = yaml.safe_load(path.read_text())
    names = list(document["models"])
    document["models"]["renamed_model"] = document["models"].pop(names[0])
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    refresh = runpy.run_path(str(project / "src/tools/refresh_training_graph.py"))["refresh"]
    refreshed = refresh(project)
    tasks = {task["task_key"]: task for task in _read_jobs(project)["train"]["tasks"]}
    assert "train_renamed_model" in tasks and "shap_renamed_model" in tasks
    assert f"train_{names[0]}" not in tasks
    assert f"shap_{names[0]}" not in tasks
    params = tasks["initialize_run"]["notebook_task"]["base_parameters"]
    assert json.loads(params["model_keys_json"]) == refreshed
    before = (project / "resources/train.job.yml").read_text()
    refresh(project)
    assert (project / "resources/train.job.yml").read_text() == before
