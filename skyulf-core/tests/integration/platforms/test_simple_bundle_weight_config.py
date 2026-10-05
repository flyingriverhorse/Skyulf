"""Trusted weight settings are captured once and reserve every declared source role."""

import hashlib
import json
from copy import deepcopy

import pytest

from skyulf.integrations.databricks.jobs.training.branch_notebook import (
    load_training_branch_configs,
)
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config


def _project(tmp_path, workflow_config, layout, source=None, branch_overlays=None):
    """Build real editable projects with two independent branch or candidate models."""
    features = tmp_path / "src/features"
    features.mkdir(parents=True)
    features.joinpath("__init__.py").write_text(
        "def build_preprocessing(recipe='default'):\n    return []\n", encoding="utf-8"
    )
    modeling = tmp_path / "src/modeling"
    modeling.mkdir()
    model = {"type": "logistic_regression", "params": {"class_weight": "balanced"}}
    config = deepcopy(workflow_config)
    config.update(
        training_layout=layout,
        task="classification",
        metric="heldout_accuracy",
        score_handoff="disabled",
        pipeline={"preprocessing": [], "modeling": model},
    )
    modeling.joinpath("single_model.py").write_text(
        f"def build_modeling():\n    return {model!r}\n", encoding="utf-8"
    )
    if layout == "single_model":
        config["pipeline"]["modeling"] = {}
    if layout == "model_competition":
        config.update(cv_enabled=True, cv_folds=3)
        candidates = {name: {"modeling": model} for name in ("risk", "revenue")}
        modeling.joinpath("model_competition.py").write_text(
            f"def build_candidates(task):\n    return {candidates!r}\n", encoding="utf-8"
        )
    # A custom feature directory must not change the model declaration location.
    custom = tmp_path / "src/custom/risk"
    custom.mkdir(parents=True)
    custom.joinpath("__init__.py").write_text(features.joinpath("__init__.py").read_text())
    entries = {
        name: {"workflow": overlay}
        for name, overlay in (branch_overlays or {"risk": {}, "revenue": {}}).items()
    }
    entries["risk"]["features_path"] = "../custom/risk"
    modeling.joinpath("multi_model.py").write_text(
        f"def build_training_branches():\n    return {entries!r}\n", encoding="utf-8"
    )
    if source is not None:
        filename = {
            "single_model": "single_model.py",
            "model_competition": "model_competition.py",
            "multi_target": "multi_model.py",
        }[layout]
        hook = modeling / filename
        hook.write_bytes((source + "\n" + hook.read_text()).encode("utf-8"))
    path = tmp_path / "config/workflow.json"
    path.parent.mkdir()
    path.write_text(json.dumps(config), encoding="utf-8")
    values = {
        "config_path": str(path),
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
        "workflow_contract": "3",
        "deployed_score_handoff": "disabled",
    }
    return config, features, modeling, values


def _load(project):
    """Use each production layout's public loader without remote services."""
    config, features, _, values = project
    if config["training_layout"] == "multi_target":
        return load_training_branch_configs(values)
    return {"single": load_project_workflow(config, features)}


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_model_file_owns_weight_column_without_separate_hook(tmp_path, workflow_config, layout):
    """Editable model declarations alone must freeze weights for each new training request."""
    project = _project(tmp_path, workflow_config, layout)
    filename = {
        "single_model": "single_model.py",
        "model_competition": "model_competition.py",
        "multi_target": "multi_model.py",
    }[layout]
    path = project[2] / filename
    if layout == "multi_target":
        source = path.read_text().replace(
            "'workflow': {}", "'workflow': {'weight_column': 'training_weight'}"
        )
    else:
        source = "WEIGHT_COLUMN = 'training_weight'\n" + path.read_text()
    path.write_bytes(source.encode("utf-8"))
    project[2].joinpath("weights.py").write_text("raise RuntimeError('obsolete hook executed')")
    first = _load(project)
    path.write_text(source.replace("training_weight", "second_weight"), encoding="utf-8")
    second = _load(project)
    assert all(config["weight_column"] == "training_weight" for config in first.values())
    assert all(config["weight_column"] == "second_weight" for config in second.values())
    assert all(config["weights_python_source"] == source for config in first.values())
    assert all(
        config["reserved_weight_columns"] == ["training_weight"] for config in first.values()
    )


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("column", [None, "training_weight"])
def test_model_source_is_executed_once_and_snapshot_is_exact(
    tmp_path, workflow_config, layout, column
):
    """A self-editing model source cannot change the saved setting within a request."""
    overlays = {name: {"weight_column": column} for name in ("risk", "revenue")}
    project = _project(tmp_path, workflow_config, layout, branch_overlays=overlays)
    name = {
        "single_model": "single_model.py",
        "model_competition": "model_competition.py",
        "multi_target": "multi_model.py",
    }[layout]
    path = project[2] / name
    source = (
        "from pathlib import Path\n"
        + f"Path({str(path)!r}).write_text(\"raise RuntimeError('reread')\")\n"
        + f"WEIGHT_COLUMN = {column!r}\n"
        + path.read_text()
    )
    path.write_bytes(source.encode())
    loaded = _load(project)
    for config in loaded.values():
        assert config["weight_column"] == column
        assert config["weights_python_source"] == source
        assert config["weights_python_sha256"] == hashlib.sha256(source.encode()).hexdigest()
        assert config["pipeline"]["modeling"]["params"]["class_weight"] == "balanced"
    assert path.read_text() == "raise RuntimeError('reread')"


@pytest.mark.parametrize(
    "value", ["", "bad.column", " training_weight", 1, True, "_COMMIT_VERSION"]
)
def test_invalid_model_weight_declaration_fails(tmp_path, workflow_config, value):
    """Bad source identifiers must fail before opening any training data."""
    project = _project(tmp_path, workflow_config, "single_model", f"WEIGHT_COLUMN = {value!r}")
    with pytest.raises(ValueError, match="weight_column"):
        _load(project)


@pytest.mark.parametrize(
    "role",
    [
        "input_columns",
        "target_column",
        "record_key_columns",
        "event_column",
        "result_available_at_column",
        "cv_group_column",
    ],
)
def test_disabled_branch_reserves_sibling_weight(tmp_path, workflow_config, role):
    """An opted-out branch cannot use a sibling weight as features or identity metadata."""
    value = ["REVENUE_WEIGHT"] if role.endswith("columns") else "REVENUE_WEIGHT"
    overlays = {
        "risk": {"weight_column": None, role: value},
        "revenue": {"weight_column": "revenue_weight"},
    }
    with pytest.raises(ValueError, match="Weight columns must be distinct"):
        _load(_project(tmp_path, workflow_config, "multi_target", branch_overlays=overlays))


def test_branch_union_and_opt_out(tmp_path, workflow_config):
    """All declarations reserve their weight column even for inactive siblings."""
    overlays = {"risk": {"weight_column": None}, "revenue": {"weight_column": "revenue_weight"}}
    loaded = _load(_project(tmp_path, workflow_config, "multi_target", branch_overlays=overlays))
    assert loaded["risk"]["weight_column"] is None
    assert loaded["revenue"]["weight_column"] == "revenue_weight"
    assert all(
        config["reserved_weight_columns"] == ["revenue_weight"] for config in loaded.values()
    )


def test_saved_workflow_validates_without_reading_model(workflow_config):
    """Replay validates the frozen source roles without consulting editable Python files."""
    config = {**workflow_config, "weight_column": "X", "reserved_weight_columns": ["X"]}
    with pytest.raises(ValueError, match="Weight columns must be distinct"):
        validate_workflow_config(config, action="score")


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_legacy_models_without_declaration_remain_unweighted(tmp_path, workflow_config, layout):
    """Existing models do not acquire weight settings when the declaration is absent."""
    loaded = _load(_project(tmp_path, workflow_config, layout))
    assert all("weight_column" not in config for config in loaded.values())
