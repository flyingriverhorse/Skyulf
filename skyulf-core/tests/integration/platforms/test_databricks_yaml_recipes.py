"""YAML-owned recipes must preserve project identity, phase order and saved replay."""

import json

import pandas as pd
import pytest

yaml = pytest.importorskip("yaml")

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline
from skyulf.inference.pipeline_scoring import score_pipeline
from skyulf.inference.project_code import load_project_module, project_source_digest
from skyulf.inference.project_package import discard_project_package
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow


@pytest.fixture
def recipe_project(tmp_path):
    """Provide a feature package whose Python defines functions, never recipe lists."""
    features = tmp_path / "src/features"
    features.mkdir(parents=True)
    (features / "__init__.py").write_text("", encoding="utf-8")
    (features / "preprocessing.py").write_text(
        "from skyulf.preprocessing import column_step\n"
        "def double(df, params):\n    return df[params['column']] * 2\n"
        "def doubled(column):\n"
        "    return column_step('double', double, output='twice', params={'column': column})\n",
        encoding="utf-8",
    )
    (tmp_path / "config").mkdir()
    return features


def _declarations(features, phase, recipes):
    """Write the editable declarations at their generated project location."""
    path = features.parents[1] / "config" / f"{phase}.yml"
    path.write_text(yaml.safe_dump({"version": 1, "recipes": recipes}), encoding="utf-8")
    return path


def _workflow(features, **selections):
    """Exercise the real project entrypoint used by training and preview."""
    return load_project_workflow(
        {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}},
        features,
        **selections,
    )


def _fill():
    """Keep a standard recipe step independent of test-specific Python implementations."""
    return {
        "name": "fill",
        "transformer": "SimpleImputer",
        "params": {"columns": ["x"], "strategy": "mean"},
    }


def test_yaml_builtin_and_custom_order_survives_saved_model_reload(recipe_project, tmp_path):
    """Edited YAML or Python must not change predictions made by an existing version."""
    declarations = _declarations(
        recipe_project,
        "preprocessing",
        {"default": [_fill(), {"custom": "preprocessing.doubled", "params": {"column": "x"}}]},
    )
    resolved = _workflow(recipe_project)
    assert resolved["pipeline"]["preprocessing"][0] == _fill()
    assert resolved["pipeline"]["preprocessing"][1]["name"] == "double"
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    artifact = fit_workflow(
        resolved["pipeline"],
        SplitDataset(train=frame, test=frame[:0]),
        target_column="target",
        artifact_path=tmp_path / "model",
        max_rows=10,
        max_bytes=10000,
    )
    raw = pd.DataFrame({"x": [None, 5.0]})
    expected = score_pipeline(raw, artifact)
    source = resolved["pipeline"]["project_python_source"]
    declarations.write_text("not: a recipe", encoding="utf-8")
    (recipe_project / "preprocessing.py").write_text(
        "raise RuntimeError('edited')", encoding="utf-8"
    )
    discard_project_package(load_project_module(source).__name__)
    replay = load_project_module(source)
    assert replay.build_preprocessing() == resolved["pipeline"]["preprocessing"]
    loaded = load_pipeline(tmp_path / "model")
    pd.testing.assert_frame_equal(expected, score_pipeline(raw, loaded))
    assert expected.prediction.tolist() == pytest.approx([6.0, 11.0])


def test_yaml_selection_and_pre_split_are_independent(recipe_project):
    """Selecting one preprocessing recipe must not silently change row eligibility."""
    filter_step = {
        "name": "known_target",
        "transformer": "DropMissingRows",
        "params": {"subset": ["target"], "how": "any"},
    }
    _declarations(recipe_project, "preprocessing", {"default": [], "filled": [_fill()]})
    _declarations(recipe_project, "pre_split", {"default": [filter_step]})
    resolved = _workflow(recipe_project, preprocessing_recipe="filled")
    assert resolved["pipeline"]["preprocessing"] == [_fill()]
    assert resolved["pre_split_steps"] == [filter_step]
    empty = _workflow(recipe_project, preprocessing_recipe="none", pre_split_recipe="none")
    assert empty["pipeline"]["preprocessing"] == []
    assert empty["pre_split_steps"] == []


def test_recipe_declarations_change_source_identity(recipe_project):
    """Changed parameters must invalidate cached project modules and training evidence."""
    _declarations(recipe_project, "preprocessing", {"default": [_fill()]})
    first = _workflow(recipe_project)["pipeline"]["project_python_source"]
    step = _fill()
    step["params"]["strategy"] = "median"
    _declarations(recipe_project, "preprocessing", {"default": [step]})
    second = _workflow(recipe_project)["pipeline"]["project_python_source"]
    assert project_source_digest(first) != project_source_digest(second)
    assert load_project_module(first).build_preprocessing()[0]["params"]["strategy"] == "mean"


@pytest.mark.parametrize(
    "document, message",
    [
        ({"version": True, "recipes": {"default": []}}, "version"),
        ({"version": 1, "recipes": []}, "recipes"),
        ({"version": 1, "recipes": {"typo": []}}, "default"),
        ({"version": 1, "recipes": {"default": [], "none": [_fill()]}}, "none"),
        ({"version": 1, "recipes": {"default": {}}}, "list"),
        ({"version": 1, "recipes": {"default": [{"custom": "os.system"}]}}, "custom"),
        (
            {
                "version": 1,
                "recipes": {
                    "default": [{"custom": "preprocessing.doubled", "transformer": "SimpleImputer"}]
                },
            },
            "custom",
        ),
        (
            {
                "version": 1,
                "recipes": {
                    "default": [{"name": "fill", "transformer": "SimpleImputer", "prams": {}}]
                },
            },
            "prams",
        ),
    ],
)
def test_invalid_recipe_declarations_fail_before_python_execution(
    recipe_project, document, message
):
    """A malformed declaration must never get as far as executing project imports."""
    (recipe_project / "__init__.py").write_text(
        "raise RuntimeError('must not execute')", encoding="utf-8"
    )
    path = recipe_project.parents[1] / "config/preprocessing.yml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _workflow(recipe_project)


def test_python_and_yaml_recipe_owners_are_rejected(recipe_project):
    """A second recipe definition must not silently override a user's chosen steps."""
    (recipe_project / "__init__.py").write_text(
        "def build_preprocessing(recipe='default'):\n    return []\n", encoding="utf-8"
    )
    _declarations(recipe_project, "preprocessing", {"default": [_fill()]})
    with pytest.raises(ValueError, match="both.*preprocessing"):
        _workflow(recipe_project)


def test_unknown_recipe_does_not_fall_back_to_default(recipe_project):
    """Misspelled recipe selection must stop training instead of omitting preprocessing."""
    _declarations(recipe_project, "preprocessing", {"default": [_fill()]})
    with pytest.raises(ValueError, match="Unknown.*recipe"):
        _workflow(recipe_project, preprocessing_recipe="typo")


def test_yaml_recipe_results_are_fresh_copies(recipe_project):
    """A caller editing a resolved recipe must not mutate another model's configuration."""
    _declarations(recipe_project, "preprocessing", {"default": [_fill()]})
    source = _workflow(recipe_project)["pipeline"]["project_python_source"]
    module = load_project_module(source)
    first = module.build_preprocessing()
    first[0]["params"]["columns"].append("another")
    assert module.build_preprocessing() == [_fill()]


def _training_project(features, workflow_config, models):
    """Use the same JSON/YAML boundary as existing project lifecycle callers."""
    directory = features.parents[1] / "config"
    config = {**workflow_config, "pipeline": {"preprocessing": [], "modeling": {}}}
    path = directory / "workflow.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    (directory / "training.yml").write_text(
        yaml.safe_dump({"version": 1, "models": models}), encoding="utf-8"
    )
    return path


def test_yaml_recipes_reach_competition_candidates(recipe_project, workflow_config):
    """Candidate choices and shared custom eligibility must survive saved-source replay."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    workflow_config.update(
        training_layout="model_competition",
        cv_enabled=True,
        cv_folds=2,
        cv_type="k_fold",
        cv_shuffle=True,
        cv_random_state=42,
    )
    (recipe_project / "pre_split.py").write_text(
        "from skyulf.preprocessing import filter_step\n"
        "def valid(df):\n    return df['x'].notna()\n"
        "def eligible():\n    return filter_step('valid', valid, columns=['x'])\n",
        encoding="utf-8",
    )
    _declarations(recipe_project, "pre_split", {"default": [{"custom": "pre_split.eligible"}]})
    _declarations(recipe_project, "preprocessing", {"default": [], "filled": [_fill()]})
    path = _training_project(
        recipe_project,
        workflow_config,
        {
            "ridge": {"model": {"type": "ridge_regression"}, "preprocessing_recipe": "filled"},
            "linear": {"model": {"type": "linear_regression"}, "preprocessing_recipe": "none"},
        },
    )
    result = load_project_workflow(read_workflow_config(path), recipe_project)
    candidates = result["competition"]["candidates"]
    assert candidates["ridge"]["pipeline"]["preprocessing"] == [_fill()]
    assert candidates["linear"]["pipeline"]["preprocessing"] == []
    for candidate in candidates.values():
        module = load_project_module(candidate["pipeline"]["project_python_source"])
        assert module.build_pre_split_steps() == result["pre_split_steps"]
    assert result["pre_split_steps"][0]["name"] == "valid"


def test_yaml_recipes_reach_independent_target_branches(recipe_project, workflow_config):
    """Branch-level selectors must resolve their own YAML steps through notebook loading."""
    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        load_training_branch_configs,
    )

    workflow_config.update(training_layout="multi_target", score_handoff="disabled")
    workflow_config.pop("target_column")
    _declarations(recipe_project, "preprocessing", {"default": [], "filled": [_fill()]})
    _declarations(recipe_project, "pre_split", {"default": []})
    path = _training_project(
        recipe_project,
        workflow_config,
        {
            "revenue": {
                "model": {"type": "ridge_regression"},
                "target_column": "revenue",
                "preprocessing_recipe": "filled",
            },
            "cost": {
                "model": {"type": "linear_regression"},
                "target_column": "cost",
                "preprocessing_recipe": "none",
            },
        },
    )
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
    branches = load_training_branch_configs(values)
    assert branches["revenue"]["pipeline"]["preprocessing"] == [_fill()]
    assert branches["cost"]["pipeline"]["preprocessing"] == []
    source = branches["revenue"]["pipeline"]["project_python_source"]
    _declarations(recipe_project, "preprocessing", {"default": []})
    assert load_project_module(source).build_preprocessing() == [_fill()]


def test_smoke_validates_yaml_without_importing_custom_functions(recipe_project, workflow_config):
    """Offline checks must report YAML errors without executing valid custom hooks either."""
    from skyulf.integrations.databricks.projects.project_checks import check_project

    (recipe_project / "__init__.py").write_text(
        "raise RuntimeError('must not execute')", encoding="utf-8"
    )
    path = recipe_project.parents[1] / "config/workflow.json"
    path.write_text(json.dumps(workflow_config), encoding="utf-8")
    _declarations(
        recipe_project,
        "preprocessing",
        {"default": [{"custom": "preprocessing.doubled", "params": {"column": "x"}}]},
    )
    bindings = {
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
    }
    result = check_project(recipe_project.parents[1], bindings)
    assert result["project_hooks_executed"] is False
    assert result["project_packages"] == ["src/features"]
    _declarations(recipe_project, "preprocessing", {"default": {}})
    with pytest.raises(ValueError, match="preprocessing.yml.*list"):
        check_project(recipe_project.parents[1], bindings)
