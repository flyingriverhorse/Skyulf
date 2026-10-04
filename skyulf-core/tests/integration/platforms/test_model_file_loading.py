"""Model files resolve once into the existing immutable pipeline contract."""

import json
from copy import deepcopy

import pytest

from skyulf.integrations.databricks.branch_notebook import load_training_branch_configs
from skyulf.integrations.databricks.project import load_project_workflow


def _project(tmp_path):
    """Create feature and model directories without relying on generated templates."""
    features = tmp_path / "src/features"
    features.mkdir(parents=True)
    features.joinpath("__init__.py").write_text(
        "def build_preprocessing(recipe='default'):\n    return []\n", encoding="utf-8"
    )
    modeling = tmp_path / "src/modeling"
    modeling.mkdir()
    return features, modeling


def _single_config():
    """Keep model ownership empty until the Python builder resolves it."""
    return {"training_layout": "single_model", "pipeline": {"preprocessing": [], "modeling": {}}}


@pytest.mark.parametrize("legacy_layout", [False, True])
def test_single_model_resolves_parameters_without_mutating_json(tmp_path, legacy_layout):
    """Training must use edited Python parameters while retaining the caller's blank config."""
    features, modeling = _project(tmp_path)
    modeling.joinpath("single_model.py").write_text(
        "def build_modeling():\n"
        "    return {'type': 'ridge_regression', 'params': {'alpha': 7.25}}\n",
        encoding="utf-8",
    )
    config = _single_config()
    if legacy_layout:
        config.pop("training_layout")
    original = deepcopy(config)
    loaded = load_project_workflow(config, features)
    assert loaded["pipeline"]["modeling"] == {"type": "ridge_regression", "params": {"alpha": 7.25}}
    assert config == original


@pytest.mark.parametrize(
    "expression",
    [
        "None",
        "[]",
        "{'params': {'alpha': float('nan')}}",
        "{'params': {'alpha': float('inf')}}",
        "{'params': {'alpha': (1, 2)}}",
        "{1: 'ridge_regression'}",
    ],
)
def test_single_model_rejects_non_json_objects(tmp_path, expression):
    """Saved parameters cannot lose types or nonfinite values during JSON serialization."""
    features, modeling = _project(tmp_path)
    modeling.joinpath("single_model.py").write_text(
        f"def build_modeling():\n    return {expression}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="single_model.py.*JSON"):
        load_project_workflow(_single_config(), features)


@pytest.mark.parametrize(
    "source,error",
    [
        ("build_modeling = 1\n", "build_modeling"),
        ("#" * 65537, "64 KiB"),
        ("def build_modeling():\n    return {'params': {'x': 'x' * 65537}}\n", "64 KiB"),
    ],
    ids=["missing-factory", "source-limit", "output-limit"],
)
def test_single_model_bounds_source_and_output(tmp_path, source, error):
    """Invalid builders and oversized artifacts must fail before training begins."""
    features, modeling = _project(tmp_path)
    modeling.joinpath("single_model.py").write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_project_workflow(_single_config(), features)


def test_single_model_rejects_conflicting_json_before_executing_file(tmp_path):
    """Two model owners must produce an error instead of silently changing a selected model."""
    features, modeling = _project(tmp_path)
    modeling.joinpath("single_model.py").write_text("raise RuntimeError('executed')\n")
    config = _single_config()
    config["pipeline"]["modeling"] = {"type": "linear_regression"}
    with pytest.raises(ValueError, match="modeling.*empty"):
        load_project_workflow(config, features)


def test_legacy_json_model_loads_without_single_file(tmp_path):
    """Existing projects must keep their inline model configuration unchanged."""
    features, _ = _project(tmp_path)
    config = _single_config()
    config["pipeline"]["modeling"] = {"type": "linear_regression"}
    assert load_project_workflow(config, features)["pipeline"]["modeling"] == {
        "type": "linear_regression"
    }


def test_saved_model_parameters_replay_without_editable_file(tmp_path):
    """Saved training plans must retain resolved parameters after project files change."""
    from skyulf.integrations.databricks.local_branches import (
        TrainingBranch,
        branch_training_payload,
        restore_training_branches,
    )
    from skyulf.integrations.databricks.local_retraining import LocalTrainingSpec

    features, modeling = _project(tmp_path)
    path = modeling / "single_model.py"
    path.write_text(
        "def build_modeling():\n"
        "    return {'type': 'ridge_regression', 'params': {'alpha': 7.25}}\n"
    )
    config = load_project_workflow(_single_config(), features)
    branch = TrainingBranch(
        name="result",
        spec=LocalTrainingSpec(
            table="workspace.test.rows",
            version=0,
            record_key_columns=("id",),
            input_columns=("x",),
            target_column="target",
            max_rows=30,
            max_bytes=100000,
            drop_missing_labels=True,
        ),
        pipeline=config["pipeline"],
        model_name="workspace.test.model",
        metric="heldout_rmse",
    )
    payload = json.loads(json.dumps(branch_training_payload((branch,))))
    path.write_text("raise RuntimeError('edited model')\n")
    restored = restore_training_branches(payload)
    assert restored[0].pipeline["modeling"]["params"] == {"alpha": 7.25}
    with pytest.raises(ValueError, match="modeling.*empty"):
        load_project_workflow(config, features)


def _competition_config():
    """Use shared CV policy and a placeholder model replaced by candidate recipes."""
    return {
        "training_layout": "model_competition",
        "task": "regression",
        "metric": "heldout_rmse",
        "cv_enabled": True,
        "cv_folds": 3,
        "target_column": "target",
        "input_columns": ["x"],
        "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
    }


@pytest.mark.parametrize("filename", ["model_competition.py", "candidates.py"])
def test_competition_file_names_resolve_without_single_model(tmp_path, filename):
    """Both generations of competition projects must ignore an unrelated single model file."""
    features, modeling = _project(tmp_path)
    modeling.joinpath("single_model.py").write_text("raise RuntimeError('wrong layout')\n")
    modeling.joinpath(filename).write_text(
        "def build_candidates(task):\n"
        "    assert task == 'regression'\n"
        "    return {'linear': {'modeling': {'type': 'linear_regression'}},\n"
        "            'ridge': {'modeling': {'type': 'ridge_regression', 'params': {'alpha': 3}}}}\n"
    )
    resolved = load_project_workflow(_competition_config(), features)
    assert resolved["competition"]["candidates"]["ridge"]["pipeline"]["modeling"] == {
        "type": "ridge_regression",
        "params": {"alpha": 3},
    }


def test_competition_rejects_ambiguous_files_before_executing(tmp_path):
    """Renaming a model file must never leave two silently competing sources of truth."""
    features, modeling = _project(tmp_path)
    for name in ("model_competition.py", "candidates.py"):
        modeling.joinpath(name).write_text("raise RuntimeError('ambiguous executed')\n")
    with pytest.raises(ValueError, match="model_competition.py.*candidates.py"):
        load_project_workflow(_competition_config(), features)


def _branch_project(tmp_path, workflow_config, filename):
    """Build real notebook inputs with an explicit per-branch model overlay."""
    _, modeling = _project(tmp_path)
    config = deepcopy(workflow_config)
    config.update(training_layout="multi_target", score_handoff="disabled")
    path = tmp_path / "config/workflow.json"
    path.parent.mkdir()
    path.write_text(json.dumps(config))
    modeling.joinpath(filename).write_text(
        "def build_training_branches():\n"
        "    return {'revenue': {'workflow': {'training_layout': 'single_model',\n"
        "        'pipeline': {'preprocessing': [], 'modeling':\n"
        "            {'type': 'ridge_regression', 'params': {'alpha': 5}}}}}}\n"
    )
    modeling.joinpath("single_model.py").write_text("raise RuntimeError('wrong layout')\n")
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
    return modeling, values


@pytest.mark.parametrize("filename", ["multi_model.py", "branches.py"])
def test_multi_model_names_keep_branch_layout_and_model(tmp_path, workflow_config, filename):
    """Independent branch models must survive loading even beside a single-model builder."""
    _, values = _branch_project(tmp_path, workflow_config, filename)
    loaded = load_training_branch_configs(values)["revenue"]
    assert loaded["training_layout"] == "multi_target"
    assert loaded["pipeline"]["modeling"] == {"type": "ridge_regression", "params": {"alpha": 5}}


def test_multi_model_rejects_ambiguous_files(tmp_path, workflow_config):
    """Both branch filenames must fail before selecting or executing either builder."""
    modeling, values = _branch_project(tmp_path, workflow_config, "multi_model.py")
    modeling.joinpath("branches.py").write_text("raise RuntimeError('ambiguous executed')\n")
    with pytest.raises(ValueError, match="multi_model.py.*branches.py"):
        load_training_branch_configs(values)


def _operator_values(action, model_name):
    """Supply concrete saved evidence so notebook preparation needs no registry calls."""
    if action == "train":
        return {}
    if action == "rollback":
        return {
            "expected_champion_version": "2",
            "promotion_receipt_json": json.dumps(
                {
                    "event_id": "promotion",
                    "kind": "promotion",
                    "model_name": model_name,
                    "alias": "champion",
                    "prior_version": "1",
                    "new_version": "2",
                    "comparison_sha256": "a" * 64,
                    "parent_event_id": None,
                }
            ),
        }
    options = {
        "candidate_version": "2",
        "expected_champion_version": "1",
        "comparison_sha256": "a" * 64,
    }
    if action == "reject":
        options["rejection_reason"] = "Quality review declined the candidate."
    return options


@pytest.mark.parametrize("action", ["train", "approve", "reject", "rollback"])
def test_notebook_preparation_loads_single_model_only_for_train(tmp_path, workflow_config, action):
    """Saved-model operations must work when editable model and feature files cannot execute."""
    from skyulf.integrations.databricks.job_runtime import _prepared_notebook_request

    features, modeling = _project(tmp_path)
    model_source = (
        "def build_modeling():\n"
        "    return {'type': 'ridge_regression', 'params': {'alpha': 7.25}}\n"
    )
    if action != "train":
        model_source = "raise RuntimeError('editable model must not execute')\n"
        features.joinpath("__init__.py").write_text(
            "raise RuntimeError('editable features must not execute')\n"
        )
    modeling.joinpath("single_model.py").write_text(model_source)
    config = {**workflow_config, **_single_config()}
    path = tmp_path / "config/workflow.json"
    path.parent.mkdir()
    path.write_text(json.dumps(config))
    values = {
        "config_path": str(path),
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
        "workflow_contract": "3",
        "deployed_score_handoff": config["score_handoff"],
        "lifecycle_action": action,
        **_operator_values(action, config["model_name"]),
    }
    request = _prepared_notebook_request(values, "../src/features")
    expected = {"type": "ridge_regression", "params": {"alpha": 7.25}} if action == "train" else {}
    assert request["config"]["pipeline"]["modeling"] == expected
    assert request["action"] == action


@pytest.mark.parametrize("legacy_layout", [False, True])
def test_score_accepts_empty_single_model_declaration(workflow_config, legacy_layout):
    """Scoring selects a saved model without needing the training-time Python declaration."""
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    config = {**workflow_config, **_single_config()}
    if legacy_layout:
        config.pop("training_layout")
    assert validate_workflow_config(config, action="score") == config


@pytest.mark.parametrize(
    "pipeline",
    [
        None,
        {},
        {"modeling": None},
        {"modeling": []},
        {"modeling": "ridge_regression"},
        {"modeling": {"params": {}}},
        {"modeling": {"type": "missing_model"}},
        {"modeling": {"type": "logistic_regression"}},
    ],
)
def test_saved_actions_still_reject_missing_or_invalid_models(workflow_config, pipeline):
    """The empty-declaration exception must not admit malformed or incompatible models."""
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    config = {**workflow_config, "training_layout": "single_model", "pipeline": pipeline}
    with pytest.raises(ValueError):
        validate_workflow_config(config, action="score")


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"preprocessing": "invalid"}, "preprocessing"),
        (
            {"preprocessing": [{"name": "model", "transformer": "linear_regression"}]},
            "preprocessing",
        ),
        (
            {"preprocessing": [{"name": "unknown", "transformer": "missing_transformer"}]},
            "registry",
        ),
        ({"explainability": {"method": "unsupported"}}, "explainability"),
    ],
)
def test_empty_saved_model_keeps_pipeline_validation(workflow_config, changes, error):
    """Saved-artifact actions still validate preprocessing structure, node types and explanations."""
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    config = {**workflow_config, **_single_config()}
    config["pipeline"].update(changes)
    with pytest.raises(ValueError, match=error):
        validate_workflow_config(config, action="score")


@pytest.mark.parametrize(
    "layout,action",
    [("single_model", "train"), ("multi_target", "score"), ("model_competition", "score")],
)
def test_empty_model_exception_excludes_training_and_other_layouts(workflow_config, layout, action):
    """Training and other layout contracts must still require a resolved model declaration."""
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    config = {**workflow_config, **_single_config(), "training_layout": layout}
    with pytest.raises(ValueError, match="modeling"):
        validate_workflow_config(config, action=action)
