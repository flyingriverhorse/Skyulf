"""Competition recipes preserve common inputs and freeze each candidate pipeline."""

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from skyulf.inference.project_code import load_project_module
from skyulf.integrations.databricks.projects.competition_project import validate_competition_config
from skyulf.integrations.databricks.projects.project import load_project_workflow


def _project(tmp_path, candidates=None):
    """Create an executable package with independent named feature recipes."""
    features = tmp_path / "features"
    features.mkdir()
    features.joinpath("__init__.py").write_text(
        "def build_preprocessing(recipe='default'):\n"
        "    if recipe == 'default': return []\n"
        "    if recipe != 'scaled': raise ValueError('Unknown recipe')\n"
        "    return [{'name': 'scale', 'transformer': 'StandardScaler', 'params': {'columns': ['x']}}]\n"
        "def build_pre_split_steps(recipe='default'):\n"
        "    assert recipe == 'default'\n"
        "    return []\n",
        encoding="utf-8",
    )
    modeling = tmp_path / "modeling"
    modeling.mkdir()
    if candidates is None:
        candidates = {
            "linear": {"modeling": {"type": "linear_regression"}},
            "ridge": {
                "modeling": {"type": "ridge_regression"},
                "preprocessing_recipe": "scaled",
            },
        }
    modeling.joinpath("candidates.py").write_text(
        f"def build_candidates(task):\n    assert task == 'regression'\n    return {candidates!r}\n",
        encoding="utf-8",
    )
    return features


def _config():
    """Keep workflow-owned inputs separate from candidate model recipes."""
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


def test_candidates_resolve_named_recipes_and_capture_source(tmp_path):
    """Each saved candidate must replay its own preprocessing without editable files."""
    root = _project(tmp_path)
    (root / "groups").mkdir()
    (root / "groups/company.py").write_text("raise AssertionError('Spark producer')\n")
    config = _config()
    original = deepcopy(config)
    resolved = load_project_workflow(config, root)
    candidates = resolved["competition"]["candidates"]
    assert list(candidates) == ["linear", "ridge"]
    assert candidates["linear"]["pipeline"]["preprocessing"] == []
    ridge = candidates["ridge"]["pipeline"]
    assert ridge["preprocessing"][0]["transformer"] == "StandardScaler"
    replay = load_project_module(ridge["project_python_source"])
    assert replay.build_preprocessing() == ridge["preprocessing"]
    assert resolved["competition"]["candidates_source"]
    assert config == original


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        ({"modeling": {"type": "logistic_regression"}}, "task"),
        ({"modeling": {"type": "ridge_regression"}, "target_column": "other"}, "setting"),
        ({"modeling": {"type": "ridge_regression"}, "pre_split_recipe": "other"}, "setting"),
        ({"modeling": {"type": "ridge_regression"}, "preprocessing_recipe": "missing"}, "recipe"),
    ],
)
def test_invalid_candidates_fail_during_loading(tmp_path, entry, error):
    """Mixed tasks and candidate overrides must fail before opening training data."""
    root = _project(tmp_path, {"first": {"modeling": {"type": "linear_regression"}}, "bad": entry})
    with pytest.raises(ValueError, match=error):
        load_project_workflow(_config(), root)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("cv_enabled", False, "cv_enabled"),
        ("competition_max_candidates", 1, "competition_max_candidates"),
        ("competition_max_candidates", True, "competition_max_candidates"),
        ("competition_max_candidates", 33, "competition_max_candidates"),
        ("competition_max_trials", 0, "competition_max_trials"),
        ("competition_max_trials", 1.5, "competition_max_trials"),
    ],
)
def test_invalid_competition_bounds_fail_early(tmp_path, key, value, error):
    """Competition must require CV and bounded integer admission controls."""
    root = _project(tmp_path)
    with pytest.raises(ValueError, match=error):
        load_project_workflow({**_config(), key: value}, root)


@pytest.mark.parametrize("names", [["one"], ["one", "../two"], ["one", "two", "three"]])
def test_candidate_names_and_count_are_bounded(tmp_path, names):
    """Candidate identities must remain safe and inside the configured count limit."""
    root = _project(tmp_path, {name: {"modeling": {"type": "linear_regression"}} for name in names})
    with pytest.raises(ValueError, match="candidate"):
        load_project_workflow({**_config(), "competition_max_candidates": 2}, root)


@pytest.mark.parametrize(
    "override,error", [({"metric": "r2"}, "metric"), ({"cv_folds": 2}, "cv_folds")]
)
def test_search_cannot_override_shared_selection_policy(tmp_path, override, error):
    """A tuner must compare candidates with the same objective and fold policy."""
    tuner = {
        "type": "hyperparameter_tuner",
        "base_model": {"type": "ridge_regression"},
        "search_space": {"alpha": [0.1, 1.0]},
        **override,
    }
    root = _project(
        tmp_path,
        {"fixed": {"modeling": {"type": "linear_regression"}}, "search": {"modeling": tuner}},
    )
    with pytest.raises(ValueError, match=error):
        load_project_workflow(_config(), root)


def test_custom_recipe_registers_only_selected_candidate_and_replays(tmp_path):
    """Captured custom registrations must survive competition recipe selection."""
    root = _project(tmp_path)
    template = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/features/preprocessing.py"
    )
    shutil.copyfile(template, root / "custom.py")
    root.joinpath("__init__.py").write_text(
        "from .custom import frequency_encoding\n"
        "def build_preprocessing(recipe='default'):\n"
        "    return [] if recipe == 'default' else [frequency_encoding(['x'])]\n",
        encoding="utf-8",
    )
    result = load_project_workflow(_config(), root)
    pipeline = result["competition"]["candidates"]["ridge"]["pipeline"]
    source = pipeline["project_python_source"]
    root.joinpath("__init__.py").write_text("raise RuntimeError('edited')\n", encoding="utf-8")
    replay = load_project_module(source)
    assert replay.build_preprocessing() == pipeline["preprocessing"]
    assert pipeline["preprocessing"][0]["params"]["params"]["columns"] == ["x"]


def test_competition_binds_metric_and_runs_ensemble_before_tuning(tmp_path):
    """Candidate searches must consume merged ensemble parameters and shared objectives."""
    tuner = {
        "type": "hyperparameter_tuner",
        "base_model": {
            "type": "voting_regressor",
            "params": {"base_estimators": ["linear_regression", "ridge"]},
        },
        "strategy": "grid",
        "search_space": {},
    }
    root = _project(
        tmp_path,
        {"fixed": {"modeling": {"type": "linear_regression"}}, "search": {"modeling": tuner}},
    )
    modeling = tmp_path / "modeling"
    modeling.joinpath("ensemble.py").write_text(
        "def build_ensemble_params(model_type):\n"
        "    return {'weights': {'linear_regression': 1.0, 'ridge': 2.0}, 'n_jobs': 1}\n",
        encoding="utf-8",
    )
    modeling.joinpath("tuning.py").write_text(
        "def build_search_space(*, model_type, strategy, params):\n"
        "    assert params['weights']['ridge'] == 2.0\n"
        "    return {'ridge__alpha': [0.5, 1.0]}\n",
        encoding="utf-8",
    )
    result = load_project_workflow(_config(), root)
    pipeline = result["competition"]["candidates"]["search"]["pipeline"]
    assert pipeline["modeling"]["metric"] == "rmse"
    assert pipeline["modeling"]["search_space"] == {"ridge__alpha": [0.5, 1.0]}
    assert pipeline["ensemble_python_source"]
    assert pipeline["search_python_source"]


@pytest.mark.parametrize("expression", ["float('nan')", "float('inf')", "(1, 2)", "{1: 2}"])
def test_candidates_require_finite_json_output(tmp_path, expression):
    """Hook results cannot silently change type or lose nonfinite values in saved evidence."""
    root = _project(tmp_path)
    (tmp_path / "modeling/candidates.py").write_text(
        "def build_candidates(task):\n"
        "    return {'a': {'modeling': {'type': 'ridge_regression', 'params': {'alpha': "
        + expression
        + "}}}, 'b': {'modeling': {'type': 'linear_regression'}}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="candidates.py.*JSON"):
        load_project_workflow(_config(), root)


def test_resolved_validation_does_not_mutate_caller(tmp_path):
    """SDK preflight must preserve the captured candidate plans exactly."""
    config = load_project_workflow(_config(), _project(tmp_path))
    original = deepcopy(config)
    validate_competition_config(config)
    assert config == original


@pytest.mark.parametrize("key", ["target_column", "cv_folds", "pre_split_steps"])
def test_resolved_sdk_candidate_cannot_override_shared_settings(tmp_path, key):
    """Direct SDK callers must receive the same candidate ownership checks as project hooks."""
    config = load_project_workflow(_config(), _project(tmp_path))
    config["competition"]["candidates"]["ridge"][key] = "override"
    with pytest.raises(ValueError, match="settings"):
        validate_competition_config(config)


@pytest.mark.parametrize("key", ["target_column", "input_columns", "pre_split_steps", "cv_folds"])
def test_resolved_pipeline_cannot_embed_workflow_overrides(tmp_path, key):
    """Moving a shared override inside a pipeline must not bypass the SDK boundary."""
    config = load_project_workflow(_config(), _project(tmp_path))
    config["competition"]["candidates"]["ridge"]["pipeline"][key] = "override"
    with pytest.raises(ValueError, match="shared workflow"):
        validate_competition_config(config)


def test_candidate_row_filters_fail_before_training(tmp_path):
    """Candidates cannot compare different eligible populations on nominally shared folds."""
    config = load_project_workflow(_config(), _project(tmp_path))
    config["competition"]["candidates"]["ridge"]["pipeline"]["preprocessing"] = [
        {"name": "filter", "transformer": "DropMissingRows", "params": {"columns": ["x"]}}
    ]
    with pytest.raises(ValueError, match="shared pre_split"):
        validate_competition_config(config)


def _shared_filter_project(tmp_path, reuse_scoring=True):
    """Build real template custom classes used by both candidate and shared recipe paths."""
    root = _project(tmp_path)
    template = (
        Path(__file__).resolve().parents[3]
        / "templates/databricks/template/{{.project_name}}/src/features"
    )
    for filename in ("preprocessing.py", "pre_split.py"):
        shutil.copyfile(template / filename, root / filename)
    root.joinpath("__init__.py").write_text(
        "from .preprocessing import frequency_encoding\n"
        "from .pre_split import minimum_completeness\n"
        "def build_preprocessing(recipe='default'):\n"
        "    steps = [frequency_encoding(['category'])]\n"
        "    if recipe == 'scaled':\n"
        "        steps.append({'name':'scale', 'transformer':'StandardScaler', 'params':{'columns':['x']}})\n"
        "    return steps\n"
        "def build_pre_split_steps():\n"
        "    return [minimum_completeness(['x'], min_present=1)]\n"
        + (
            "def build_scoring():\n"
            "    return {'reuse_pre_split': True, 'skip_target_steps': False}\n"
            if reuse_scoring
            else ""
        ),
        encoding="utf-8",
    )
    return root, load_project_workflow({**_config(), "input_columns": ["x", "category"]}, root)


@pytest.mark.parametrize("reuse_scoring", [False, True])
def test_named_candidates_share_custom_presplit_identity_and_fresh_replay(tmp_path, reuse_scoring):
    """Recipe selection must preserve one canonical custom eligibility identity on fresh replay."""
    root, result = _shared_filter_project(tmp_path, reuse_scoring)
    candidates = result["competition"]["candidates"]
    custom_ids = []
    for candidate in candidates.values():
        pipeline = candidate["pipeline"]
        saved = load_project_module(pipeline["project_python_source"])
        assert saved.build_pre_split_steps() == result["pre_split_steps"]
        # Each candidate's learn function resolves from its own saved source snapshot.
        custom_ids.append(pipeline["preprocessing"][0]["params"]["learn"])
        if reuse_scoring:
            assert pipeline["project_scoring"]["pre_split"]["steps"] == result["pre_split_steps"]
    assert custom_ids[0] != custom_ids[1]
    payload = tmp_path / "saved.json"
    payload.write_text(json.dumps(result), encoding="utf-8")
    root.joinpath("__init__.py").write_text("raise RuntimeError('edited')\n", encoding="utf-8")
    code = (
        "import json,sys\n"
        "from pathlib import Path\n"
        "from skyulf.inference.project_code import load_project_module\n"
        "from skyulf.inference.project_scoring import validate_scoring_config\n"
        "from skyulf.registry import NodeRegistry\n"
        "saved=json.loads(Path(sys.argv[1]).read_text())\n"
        "for candidate in saved['competition']['candidates'].values():\n"
        "    p=candidate['pipeline']; source=p['project_python_source']\n"
        "    module=load_project_module(source)\n"
        "    assert module.build_pre_split_steps()==saved['pre_split_steps']\n"
        "    assert module.build_preprocessing()==p['preprocessing']\n"
        "    for step in saved['pre_split_steps']: NodeRegistry.get_calculator(step['transformer'])\n"
        "    if 'project_scoring' in p: validate_scoring_config(p['project_scoring'],source)\n"
        "print('replayed')\n"
    )
    replay = subprocess.run(
        [sys.executable, "-c", code, str(payload)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stderr
    assert replay.stdout.strip() == "replayed"


@pytest.mark.parametrize("change", ["parameters", "candidate_identity", "builder"])
def test_shared_scoring_requires_exact_canonical_filter_recipe(tmp_path, change):
    """Canonical source support must not admit modified filters or candidate-local substitutes."""
    from skyulf.inference.project_scoring import validate_scoring_config

    _, result = _shared_filter_project(tmp_path)
    pipeline = result["competition"]["candidates"]["ridge"]["pipeline"]
    source = pipeline["project_python_source"]
    policy = deepcopy(pipeline["project_scoring"])
    if change == "parameters":
        policy["pre_split"]["steps"][0]["params"]["columns"] = ["category"]
    elif change == "candidate_identity":
        policy["pre_split"]["steps"][0] = load_project_module(source).minimum_completeness(["x"])
    else:
        source += "\nbuild_pre_split_steps = lambda: []\n"
    with pytest.raises(ValueError, match="canonical"):
        validate_scoring_config(policy, source)
