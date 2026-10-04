"""Project ensemble recipes must be bounded, optional and pinned before search."""

import hashlib
from pathlib import Path

import pytest

from skyulf.integrations.databricks.project import load_project_workflow


def _project(tmp_path: Path, model_type: str = "voting_regressor") -> tuple[dict, Path]:
    """Build the smallest trusted project that can resolve optional sibling hooks."""
    source = tmp_path / "preprocessing.py"
    source.write_text("def build_preprocessing():\n    return []\n", encoding="utf-8")
    config = {
        "pipeline": {
            "preprocessing": [],
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {
                    "type": model_type,
                    "params": {"base_estimators": ["linear_regression", "ridge"]},
                },
                "strategy": "grid",
                "search_space": {},
            },
        }
    }
    return config, source


def test_ensemble_hook_merges_before_search_and_pins_source(tmp_path):
    """Search must see the effective ensemble composition and its exact recipe source."""
    config, source = _project(tmp_path)
    code = (
        "def build_ensemble_params(model_type):\n"
        "    assert model_type == 'voting_regressor'\n"
        "    return {'weights': {'linear_regression': 1.0, 'ridge': 2.0}, 'n_jobs': 1}\n"
    )
    hook = tmp_path / "ensemble.py"
    hook.write_text(code, encoding="utf-8")
    (tmp_path / "tuning.py").write_text(
        "def build_search_space(*, model_type, strategy, params):\n"
        "    assert params['weights']['ridge'] == 2.0\n"
        "    return {'ridge__alpha': [0.5, 1.0]}\n",
        encoding="utf-8",
    )

    result = load_project_workflow(config, source)

    pipeline = result["pipeline"]
    params = pipeline["modeling"]["base_model"]["params"]
    assert params == {
        "base_estimators": ["linear_regression", "ridge"],
        "weights": {"linear_regression": 1.0, "ridge": 2.0},
        "n_jobs": 1,
    }
    assert pipeline["modeling"]["search_space"] == {"ridge__alpha": [0.5, 1.0]}
    pinned = hook.read_bytes().decode("utf-8")
    assert pipeline["ensemble_python_source"] == pinned
    assert pipeline["ensemble_python_sha256"] == hashlib.sha256(pinned.encode()).hexdigest()
    assert config["pipeline"]["modeling"]["base_model"]["params"] == {
        "base_estimators": ["linear_regression", "ridge"]
    }


@pytest.mark.parametrize(
    "hook_code", [None, "def build_ensemble_params(model_type):\n    return None\n"]
)
def test_optional_ensemble_hook_preserves_json_params(tmp_path, hook_code):
    """Existing projects and an unused recipe should retain their configured composition."""
    config, source = _project(tmp_path)
    if hook_code is not None:
        (tmp_path / "ensemble.py").write_text(hook_code, encoding="utf-8")

    result = load_project_workflow(config, source)

    assert result["pipeline"]["modeling"]["base_model"]["params"] == {
        "base_estimators": ["linear_regression", "ridge"]
    }
    assert ("ensemble_python_source" in result["pipeline"]) is (hook_code is not None)


def test_nonensemble_skips_sibling_code(tmp_path):
    """A normal model must never execute unrelated ensemble project code."""
    config, source = _project(tmp_path, "ridge_regression")
    (tmp_path / "ensemble.py").write_text("raise RuntimeError('must not execute')\n")

    result = load_project_workflow(config, source)

    assert "ensemble_python_source" not in result["pipeline"]


@pytest.mark.parametrize(
    "hook_code",
    [
        "answer = 1\n",
        "def build_ensemble_params(model_type):\n    return []\n",
        "def build_ensemble_params(model_type):\n    return {'weights': {'ridge': float('nan')}}\n",
        "def build_ensemble_params(model_type):\n    return {'weights': object()}\n",
    ],
)
def test_invalid_ensemble_recipe_fails_before_search(tmp_path, hook_code):
    """Invalid or non-JSON hook output must fail before the tuner runs."""
    config, source = _project(tmp_path)
    (tmp_path / "ensemble.py").write_text(hook_code, encoding="utf-8")

    with pytest.raises(ValueError, match="ensemble.py|build_ensemble_params"):
        load_project_workflow(config, source)


def test_ensemble_source_is_bounded(tmp_path):
    """Oversized project code cannot enter the pinned training request."""
    config, source = _project(tmp_path)
    (tmp_path / "ensemble.py").write_bytes(b"#" * (64 * 1024 + 1))

    with pytest.raises(ValueError, match="ensemble.py source exceeds 64 KiB"):
        load_project_workflow(config, source)


def test_ensemble_output_is_bounded(tmp_path):
    """A short hook cannot produce an unbounded pinned parameter payload."""
    config, source = _project(tmp_path)
    (tmp_path / "ensemble.py").write_text(
        "def build_ensemble_params(model_type):\n    return {'weights': {'ridge': 'x' * 70000}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ensemble.py returned more than 64 KiB"):
        load_project_workflow(config, source)


@pytest.mark.parametrize("hook_name", ["ensemble.py", "tuning.py"])
def test_new_template_omits_shared_modeling_hooks(hook_name):
    """Fresh projects must keep model settings in their own model definitions."""
    template = (
        Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"
    )
    source = template / "src/modeling" / hook_name
    assert not source.exists()
    assert not source.with_suffix(".py.tmpl").exists()
