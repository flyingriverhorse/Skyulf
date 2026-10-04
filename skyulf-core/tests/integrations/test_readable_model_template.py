"""Model declarations are editable Python with one clear owner per layout."""

import ast
import json
import os
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "templates/databricks"


def test_model_templates_use_readable_names():
    """New projects must present the same file names as their documented training layouts."""
    directory = ROOT / "template/{{.project_name}}/src/modeling"
    for name in ("single_model", "model_competition", "multi_model"):
        assert (directory / f"{name}.py.tmpl").is_file()
    assert not (directory / "candidates.py.tmpl").exists()
    assert not (directory / "branches.py.tmpl").exists()


@pytest.mark.skipif(not os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"), reason="CLI opt-in")
@pytest.mark.parametrize(
    "layout,name",
    [
        ("single_model", "single_model"),
        ("model_competition", "model_competition"),
        ("multi_target", "multi_model"),
    ],
)
def test_generated_settings_are_python_dicts(tmp_path, layout, name):
    """Users must edit real model dictionaries instead of JSON inside Python strings."""
    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(tmp_path, training_layout=layout, competition_candidate_count="2")
    source = (project / f"src/modeling/{name}.py").read_text()
    tree = ast.parse(source)
    assert any(isinstance(node, ast.Dict) for node in ast.walk(tree))
    assert "json.loads" not in source
    module = runpy.run_path(str(project / f"src/modeling/{name}.py"))
    if layout == "single_model":
        config = json.loads((project / "config/workflow.json").read_text())
        assert config["pipeline"]["modeling"] == {}
        assert module["build_modeling"]()["type"] == "hyperparameter_tuner"
        assert module["DECISION_THRESHOLD"] == {"mode": "off"}
    elif layout == "model_competition":
        assert len(module["build_candidates"](task="regression")) == 2
        assert all(
            model["decision_threshold"] == {"mode": "off"}
            for model in module["build_candidates"](task="regression").values()
        )
    else:
        assert len(module["build_training_branches"]()) == 2
        assert all(
            model["workflow"]["pipeline"]["decision_threshold"] == {"mode": "off"}
            for model in module["build_training_branches"]().values()
        )


@pytest.mark.skipif(not os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"), reason="CLI opt-in")
def test_explicit_json_initialization_becomes_equivalent_python_values(tmp_path):
    """Booleans/null and quoted keywords or escaped strings must retain their exact values."""
    from test_databricks_bundle_generation import _generate_project

    params = {
        "enabled": True,
        "unset": None,
        "text": 'false null true / "quote" \\ud800',
        "unicode": "\U0001f600",
        "nested": [False, {"name": "null"}],
    }
    project = _generate_project(
        tmp_path,
        model_params=json.dumps(params).replace("/", r"\/"),
        search_space='{"fit_intercept": [true, false]}',
    )
    module = runpy.run_path(str(project / "src/modeling/single_model.py"))
    model = module["build_modeling"]()
    assert model["base_model"]["params"] == params
    assert model["search_space"] == {"fit_intercept": [True, False]}
