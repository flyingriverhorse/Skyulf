"""Model declarations use readable YAML with one clear owner per setting."""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"
CLI_ENABLED = (
    bool(os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"))
    or os.environ.get("SKYULF_BUNDLE_OFFLINE_CLI") == "1"
)


def test_model_templates_use_readable_names():
    """Every layout must share the documented training and inference entrypoints."""
    directory = ROOT / "template/{{.project_name}}"
    assert (directory / "config/training.yml.tmpl").is_file()
    assert (directory / "config/inference.yml.tmpl").is_file()
    assert not (directory / "src/modeling").exists()


@pytest.mark.skipif(not CLI_ENABLED, reason="CLI opt-in")
@pytest.mark.parametrize(
    "layout,count", [("single_model", 1), ("model_competition", 2), ("multi_target", 2)]
)
def test_generated_settings_are_yaml_mappings(tmp_path, layout, count):
    """Users edit plain model settings without embedded JSON strings or Python factories."""
    from test_databricks_bundle_generation import _generate_project

    from skyulf.integrations.databricks.projects.yaml_config import read_training_config

    project = _generate_project(tmp_path, training_layout=layout, competition_candidate_count="2")
    document = read_training_config(project / "config")
    assert document is not None
    assert len(document["models"]) == count
    for entry in document["models"].values():
        assert isinstance(entry["model"], dict)
        assert isinstance(entry["tuning"]["search_space"], dict)
        assert "decision_threshold" not in entry
    assert not (project / "src/modeling").exists()


@pytest.mark.skipif(not CLI_ENABLED, reason="CLI opt-in")
def test_explicit_json_initialization_becomes_equivalent_yaml_values(tmp_path):
    """Booleans/null and escaped strings must retain exact values after YAML generation."""
    from test_databricks_bundle_generation import _generate_project, _read_modeling

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
    model = _read_modeling(project)
    assert model["base_model"]["params"] == params
    assert model["search_space"] == {"fit_intercept": [True, False]}
