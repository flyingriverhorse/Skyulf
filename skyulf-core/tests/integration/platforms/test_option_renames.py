"""Public YAML option names reach the runtime under their internal names; old names fail."""

import json

import pytest


def _project(tmp_path):
    """Make a minimal organized project with a JSON workflow and empty YAML owners."""
    (tmp_path / "config").mkdir()
    features = tmp_path / "src/features"
    features.mkdir(parents=True)
    (features / "__init__.py").write_text(
        "def build_preprocessing(recipe='default'):\n    return []\n", encoding="utf-8"
    )
    (tmp_path / "src/modeling").mkdir()
    path = tmp_path / "config/workflow.json"
    path.write_text(
        json.dumps(
            {"training_layout": "single_model", "pipeline": {"preprocessing": [], "modeling": {}}}
        ),
        encoding="utf-8",
    )
    return path


def _inference(path, body):
    """Write config/inference.yml next to the workflow."""
    path.with_name("inference.yml").write_text("version: 1\n" + body, encoding="utf-8")


def test_public_inference_options_become_internal_non_default_values(tmp_path):
    """Non-default public values must change the internal settings that runtime code reads."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path = _project(tmp_path)
    _inference(
        path,
        "follow_champion: true\non_model_change: rescore_all\n"
        "rescore_when_history_expires: true\nbatch_compute: spark\n",
    )
    config = read_workflow_config(path)
    assert config["score_handoff"] == "after_alias_change"
    assert config["model_change_mode"] == "full_rebuild"
    assert config["auto_rebuild_on_cdf_expiry"] is True
    assert config["inference_mode"] == "spark"
    for public in (
        "follow_champion",
        "on_model_change",
        "rescore_when_history_expires",
        "batch_compute",
    ):
        assert public not in config


def test_public_inference_defaults_map_to_existing_internal_defaults(tmp_path):
    """The other public values must map to today's behavior, not to a silent default."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path = _project(tmp_path)
    _inference(
        path,
        "follow_champion: false\non_model_change: score_new_rows\nbatch_compute: single_node\n",
    )
    config = read_workflow_config(path)
    assert config["score_handoff"] == "disabled"
    assert config["model_change_mode"] == "incremental_append"
    assert config["inference_mode"] == "local"


@pytest.mark.parametrize(
    "line",
    [
        "score_handoff: after_alias_change",
        "model_change_mode: full_rebuild",
        "auto_rebuild_on_cdf_expiry: true",
        "inference_mode: spark",
    ],
)
def test_old_inference_names_are_rejected(tmp_path, line):
    """Renamed keys have no compatibility alias; the old spelling must fail loudly."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path = _project(tmp_path)
    _inference(path, line + "\n")
    with pytest.raises(ValueError, match="Unknown inference.yml settings"):
        read_workflow_config(path)


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("follow_champion: yes_please", "follow_champion must be one of"),
        ("follow_champion: 1", "follow_champion must be one of"),
        ("on_model_change: full_rebuild", "on_model_change must be one of"),
        ("batch_compute: local", "batch_compute must be one of"),
        ("rescore_when_history_expires: 'true'", "must be true or false"),
    ],
)
def test_invalid_public_values_name_the_public_option(tmp_path, line, match):
    """Errors must use the option name and values the user actually writes."""
    from skyulf.integrations.databricks.projects.yaml_config import read_inference_config

    path = _project(tmp_path)
    _inference(path, line + "\n")
    with pytest.raises(ValueError, match=match):
        read_inference_config(path.parent)


def test_public_training_options_reach_defaults_and_model_overrides(tmp_path):
    """Training defaults and per-model overrides both translate before the workflow merge."""
    from skyulf.integrations.databricks.projects.yaml_config import read_training_config

    path = _project(tmp_path)
    path.with_name("training.yml").write_text(
        "version: 1\ndefaults:\n  save_preprocessing_diagnostics: true\n"
        "  allow_tested_custom_steps: true\nmodels:\n  main:\n    model: {type: linear_regression}\n"
        "    save_preprocessing_diagnostics: false\n",
        encoding="utf-8",
    )
    training = read_training_config(path.parent)
    assert training is not None
    assert training["defaults"]["preprocessing_probe"] is True
    assert training["defaults"]["admit_probed_custom_steps"] is True
    assert training["models"]["main"]["preprocessing_probe"] is False
    assert "save_preprocessing_diagnostics" not in training["defaults"]


@pytest.mark.parametrize("key", ["preprocessing_probe", "admit_probed_custom_steps"])
def test_old_training_names_are_rejected(tmp_path, key):
    """Old training option names are unknown settings, also inside a model override."""
    from skyulf.integrations.databricks.projects.yaml_config import read_training_config

    path = _project(tmp_path)
    path.with_name("training.yml").write_text(
        f"version: 1\nmodels:\n  main:\n    model: {{type: linear_regression}}\n    {key}: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown model main settings"):
        read_training_config(path.parent)


_MODEL_SET = (
    "model_set:\n  model_name: '{catalog}.{metadata_schema}.financial'\n"
    "  prediction_table: '{catalog}.{output_schema}.financial_scores'\n"
)


def test_model_set_on_source_change_translates(tmp_path):
    """rescore_all must select the existing rebuild policy for model-set source corrections."""
    from skyulf.integrations.databricks.projects.yaml_config import read_inference_config

    path = _project(tmp_path)
    for value, internal in (("rescore_all", "rebuild_on_change"), ("stop", "reject")):
        _inference(path, _MODEL_SET + f"  on_source_change: {value}\n")
        document = read_inference_config(path.parent)
        assert document is not None
        assert document["model_set"]["source_change_policy"] == internal


def test_model_set_old_source_change_name_is_rejected(tmp_path):
    """The internal model-set key must not be accepted from YAML."""
    from skyulf.integrations.databricks.projects.yaml_config import read_inference_config

    path = _project(tmp_path)
    _inference(path, _MODEL_SET + "  source_change_policy: rebuild_on_change\n")
    with pytest.raises(ValueError, match="source_change_policy"):
        read_inference_config(path.parent)
