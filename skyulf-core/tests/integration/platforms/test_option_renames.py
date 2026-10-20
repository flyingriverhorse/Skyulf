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


@pytest.mark.parametrize("delivery", ["batch", "both", "endpoint"])
def test_omitted_optional_inference_policies_validate_and_preview(
    tmp_path, workflow_config, delivery
):
    """Deleting optional YAML controls must retain safe defaults through the real project preflight."""
    from skyulf.integrations.databricks.projects.workflow_config import preview_workflow_config
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path = _project(tmp_path)
    base = {**workflow_config, "engine": "pandas"}
    for name in ("score_handoff", "model_change_mode"):
        base.pop(name)
    if delivery == "endpoint":
        base.pop("prediction_table")
        base.pop("score_model_selection")
    path.write_text(json.dumps(base), encoding="utf-8")
    _inference(path, f"prediction_delivery: {delivery}\n")
    config = read_workflow_config(path)
    preview = preview_workflow_config(config, action="train")
    assert config["score_handoff"] == "disabled"
    assert config["model_change_mode"] == "incremental_append"
    assert config["score_model_selection"] == "champion"
    assert ("endpoint only" in preview) == (delivery == "endpoint")


@pytest.mark.parametrize("delivery", ["batch", "both"])
def test_yaml_batch_requires_explicit_model_selection(tmp_path, workflow_config, delivery):
    """Deleting model selection must not silently change which model batch scoring uses."""
    from skyulf.integrations.databricks.projects.workflow_config import validate_workflow_config
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path = _project(tmp_path)
    base = {**workflow_config, "engine": "pandas"}
    base.pop("score_model_selection")
    path.write_text(json.dumps(base), encoding="utf-8")
    _inference(path, f"prediction_delivery: {delivery}\n")
    with pytest.raises(ValueError, match="score_model_selection"):
        validate_workflow_config(read_workflow_config(path), action="train")


@pytest.mark.parametrize(
    "legacy,selection", [("pinned_version", "pinned_version"), ("auto_champion", "champion")]
)
def test_legacy_json_read_preserves_explicit_migration(
    tmp_path, workflow_config, legacy, selection
):
    """Reading old JSON must not inject new policy fields that break its supported migration."""
    from skyulf.integrations.databricks.projects.workflow_config import migrate_workflow_config
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path = _project(tmp_path)
    base = dict(workflow_config)
    for key in ("score_model_selection", "promotion_policy", "score_handoff", "model_change_mode"):
        base.pop(key)
    base["model_selection_mode"] = legacy
    path.write_text(json.dumps(base), encoding="utf-8")
    config = read_workflow_config(path)
    assert config == base
    migrated = migrate_workflow_config(config, task="regression", score_handoff="disabled")
    assert migrated["score_model_selection"] == selection
    assert "model_selection_mode" not in migrated


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


@pytest.mark.parametrize(
    "public,internal",
    [
        ("training_table_version", "training_version"),
        ("training_label_table_version", "training_label_version"),
    ],
)
@pytest.mark.parametrize("version", [None, 0, 17])
def test_training_table_versions_translate_defaults_and_overrides(
    tmp_path, public, internal, version
):
    """Public table versions must preserve null and zero while reaching existing snapshot contracts."""
    from skyulf.integrations.databricks.projects.yaml_config import read_training_config

    path = _project(tmp_path)
    path.with_name("training.yml").write_text(
        json.dumps(
            {
                "version": 1,
                "defaults": {public: version},
                "models": {"main": {"model": {"type": "linear_regression"}, public: 3}},
            }
        ),
        encoding="utf-8",
    )
    training = read_training_config(path.parent)
    assert training is not None
    assert training["defaults"][internal] == version
    assert training["models"]["main"][internal] == 3
    assert public not in training["defaults"] and public not in training["models"]["main"]


@pytest.mark.parametrize(
    "key",
    [
        "preprocessing_probe",
        "admit_probed_custom_steps",
        "training_version",
        "training_label_version",
    ],
)
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
