"""Single-source YAML must resolve without changing saved training contracts."""

import json

import pandas as pd
import pytest


def _project(tmp_path, config=None):
    """Make a minimal organized project without executing editable model code."""
    (tmp_path / "config").mkdir()
    features = tmp_path / "src/features"
    features.mkdir(parents=True)
    (features / "__init__.py").write_text(
        "def build_preprocessing(recipe='default'):\n    return []\n", encoding="utf-8"
    )
    (tmp_path / "src/modeling").mkdir()
    workflow = config or {
        "training_layout": "single_model",
        "pipeline": {"preprocessing": [], "modeling": {}},
    }
    path = tmp_path / "config/workflow.json"
    path.write_text(json.dumps(workflow), encoding="utf-8")
    return path, features


def test_yaml_single_model_freezes_model_and_custom_source(tmp_path):
    """Changing YAML after fitting must never change an already saved predictor."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
    from skyulf.integrations.databricks.projects.project import load_project_workflow
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config
    from skyulf.integrations.databricks.scoring.batch.local_batch import fit_local_workflow

    path, features = _project(tmp_path)
    yaml_path = path.with_name("training.yml")
    yaml_path.write_text(
        "version: 1\ndefaults:\n  weight_column: null\nmodels:\n"
        "  main:\n    model: {type: linear_regression, params: {}}\n",
        encoding="utf-8",
    )
    config = load_project_workflow(read_workflow_config(path), features)
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    artifact = fit_local_workflow(
        config["pipeline"],
        SplitDataset(train=frame, test=frame[:0]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=10,
        max_bytes=10000,
    )
    yaml_path.write_text("invalid: edited", encoding="utf-8")
    loaded = load_local_pipeline(tmp_path / "artifact")
    assert config["pipeline"]["modeling"] == {"type": "linear_regression", "params": {}}
    assert artifact.manifest.project_source_sha256
    assert predict_local_pipeline(pd.DataFrame({"x": [5.0]}), loaded)[
        "prediction"
    ].tolist() == pytest.approx([11.0])


@pytest.mark.parametrize(
    "content,match",
    [
        ("version: 1\nmodels: {}\nmodels: {}\n", "Duplicate"),
        ("version: 1\nmodels: {main: {model: {type: ridge_regression}, typo: 2}}", "Unknown"),
        ("version: 1\ndefaults: {random_state: 1, random_state: 2}\nmodels: {}", "Duplicate"),
        ("version: 1\nmodels: {main: {model: {type: ridge_regression, typo: 2}}}", "Unknown"),
        ("version: true\nmodels: {}", "version"),
        (
            "version: 1\nmodels: {main: {model: {type: ridge_regression, params: {alpha: .nan}}}}",
            "finite",
        ),
        (
            "version: 1\nmodels: {main: {model: {type: ridge_regression}, tuning: {typo: 2}}}",
            "Unknown",
        ),
    ],
)
def test_yaml_rejects_ambiguous_or_unknown_settings(tmp_path, content, match):
    """Typos and duplicated declarations must fail before model code or cloud I/O."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path, _ = _project(tmp_path)
    path.with_name("training.yml").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_workflow_config(path)


def test_yaml_rejects_json_and_python_double_ownership(tmp_path):
    """Equal values still conflict so there is exactly one editable owner."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path, _ = _project(tmp_path, {"random_state": 42, "pipeline": {"modeling": {}}})
    yaml_path = path.with_name("training.yml")
    yaml_path.write_text(
        "version: 1\ndefaults: {random_state: 42}\nmodels:\n"
        "  main: {model: {type: ridge_regression}}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="defined in both"):
        read_workflow_config(path)
    path.write_text('{"pipeline": {"modeling": {}}}', encoding="utf-8")
    (tmp_path / "src/modeling/single_model.py").write_text(
        "raise RuntimeError('must not execute')", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="defined in both"):
        read_workflow_config(path)


def test_yaml_defaults_and_per_model_overrides_are_independent(tmp_path):
    """A branch override must not mutate its shared defaults or a sibling model."""
    from skyulf.integrations.databricks.projects.yaml_config import read_training_config
    from skyulf.integrations.databricks.projects.yaml_models import training_branches

    path, _ = _project(tmp_path)
    path.with_name("training.yml").write_text(
        "version: 1\ndefaults:\n  task: regression\n  weight_column: null\n"
        "  model: {type: ridge_regression, params: {alpha: 1.0}}\n"
        "models:\n  revenue: {target_column: revenue}\n"
        "  cost: {target_column: cost, model: {type: linear_regression}}\n",
        encoding="utf-8",
    )
    declarations = read_training_config(path.parent)
    assert declarations is not None
    branches, _ = training_branches(declarations)
    assert branches["revenue"]["workflow"]["pipeline"]["modeling"]["type"] == "ridge_regression"
    assert branches["cost"]["workflow"]["pipeline"]["modeling"]["type"] == "linear_regression"
    assert branches["cost"]["workflow"]["task"] == "regression"
    branches["revenue"]["workflow"]["pipeline"]["modeling"]["params"]["alpha"] = 99
    assert declarations["defaults"]["model"]["params"]["alpha"] == 1.0


def test_optional_inference_yaml_owns_only_its_fields(tmp_path):
    """Scoring can use a YAML policy without changing legacy Python training."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path, _ = _project(tmp_path)
    path.with_name("inference.yml").write_text(
        "version: 1\ninference_mode: spark\nspark_udf_prediction_batch_rows: 1000\n",
        encoding="utf-8",
    )
    config = read_workflow_config(path)
    assert config["inference_mode"] == "spark"
    assert config["spark_udf_prediction_batch_rows"] == 1000


def test_yaml_competition_resolves_candidates(tmp_path, workflow_config):
    """Declarative candidates must enter the same CV and source capture path as Python."""
    from skyulf.integrations.databricks.projects.project import load_project_workflow
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    workflow_config.update(
        training_layout="model_competition",
        cv_enabled=True,
        cv_folds=2,
        cv_type="k_fold",
        cv_shuffle=True,
        cv_random_state=42,
    )
    workflow_config["pipeline"]["modeling"] = {}
    path, features = _project(tmp_path, workflow_config)
    path.with_name("training.yml").write_text(
        "version: 1\nmodels:\n  ridge: {model: {type: ridge_regression}}\n"
        "  linear: {model: {type: linear_regression}}\n",
        encoding="utf-8",
    )
    result = load_project_workflow(read_workflow_config(path), features)
    assert set(result["competition"]["candidates"]) == {"ridge", "linear"}
    assert result["competition"]["candidates_source"].startswith("YAML_DECLARATIONS =")


def test_yaml_branches_and_model_set_use_shared_notebook_boundary(tmp_path, workflow_config):
    """Notebook branches and destinations must consume YAML while freezing Python features."""
    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        load_training_branch_configs,
    )
    from skyulf.integrations.databricks.model_sets.model_set_project import load_project_model_set

    workflow_config.update(training_layout="multi_target", score_handoff="disabled")
    workflow_config.pop("target_column")
    workflow_config["pipeline"]["modeling"] = {}
    path, _ = _project(tmp_path, workflow_config)
    path.with_name("training.yml").write_text(
        "version: 1\nmodels:\n  revenue:\n    model: {type: ridge_regression}\n"
        "    target_column: revenue\n  cost:\n    model: {type: linear_regression}\n"
        "    target_column: cost\n",
        encoding="utf-8",
    )
    path.with_name("inference.yml").write_text(
        "version: 1\nmodel_set:\n  model_name: '{catalog}.{metadata_schema}.financial'\n"
        "  prediction_table: '{catalog}.{output_schema}.financial_scores'\n"
        "  composition_config: {outputs: []}\n",
        encoding="utf-8",
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
    settings = load_project_model_set(values)
    assert settings is not None
    assert branches["cost"]["target_column"] == "cost"
    assert branches["cost"]["pipeline"]["project_python_source"]
    assert settings["model_name"] == "workspace.test.financial"


def test_migration_preserves_single_model_and_keeps_original_sources(tmp_path, workflow_config):
    """Migration must transfer ownership without losing the original files or custom features."""
    from skyulf.integrations.databricks.projects.project import load_project_workflow
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config
    from skyulf.integrations.databricks.projects.yaml_migration import migrate_project_yaml

    workflow_config["pipeline"]["modeling"] = {}
    path, features = _project(tmp_path, workflow_config)
    source = "from copy import deepcopy\nMODELING = {'type': 'ridge_regression', 'params': {}}\nWEIGHT_COLUMN = None\nDECISION_THRESHOLD = {'mode': 'off'}\ndef build_modeling():\n    return deepcopy(MODELING)\n"
    model_path = tmp_path / "src/modeling/single_model.py"
    model_path.write_text(source, encoding="utf-8")
    before = load_project_workflow(read_workflow_config(path), features)
    result = migrate_project_yaml(tmp_path)
    after = load_project_workflow(read_workflow_config(path), features)
    assert result["status"] == "migrated"
    assert before["pipeline"] == after["pipeline"]
    assert not model_path.exists()
    assert (tmp_path / ".skyulf-yaml-backup/src/modeling/single_model.py").read_text() == source
    assert features.is_dir()


def test_migration_rejects_custom_model_code_without_writes(tmp_path):
    """Custom modeling functions cannot disappear during automatic static migration."""
    from skyulf.integrations.databricks.projects.yaml_migration import migrate_project_yaml

    path, _ = _project(tmp_path)
    source = "MODELING = {'type': 'ridge_regression'}\ndef custom():\n    return 3\ndef build_modeling():\n    return MODELING\n"
    model_path = tmp_path / "src/modeling/single_model.py"
    model_path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="custom|generated"):
        migrate_project_yaml(tmp_path)
    assert model_path.read_text() == source
    assert not path.with_name("training.yml").exists()


@pytest.mark.parametrize("key,value", [("random_state", 99), ("target_column", "other")])
def test_model_overrides_cannot_hide_json_ownership(tmp_path, key, value):
    """Only YAML defaults may be overridden, keeping legacy JSON out of precedence games."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path, _ = _project(tmp_path, {key: value, "pipeline": {"modeling": {}}})
    path.with_name("training.yml").write_text(
        "version: 1\nmodels:\n  main:\n    model: {type: ridge_regression}\n"
        f"    {key}: {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="defined in both"):
        read_workflow_config(path)


def test_yaml_nesting_is_bounded_before_parsing(tmp_path):
    """A small deeply nested document must raise a configuration error, not recurse indefinitely."""
    from skyulf.integrations.databricks.projects.yaml_config import read_yaml_mapping

    path = tmp_path / "nested.yml"
    path.write_text("value: " + "[" * 600 + "0" + "]" * 600, encoding="utf-8")
    with pytest.raises(ValueError, match="nesting"):
        read_yaml_mapping(path)


@pytest.mark.parametrize(
    "factory",
    [
        "def build_modeling(x=print('side effect')):\n    return deepcopy(MODELING)\n",
        "def build_modeling() -> print('side effect'):\n    return deepcopy(MODELING)\n",
    ],
)
def test_migration_refuses_custom_factory_signatures(tmp_path, factory):
    """Default arguments and annotations are executable custom source, even with a simple body."""
    from skyulf.integrations.databricks.projects.yaml_migration import migrate_project_yaml

    _project(tmp_path)
    source = "from copy import deepcopy\nMODELING = {'type': 'ridge_regression'}\n" + factory
    path = tmp_path / "src/modeling/single_model.py"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="custom"):
        migrate_project_yaml(tmp_path)
    assert path.read_text() == source


def test_static_smoke_validates_yaml_models_without_running_python(tmp_path, workflow_config):
    """Declarative model typos must fail the offline check before a deployment."""
    from skyulf.integrations.databricks.projects.project_checks import check_project

    workflow_config["pipeline"]["modeling"] = {}
    path, _ = _project(tmp_path, workflow_config)
    path.with_name("training.yml").write_text(
        "version: 1\nmodels: {main: {model: {type: definitely_nonexistent}}}\n",
        encoding="utf-8",
    )
    bindings = {
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
    }
    with pytest.raises(ValueError, match="definitely_nonexistent|Unknown"):
        check_project(tmp_path, bindings)


def test_migration_restores_originals_after_publication_failure(tmp_path, monkeypatch):
    """A partial filesystem write must leave the original project executable and intact."""
    from pathlib import Path

    from skyulf.integrations.databricks.projects.yaml_migration import migrate_project_yaml

    path, _ = _project(tmp_path)
    original = path.read_bytes()
    model_path = tmp_path / "src/modeling/single_model.py"
    source = "from copy import deepcopy\nMODELING = {'type': 'ridge_regression'}\ndef build_modeling():\n    return deepcopy(MODELING)\n"
    model_path.write_text(source, encoding="utf-8")
    unlink = Path.unlink

    def fail_model_removal(self, *args, **kwargs):
        """Simulate a locked declaration file at the final ownership-transfer step."""
        if self == model_path:
            raise OSError("locked model file")
        return unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_model_removal)
    with pytest.raises(OSError, match="locked"):
        migrate_project_yaml(tmp_path)
    assert path.read_bytes() == original
    assert model_path.read_text() == source
    assert not path.with_name("training.yml").exists()
    assert not path.with_name("inference.yml").exists()


def test_migration_rejects_duplicate_python_literals(tmp_path):
    """Static migration must not silently retain only the last duplicate dictionary key."""
    from skyulf.integrations.databricks.projects.yaml_migration import migrate_project_yaml

    path, _ = _project(tmp_path)
    (tmp_path / "src/modeling/single_model.py").write_text(
        "from copy import deepcopy\nMODELING = {'type':'ridge_regression','params':{'alpha':1,'alpha':2}}\n"
        "def build_modeling():\n    return deepcopy(MODELING)\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        migrate_project_yaml(tmp_path)
    assert not path.with_name("training.yml").exists()


def test_migration_rejects_model_set_decorators(tmp_path):
    """Custom model-set decorators cannot be discarded when moving literal settings."""
    from skyulf.integrations.databricks.projects.yaml_migration import migrate_project_yaml

    _project(tmp_path)
    (tmp_path / "src/modeling/single_model.py").write_text(
        "from copy import deepcopy\nMODELING = {'type':'ridge_regression'}\n"
        "def build_modeling():\n    return deepcopy(MODELING)\n",
        encoding="utf-8",
    )
    (tmp_path / "src/modeling/model_set.py").write_text(
        "@custom\ndef build_model_set():\n    enabled = True\n"
        "    if not enabled:\n        return None\n"
        "    return {'model_name':'workspace.test.set','prediction_table':'workspace.test.scores'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generated"):
        migrate_project_yaml(tmp_path)


def test_migration_rejects_duplicate_model_set_literals(tmp_path):
    """Model-set destinations must not silently lose an earlier declaration during migration."""
    from skyulf.integrations.databricks.projects.yaml_migration import migrate_project_yaml

    path, _ = _project(tmp_path)
    (tmp_path / "src/modeling/single_model.py").write_text(
        "from copy import deepcopy\nMODELING = {'type':'ridge_regression'}\n"
        "def build_modeling():\n    return deepcopy(MODELING)\n",
        encoding="utf-8",
    )
    (tmp_path / "src/modeling/model_set.py").write_text(
        "def build_model_set():\n    enabled = True\n"
        "    if not enabled:\n        return None\n"
        "    return {'model_name':'workspace.test.first','model_name':'workspace.test.second',"
        "'prediction_table':'workspace.test.scores'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        migrate_project_yaml(tmp_path)
    assert not path.with_name("inference.yml").exists()


def test_yaml_single_model_rejects_ignored_feature_path(tmp_path):
    """Unsupported settings must fail instead of giving a false impression of taking effect."""
    from skyulf.integrations.databricks.projects.project import load_project_workflow
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path, features = _project(tmp_path)
    path.with_name("training.yml").write_text(
        "version: 1\nmodels:\n  main:\n    model: {type: ridge_regression}\n"
        "    features_path: ../other_features\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="features_path"):
        load_project_workflow(read_workflow_config(path), features)


def test_migration_omits_redundant_defaults_without_changing_effective_settings(
    tmp_path, workflow_config
):
    """Compact YAML must retain the exact CV, date, sampling and optional-report behavior."""
    from skyulf.integrations.databricks.data.training.training_dates import training_date_spec
    from skyulf.integrations.databricks.lifecycle.local_workflow import training_spec
    from skyulf.integrations.databricks.observability.charts.evaluation_chart_data import (
        chart_settings,
    )
    from skyulf.integrations.databricks.projects.yaml_config import (
        read_training_config,
        read_workflow_config,
    )
    from skyulf.integrations.databricks.projects.yaml_migration import migrate_project_yaml
    from skyulf.integrations.databricks.training.tuning.local_cv import LocalCVSpec

    workflow_config["pipeline"]["modeling"] = {}
    workflow_config.update(
        cv_enabled=False,
        cv_folds=5,
        cv_type="k_fold",
        cv_shuffle=True,
        cv_random_state=42,
        training_sample_rows=None,
        training_sample_seed=42,
        risk_category=None,
        champion_version=None,
        evaluation_charts={
            "enabled": False,
            "max_rows": 2000,
            "max_features": 20,
            "max_classes": 20,
        },
    )
    path, _ = _project(tmp_path, workflow_config)
    (tmp_path / "src/modeling/single_model.py").write_text(
        "from copy import deepcopy\nMODELING = {'type':'ridge_regression'}\n"
        "def build_modeling():\n    return deepcopy(MODELING)\n",
        encoding="utf-8",
    )
    migrate_project_yaml(tmp_path)
    after = read_workflow_config(path)
    document = read_training_config(path.parent)
    assert document is not None
    assert "cv_folds" not in document["defaults"]
    assert "risk_category" not in document["defaults"]
    assert document["defaults"]["evaluation_charts"] == {"enabled": False}
    assert LocalCVSpec.from_workflow(workflow_config) == LocalCVSpec.from_workflow(after)
    assert training_date_spec(workflow_config.get("event_time_parsing", {})) == training_date_spec(
        after.get("event_time_parsing", {})
    )
    assert training_spec(workflow_config) == training_spec(after)
    assert chart_settings(workflow_config["evaluation_charts"]) == chart_settings(
        after["evaluation_charts"]
    )
