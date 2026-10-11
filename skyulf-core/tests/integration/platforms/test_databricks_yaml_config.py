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
    from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline
    from skyulf.integrations.databricks.projects.project import load_project_workflow
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config
    from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow

    path, features = _project(tmp_path)
    yaml_path = path.with_name("training.yml")
    yaml_path.write_text(
        "version: 1\ndefaults:\n  weight_column: null\nmodels:\n"
        "  main:\n    model: {type: linear_regression, params: {}}\n",
        encoding="utf-8",
    )
    config = load_project_workflow(read_workflow_config(path), features)
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    artifact = fit_workflow(
        config["pipeline"],
        SplitDataset(train=frame, test=frame[:0]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=10,
        max_bytes=10000,
    )
    yaml_path.write_text("invalid: edited", encoding="utf-8")
    loaded = load_pipeline(tmp_path / "artifact")
    assert config["pipeline"]["modeling"] == {"type": "linear_regression", "params": {}}
    assert artifact.manifest.project_source_sha256
    assert predict_pipeline(pd.DataFrame({"x": [5.0]}), loaded)[
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


def test_direct_yaml_keeps_target_bindings_when_loading_model(tmp_path):
    """Loading declarative models must not replace resolved UC names with raw placeholders."""
    from skyulf.integrations.databricks.lifecycle.workflow import resolve_target_config
    from skyulf.integrations.databricks.projects.project import load_project_workflow
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    legacy, features = _project(tmp_path)
    legacy.unlink()
    path = legacy.with_name("training.yml")
    path.write_text(
        "version: 1\ndefaults:\n  training_table: '{catalog}.{input_schema}.source'\n"
        "  model_name: '{catalog}.{metadata_schema}.model{resource_suffix}'\n"
        "  random_state: 42\nmodels:\n  main:\n"
        "    model: {type: ridge_regression}\n    random_state: 9\n",
        encoding="utf-8",
    )
    path.with_name("inference.yml").write_text(
        "version: 1\nscore_source_table: '{catalog}.{input_schema}.source'\n"
        "prediction_table: '{catalog}.{output_schema}.predictions{resource_suffix}'\n",
        encoding="utf-8",
    )
    config = read_workflow_config(path)
    config = resolve_target_config(
        config,
        {
            "catalog": "workspace",
            "input_schema": "test",
            "output_schema": "test",
            "metadata_schema": "test",
            "resource_suffix": "",
        },
    )
    loaded = load_project_workflow(config, features)
    assert loaded["training_table"] == "workspace.test.source"
    assert loaded["random_state"] == 9


def test_direct_yaml_rejects_pipeline_and_model_double_ownership(tmp_path):
    """Explainability must have one owner even within one YAML document."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path = tmp_path / "training.yml"
    path.write_text(
        "version: 1\nmodels: {main: {model: {type: ridge_regression}, explainability: null}}\n"
        "pipeline: {explainability: null}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="defined in both"):
        read_workflow_config(path)


def test_yaml_flow_values_preserve_json_unicode_and_exponents(tmp_path):
    """Initializer JSON parameters embedded in YAML retain their original scalar values."""
    from skyulf.integrations.databricks.projects.yaml_config import read_yaml_mapping

    expected = {"tiny": 1e-8, "large": -2e20, "unicode": "\U0001f600"}
    path = tmp_path / "values.yml"
    path.write_text(json.dumps(expected), encoding="utf-8")
    assert read_yaml_mapping(path) == expected


def test_yaml_branch_base_does_not_leak_native_lookup():
    """A representative branch's feature lookup must not become a sibling's source contract."""
    from skyulf.integrations.databricks.projects.yaml_models import branch_base, training_branches

    lookup = {"lookups": [{"table_name": "workspace.test.features"}]}
    document = {
        "defaults": {"model": {"type": "ridge_regression"}, "engine": "pandas"},
        "models": {"left": {"feature_lookup": lookup}, "right": {}},
    }
    base = branch_base({"engine": "pandas", "feature_lookup": lookup}, document)
    branches, _ = training_branches(document)
    assert branches["left"]["workflow"]["feature_lookup"] == lookup
    assert "feature_lookup" not in {**base, **branches["right"]["workflow"]}


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_shared_yaml_pipeline_explainability_reaches_each_model(tmp_path, layout):
    """Accepted shared pipeline settings must survive independent candidate and branch adapters."""
    from skyulf.integrations.databricks.projects.yaml_config import (
        read_training_config,
        read_workflow_config,
    )
    from skyulf.integrations.databricks.projects.yaml_models import (
        competition_candidates,
        single_model,
        training_branches,
    )

    path = tmp_path / "training.yml"
    path.write_text(
        f"version: 1\ndefaults: {{training_layout: {layout}}}\n"
        "models: {main: {model: {type: ridge_regression}}}\n"
        "pipeline: {explainability: {method: shap, max_samples: 4}}\n",
        encoding="utf-8",
    )
    document = read_training_config(tmp_path)
    assert document is not None
    if layout == "single_model":
        pipeline = single_model(document, read_workflow_config(path))["pipeline"]
    elif layout == "model_competition":
        pipeline = competition_candidates(document)[0]["main"]
    else:
        pipeline = training_branches(document)[0]["main"]["workflow"]["pipeline"]
    assert pipeline["explainability"] == {"method": "shap", "max_samples": 4}
