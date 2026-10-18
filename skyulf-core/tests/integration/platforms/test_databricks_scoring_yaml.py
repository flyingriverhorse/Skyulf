"""Declarative scoring must use the existing saved policies and admission boundaries."""

import json
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

yaml = pytest.importorskip("yaml")

from skyulf.data.dataset import SplitDataset
from skyulf.inference._manifest import ColumnSpec
from skyulf.inference.fitted_pipeline import load_pipeline
from skyulf.inference.model_set_scoring import (
    compose_model_set_outputs,
    validate_model_set_composition,
)
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.pipeline_scoring import score_pipeline
from skyulf.inference.project_code import load_project_module, project_source_digest
from skyulf.inference.project_package import discard_project_package
from skyulf.integrations.databricks.model_sets.model_set_project import capture_set_rules
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.projects.yaml_recipes import feature_project_source
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.shared.preprocessing_checks import seal_partition_probe


@pytest.fixture
def scoring_project(tmp_path):
    """Keep the feature package separate from its editable scoring declarations."""
    features = tmp_path / "src/features"
    features.mkdir(parents=True)
    (features / "__init__.py").write_text(
        "def build_preprocessing(recipe='default'):\n    return []\n", encoding="utf-8"
    )
    (features / "scoring.py").write_text(
        "import pandas as pd\n"
        "def exclude(frame, params):\n"
        "    return pd.Series(None, index=frame.index, dtype='string').mask(\n"
        "        frame['x'] > params['limit'], 'too_large')\n"
        "def doubled(frame, predictions, params):\n"
        "    return (predictions['prediction'] * params['factor']).to_frame('twice')\n"
        "def profit(frame, predictions, params):\n"
        "    return (predictions['sales__prediction'] - predictions['cost__prediction'])"
        ".to_frame('profit')\n",
        encoding="utf-8",
    )
    (tmp_path / "config").mkdir()
    return features


def _write(features, document):
    """Write the public YAML envelope without bypassing the project's safe reader."""
    path = features.parents[1] / "config/scoring.yml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _workflow(features, engine="pandas"):
    """Resolve the real training entrypoint with enough metadata for pre-split replay."""
    return load_project_workflow(
        {
            "engine": engine,
            "input_columns": ["x"],
            "target_column": "target",
            "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        },
        features,
    )


def _rules():
    """Pin first-reason ordering and exact threshold boundaries with independent literals."""
    return {
        "version": 1,
        "mode": "custom",
        "eligibility": [
            {"name": "observed", "check": "required_values", "params": {"columns": ["x"]}},
            {
                "name": "range",
                "check": "value_range",
                "params": {"column": "x", "minimum": 0, "maximum": 20},
            },
        ],
        "outputs": [
            {
                "name": "band",
                "rule": "prediction_band",
                "params": {"thresholds": [5, 9], "labels": ["low", "medium", "high"]},
                "columns": [{"name": "band", "dtype": "string"}],
            }
        ],
    }


def _fit(features, tmp_path, engine="pandas", *, probe=False):
    """Fit and save a real model so rules run through persistence and native engines."""
    pipeline = _workflow(features, engine)["pipeline"]
    pipeline["admit_probed_custom_steps"] = probe
    rows = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    native = pl.from_pandas(rows) if engine == "polars" else rows
    artifact = fit_workflow(
        pipeline,
        SplitDataset(train=native, test=native[:0]),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=100,
        max_bytes=100000,
    )
    if probe:
        holdout = pd.DataFrame({"x": [float(i) / 2 for i in range(32)]})
        return seal_partition_probe(artifact, holdout, tmp_path / "artifact", enabled=True)
    return artifact


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_yaml_builtins_fit_save_reload_and_freeze(scoring_project, tmp_path, engine):
    """Saved rule behavior must survive edits to YAML, Python and package module caches."""
    path = _write(scoring_project, _rules())
    artifact = _fit(scoring_project, tmp_path, engine)
    source = artifact.pipeline.config["project_python_source"]
    path.write_text("invalid: current YAML", encoding="utf-8")
    (scoring_project / "scoring.py").write_text("raise RuntimeError('edited')", encoding="utf-8")
    discard_project_package(load_project_module(source).__name__)
    loaded = load_pipeline(tmp_path / "artifact")
    raw = pd.DataFrame({"x": [None, -1.0, 1.0, 2.0, 4.0, 21.0]}, index=[6, 2, 9, 1, 4, 3])
    native = pl.from_pandas(raw) if engine == "polars" else raw
    scored = score_pipeline(native, loaded)
    assert scored.scoring_status.tolist() == [
        "excluded",
        "excluded",
        "predicted",
        "predicted",
        "predicted",
        "excluded",
    ]
    assert scored.exclusion_reason.fillna("").tolist() == [
        "missing:x",
        "outside_range:x",
        "",
        "",
        "",
        "outside_range:x",
    ]
    assert scored.band.fillna("").tolist() == ["", "", "low", "medium", "high", ""]
    assert scored.prediction.dropna().tolist() == pytest.approx([3.0, 5.0, 9.0])
    assert score_pipeline(native[:0], loaded).empty
    assert score_pipeline(native[:2], loaded).prediction.isna().all()


def test_yaml_custom_callbacks_are_captured(scoring_project, tmp_path):
    """YAML-selected project functions retain normal eligibility and declared-output checks."""
    _write(
        scoring_project,
        {
            "version": 1,
            "mode": "custom",
            "eligibility": [{"name": "limit", "custom": "scoring.exclude", "params": {"limit": 4}}],
            "outputs": [
                {
                    "name": "twice",
                    "custom": "scoring.doubled",
                    "params": {"factor": 2},
                    "columns": [{"name": "twice", "dtype": "float64"}],
                }
            ],
        },
    )
    artifact = _fit(scoring_project, tmp_path)
    source = artifact.pipeline.config["project_python_source"]
    (scoring_project / "scoring.py").write_text("raise RuntimeError('edited')", encoding="utf-8")
    discard_project_package(load_project_module(source).__name__)
    scored = score_pipeline(pd.DataFrame({"x": [2.0, 5.0]}), load_pipeline(tmp_path / "artifact"))
    assert scored.twice.iloc[0] == pytest.approx(10.0)
    assert scored.exclusion_reason.iloc[1] == "too_large"


@pytest.mark.parametrize("mode", ["pre_split", "combined", "custom"])
def test_yaml_modes_preserve_pre_split_order(scoring_project, tmp_path, mode):
    """Pre-split selection runs first only when requested and keeps its original reason."""
    (scoring_project.parents[1] / "config/pre_split.yml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "recipes": {
                    "default": [
                        {
                            "name": "positive",
                            "transformer": "ManualBounds",
                            "params": {"bounds": {"x": {"lower": 0, "upper": 10}}},
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    document = {"version": 1, "mode": mode}
    if mode != "pre_split":
        document["eligibility"] = [
            {"name": "limit", "custom": "scoring.exclude", "params": {"limit": 4}}
        ]
    _write(scoring_project, document)
    config = _workflow(scoring_project)["pipeline"]["project_scoring"]
    assert ("pre_split" in config) is (mode != "custom")
    assert len(config["eligibility"]) == (0 if mode == "pre_split" else 1)
    result = score_pipeline(
        pd.DataFrame({"x": [-1.0, 2.0, 5.0, 11.0]}), _fit(scoring_project, tmp_path)
    )
    expected = {
        "pre_split": ["pre_split:positive", "", "", "pre_split:positive"],
        "combined": ["pre_split:positive", "", "too_large", "pre_split:positive"],
        "custom": ["", "", "too_large", "too_large"],
    }
    assert result.exclusion_reason.fillna("").tolist() == expected[mode]


@pytest.mark.parametrize(
    "update, message",
    [
        ({"version": True}, "version"),
        ({"version": 2}, "version"),
        ({"mode": "typo"}, "mode"),
        ({"skip_target_pre_split_steps": "false"}, "boolean"),
        ({"skip_target_pre_split_steps": True}, "custom"),
        ({"mode": "pre_split"}, "pre_split"),
        ({"eligibility": {}}, "list"),
        (
            {
                "outputs": [
                    {
                        "name": "bad",
                        "custom": "os.system",
                        "columns": [{"name": "bad", "dtype": "string"}],
                    }
                ]
            },
            "custom",
        ),
        ({"combined": "bad"}, "list"),
        ({"typo": 1}, "typo"),
    ],
)
def test_invalid_yaml_rejected_before_python(scoring_project, update, message):
    """Malformed ownership and declarations cannot execute editable project imports."""
    document = {**_rules(), **update}
    _write(scoring_project, document)
    (scoring_project / "__init__.py").write_text(
        "raise RuntimeError('must not execute')", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=message):
        _workflow(scoring_project)


@pytest.mark.parametrize("filename", ["__init__.py", "scoring.py", "other.py"])
@pytest.mark.parametrize("factory", ["build_model_rules", "build_scoring", "build_combined_rules"])
def test_python_yaml_ownership_is_exclusive(scoring_project, filename, factory):
    """A hidden or exported legacy builder must not be silently overridden by YAML."""
    _write(scoring_project, {"version": 1})
    (scoring_project / filename).write_text(
        f"def {factory}():\n    return {{}}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="both.*scoring"):
        _workflow(scoring_project)


def test_yaml_builtins_keep_partition_admission(scoring_project, tmp_path):
    """Saved builtin definitions must pass the existing lint and all dynamic rule probes."""
    _write(scoring_project, _rules())
    artifact = _fit(scoring_project, tmp_path, probe=True)
    admitted = require_partition_safe_pipeline(artifact)
    assert len(artifact.pipeline.config["partition_probe"]["rules"]) == 3
    assert any(item.action == "rules_probed" for item in admitted.steps)


@pytest.mark.parametrize("kind", ["weighted_sum", "weighted_mean", "custom"])
def test_yaml_combined_uses_existing_composition(scoring_project, kind):
    """Combined YAML freezes direct dependencies and runs through the existing evaluator."""
    rule = {
        "name": "profit",
        "columns": [{"name": "profit", "dtype": "float64"}],
        "required_components": ["sales", "cost"],
    }
    if kind == "custom":
        rule["custom"] = "scoring.profit"
    else:
        rule.update(
            operation=kind,
            params={"inputs": ["sales__prediction", "cost__prediction"], "weights": [2, 1]},
        )
    _write(scoring_project, {"version": 1, "combined": [rule]})
    values = {"config_path": str(scoring_project.parents[1] / "config/workflow.json")}
    settings, source = capture_set_rules(values, {"combined_rules_path": "../features"})
    components = [
        SimpleNamespace(
            branch=name, output_schema=(ColumnSpec(name="prediction", dtype="float64"),)
        )
        for name in ("sales", "cost")
    ]
    config = validate_model_set_composition(settings["composition_config"], source, components, ())
    predicted = pd.DataFrame(
        {
            "sales__prediction": [10.0, 10.0],
            "cost__prediction": [4.0, 4.0],
            "sales__scoring_status": ["predicted", "predicted"],
            "cost__scoring_status": ["predicted", "excluded"],
            "sales__exclusion_reason": [None, None],
            "cost__exclusion_reason": [None, "missing:x"],
        }
    )
    result = compose_model_set_outputs(pd.DataFrame({"x": [1, 2]}), predicted, config, source)
    assert result.profit.iloc[0] == pytest.approx(
        {"weighted_sum": 24, "weighted_mean": 8, "custom": 6}[kind]
    )
    assert pd.isna(result.profit.iloc[1])
    assert result.profit__exclusion_reason.iloc[1] == "cost: missing:x"


def test_yaml_changes_source_identity_and_builders_copy(scoring_project):
    """Mutable builder results cannot alter another model or evade source fingerprinting."""
    document = _rules()
    _write(scoring_project, document)
    first = feature_project_source(scoring_project)
    module = load_project_module(first)
    original = module.build_model_rules()
    edited = module.build_model_rules()
    edited["eligibility"][0]["params"]["columns"].append("wrong")
    assert module.build_model_rules() == original
    document["eligibility"][1]["params"]["maximum"] = 21
    _write(scoring_project, document)
    assert project_source_digest(first) != project_source_digest(
        feature_project_source(scoring_project)
    )


@pytest.mark.parametrize(
    "kind, rule",
    [
        ("eligibility", {"name": "a", "check": "unknown"}),
        ("eligibility", {"name": "a", "check": "required_values", "params": {"columns": []}}),
        (
            "eligibility",
            {"name": "a", "check": "required_values", "params": {"columns": ["x", "x"]}},
        ),
        ("eligibility", {"name": "a", "check": "required_values", "params": {"columns": [False]}}),
        ("eligibility", {"name": "a", "check": "value_range", "params": {"column": "x"}}),
        (
            "eligibility",
            {"name": "a", "check": "value_range", "params": {"column": "x", "minimum": True}},
        ),
        (
            "eligibility",
            {
                "name": "a",
                "check": "value_range",
                "params": {"column": "x", "minimum": 4, "maximum": 2},
            },
        ),
        (
            "eligibility",
            {
                "name": "a",
                "check": "value_range",
                "params": {"column": "x", "minimum": float("inf")},
            },
        ),
        ("eligibility", {"name": "a", "custom": "scoring.exclude", "check": "value_range"}),
        ("eligibility", {"name": "a", "custom": "../scoring.exclude"}),
        ("eligibility", {"name": "a", "custom": "groups.scoring.exclude"}),
        ("eligibility", {"name": "a", "custom": "scoring.exclude", "params": []}),
        ("eligibility", {"name": "a", "custom": "scoring.exclude", "version": 1}),
        (
            "outputs",
            {
                "name": "a",
                "rule": "prediction_band",
                "columns": [{"name": "band", "dtype": "int64"}],
                "params": {"thresholds": [1], "labels": ["a", "b"]},
            },
        ),
        (
            "outputs",
            {
                "name": "a",
                "rule": "prediction_band",
                "columns": [{"name": "band", "dtype": "string"}],
                "params": {"thresholds": [1, 1], "labels": ["a", "b", "c"]},
            },
        ),
        (
            "outputs",
            {
                "name": "a",
                "rule": "prediction_band",
                "columns": [{"name": "band", "dtype": "string"}],
                "params": {"thresholds": [True], "labels": ["a", "b"]},
            },
        ),
        (
            "outputs",
            {
                "name": "a",
                "rule": "prediction_band",
                "columns": [{"name": "band", "dtype": "string"}],
                "params": {"thresholds": [1], "labels": ["a"]},
            },
        ),
        (
            "outputs",
            {
                "name": "a",
                "rule": "prediction_band",
                "columns": [{"name": "band", "dtype": "string"}],
                "params": {"thresholds": [], "labels": [3]},
            },
        ),
        (
            "outputs",
            {
                "name": "a",
                "custom": "scoring.doubled",
                "columns": [{"name": "prediction", "dtype": "float64"}],
            },
        ),
        (
            "outputs",
            {"name": "a", "custom": "scoring.doubled", "columns": [{"name": "extra", "dtype": []}]},
        ),
        (
            "combined",
            {
                "name": "a",
                "custom": "scoring.profit",
                "columns": [{"name": "profit", "dtype": "float64"}],
                "required_components": [],
            },
        ),
        (
            "combined",
            {
                "name": "a",
                "operation": "weighted_mean",
                "params": {"inputs": ["sales__prediction"], "weights": [0]},
                "columns": [{"name": "profit", "dtype": "float64"}],
                "required_components": ["sales"],
            },
        ),
    ],
)
def test_invalid_rule_contracts_are_rejected_before_import(scoring_project, kind, rule):
    """Unknown, nonfinite, conflicting or malformed rule inputs must fail before user code."""
    _write(scoring_project, {"version": 1, "mode": "custom", kind: [rule]})
    (scoring_project / "__init__.py").write_text(
        "raise RuntimeError('must not execute')", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        _workflow(scoring_project)


@pytest.mark.parametrize(
    "source",
    [
        "version: 1\nversion: 1\n",
        "version: 1\noutputs: &rules []\neligibility: *rules\n",
        "version: 1\nmode: !!python/name:os.system ''\n",
        "version: 1\nextra: 2026-01-01\n",
        "version: 1\n#" + "x" * 65536,
    ],
    ids=["duplicate", "alias", "tag", "date", "oversize"],
)
def test_yaml_reader_controls_apply_to_scoring(scoring_project, source):
    """Duplicate keys, aliases, tags, non-JSON values and oversized files remain forbidden."""
    path = _write(scoring_project, {"version": 1})
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        feature_project_source(scoring_project)


@pytest.mark.parametrize(
    "binding",
    [
        "build_scoring, unused = None, None",
        "def _skyulf_yaml_required_values(frame, params):\n    return None",
        "if True:\n    def build_model_rules():\n        return {}",
    ],
)
def test_owner_bindings_fail_before_import_side_effects(scoring_project, binding):
    """Alternate binding syntax and generated-name collisions cannot evade ownership."""
    _write(scoring_project, _rules())
    (scoring_project / "__init__.py").write_text(
        "raise RuntimeError('must not execute')\n" + binding + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="both.*scoring"):
        feature_project_source(scoring_project)


def test_legacy_python_scoring_without_yaml_is_unchanged(scoring_project, tmp_path):
    """Old Python-owned projects retain captured functions and scoring behavior."""
    with (scoring_project / "__init__.py").open("a", encoding="utf-8") as stream:
        stream.write(
            "def build_scoring():\n    return {'eligibility': [{'name': 'limit', 'version': 'old', 'function': 'scoring.exclude', 'params': {'limit': 4}}], 'outputs': []}\n"
        )
    result = score_pipeline(pd.DataFrame({"x": [2.0, 5.0]}), _fit(scoring_project, tmp_path))
    assert result.prediction.iloc[0] == pytest.approx(5.0)
    assert result.exclusion_reason.iloc[1] == "too_large"


def test_yaml_combined_cannot_compete_with_legacy_composition(scoring_project):
    """The legacy composition_config cannot override a YAML-owned combined section."""
    _write(scoring_project, {"version": 1})
    values = {"config_path": str(scoring_project.parents[1] / "config/workflow.json")}
    with pytest.raises(ValueError, match="both|owner"):
        capture_set_rules(values, {"composition_config": {"outputs": []}})


def test_yaml_combined_default_package_is_captured(scoring_project):
    """Omitting the legacy relative package option must not silently drop combined YAML."""
    _write(
        scoring_project,
        {
            "version": 1,
            "combined": [
                {
                    "name": "profit",
                    "custom": "scoring.profit",
                    "columns": [{"name": "profit", "dtype": "float64"}],
                    "required_components": ["sales", "cost"],
                }
            ],
        },
    )
    values = {"config_path": str(scoring_project.parents[1] / "config/workflow.json")}
    settings, source = capture_set_rules(values, {})
    assert settings["composition_config"]["outputs"][0]["function"] == "scoring.profit"
    assert (
        load_project_module(source).build_combined_rules()
        == settings["composition_config"]["outputs"]
    )


@pytest.mark.parametrize("with_yaml", [False, True])
def test_explicit_null_combined_path_is_rejected(scoring_project, with_yaml):
    """An explicitly invalid package path must not become the omitted-path legacy default."""
    if with_yaml:
        _write(scoring_project, {"version": 1})
    values = {"config_path": str(scoring_project.parents[1] / "config/workflow.json")}
    with pytest.raises(ValueError, match="combined_rules_path"):
        capture_set_rules(values, {"combined_rules_path": None})


def test_nested_combined_package_cannot_hide_yaml(scoring_project):
    """A nested package must not redirect canonical YAML lookup and drop its rules."""
    _write(scoring_project, {"version": 1})
    nested = scoring_project / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text(
        "raise RuntimeError('must not execute')\ndef build_combined_rules():\n    return []\n",
        encoding="utf-8",
    )
    values = {"config_path": str(scoring_project.parents[1] / "config/workflow.json")}
    with pytest.raises(ValueError, match="combined_rules_path"):
        capture_set_rules(values, {"combined_rules_path": "../features/nested"})


def test_legacy_nested_combined_package_without_yaml_remains_supported(scoring_project):
    """New YAML placement constraints must not break existing saved Python package layouts."""
    nested = scoring_project / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text(
        "def build_combined_rules():\n    return []\n", encoding="utf-8"
    )
    values = {"config_path": str(scoring_project.parents[1] / "config/workflow.json")}
    settings, source = capture_set_rules(values, {"combined_rules_path": "../features/nested"})
    assert settings["composition_config"] == {"outputs": []}
    assert load_project_module(source).build_combined_rules() == []


def test_yaml_combined_package_cannot_be_single_file(scoring_project):
    """A direct src Python file must not bypass the YAML package capture boundary."""
    _write(scoring_project, {"version": 1})
    (scoring_project.parent / "legacy.py").write_text(
        "raise RuntimeError('must not execute')\ndef build_combined_rules():\n    return []\n",
        encoding="utf-8",
    )
    values = {"config_path": str(scoring_project.parents[1] / "config/workflow.json")}
    with pytest.raises(ValueError, match="combined_rules_path"):
        capture_set_rules(values, {"combined_rules_path": "../legacy.py"})


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_yaml_rules_reach_each_training_layout(scoring_project, workflow_config, layout):
    """Single, competing and independent-target model builders must all freeze YAML rules."""
    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        load_training_branch_configs,
    )
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    _write(scoring_project, _rules())
    workflow_config.update(training_layout=layout, score_handoff="disabled")
    workflow_config["pipeline"]["modeling"] = {}
    models = {"linear": {"model": {"type": "linear_regression"}}}
    if layout == "model_competition":
        workflow_config.update(
            cv_enabled=True, cv_folds=2, cv_type="k_fold", cv_shuffle=True, cv_random_state=42
        )
        models["ridge"] = {"model": {"type": "ridge_regression"}}
    if layout == "multi_target":
        workflow_config.pop("target_column")
        models = {
            name: {"model": {"type": "linear_regression"}, "target_column": name}
            for name in ("revenue", "cost")
        }
    directory = scoring_project.parents[1] / "config"
    path = directory / "workflow.json"
    path.write_text(json.dumps(workflow_config), encoding="utf-8")
    (directory / "training.yml").write_text(
        yaml.safe_dump({"version": 1, "models": models}), encoding="utf-8"
    )
    if layout == "multi_target":
        resolved = load_training_branch_configs(
            {
                "config_path": str(path),
                "catalog": "workspace",
                "input_schema": "test",
                "output_schema": "test",
                "metadata_schema": "test",
                "resource_suffix": "",
                "workflow_contract": "3",
                "deployed_score_handoff": "disabled",
            }
        )
    else:
        workflow = load_project_workflow(read_workflow_config(path), scoring_project)
        resolved = (
            workflow["competition"]["candidates"]
            if layout == "model_competition"
            else {"linear": workflow}
        )
    assert set(resolved) == set(models)
    for entry in resolved.values():
        pipeline = entry["pipeline"]
        assert pipeline["project_scoring"]["eligibility"][0]["name"] == "observed"
        assert pipeline["project_scoring"]["outputs"][0]["columns"] == [
            {"name": "band", "dtype": "string"}
        ]
        assert (
            load_project_module(pipeline["project_python_source"]).build_model_rules()
            == pipeline["project_scoring"]
        )


@pytest.mark.parametrize("skip", [False, True])
def test_yaml_target_skip_retains_input_only_filters(scoring_project, skip):
    """Skipping unknown labels must leave feature eligibility intact and remain explicit."""
    filters = [
        {"name": name, "transformer": "DropMissingRows", "params": {"subset": [column]}}
        for name, column in (("known_target", "target"), ("known_feature", "x"))
    ]
    (scoring_project.parents[1] / "config/pre_split.yml").write_text(
        yaml.safe_dump({"version": 1, "recipes": {"default": filters}}), encoding="utf-8"
    )
    _write(scoring_project, {"version": 1, "skip_target_pre_split_steps": skip})
    if not skip:
        with pytest.raises(ValueError, match="reads target"):
            _workflow(scoring_project)
        return
    policy = _workflow(scoring_project)["pipeline"]["project_scoring"]["pre_split"]
    assert policy["skipped_target_steps"] == ["known_target"]
    assert [step["name"] for step in policy["steps"]] == ["known_feature"]


@pytest.mark.parametrize("with_yaml", [False, True])
@pytest.mark.parametrize("layout", ["src", "project"])
def test_legacy_single_file_cannot_silently_ignore_scoring_yaml(scoring_project, with_yaml, layout):
    """Legacy files remain supported alone but cannot compete with canonical scoring YAML."""
    directory = scoring_project.parent if layout == "src" else scoring_project.parents[1]
    path = directory / "preprocessing.py"
    source = "def build_preprocessing():\n    return []\ndef build_scoring():\n    return {'eligibility': [], 'outputs': []}\n"
    if with_yaml:
        _write(scoring_project, _rules())
        source = "raise RuntimeError('must not execute')\n" + source
    path.write_text(source, encoding="utf-8")
    config = {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}
    if with_yaml:
        with pytest.raises(ValueError, match="scoring.yml.*package"):
            load_project_workflow(config, path)
        return
    resolved = load_project_workflow(config, path)
    assert resolved["pipeline"]["project_scoring"] == {"eligibility": [], "outputs": []}


def test_legacy_file_ignores_unrelated_parent_project_yaml(scoring_project):
    """The package guard must not search above the legacy file's own project root."""
    _write(scoring_project, _rules())
    project = scoring_project.parents[1] / "independent_project"
    project.mkdir()
    path = project / "preprocessing.py"
    path.write_text("def build_preprocessing():\n    return []\n", encoding="utf-8")
    resolved = load_project_workflow(
        {"pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}}}, path
    )
    assert "project_scoring" not in resolved["pipeline"]
