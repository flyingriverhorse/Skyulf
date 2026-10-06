"""Project model-set activation preserves deployment ownership and frozen scoring."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_databricks_branch_template import _project


def _enable(tmp_path, **overrides):
    """Declare a minimal coherent set without activating any business rule."""
    settings = {
        "model_name": "{catalog}.{metadata_schema}.example_set{resource_suffix}",
        "prediction_table": "{catalog}.{output_schema}.example_set_scores{resource_suffix}",
        "composition_config": {"outputs": []},
        **overrides,
    }
    (tmp_path / "src/modeling/model_set.py").write_text(
        f"def build_model_set():\n    return {settings!r}\n"
    )


def test_combined_rules_capture_uses_shared_features_only_during_training(
    tmp_path, workflow_config
):
    """New rules must be captured with the set while score configuration stays source-free."""
    from skyulf.integrations.databricks.model_sets.model_set_project import (
        capture_set_rules,
        load_project_model_set,
    )

    values, _ = _project(tmp_path, workflow_config)
    settings = {
        "model_name": "{catalog}.{metadata_schema}.example_set{resource_suffix}",
        "prediction_table": "{catalog}.{output_schema}.example_results{resource_suffix}",
        "combined_rules_path": "../features",
        "publication": {
            "mode": "separate_views",
            "model_views": {"revenue": "{catalog}.{output_schema}.revenue{resource_suffix}"},
            "combined_view": "{catalog}.{output_schema}.profit{resource_suffix}",
        },
    }
    (tmp_path / "src/modeling/model_set.py").write_text(
        f"def build_model_set():\n    return {settings!r}\n"
    )
    features = tmp_path / "src/features"
    (features / "__init__.py").write_text("def build_combined_rules():\n    return []\n")
    loaded = load_project_model_set(values)
    assert loaded is not None
    assert loaded["publication"]["combined_view"] == "workspace.outputs.profit_dev"
    captured, source = capture_set_rules(values, loaded)
    assert captured["composition_config"] == {"outputs": []} and source
    (features / "__init__.py").write_text("raise RuntimeError('editable code must not execute')")
    assert load_project_model_set(values) == loaded


def test_output_view_bindings_cannot_escape_deployment(tmp_path, workflow_config):
    """Nested publication destinations must follow the same ownership policy as the table."""
    from skyulf.integrations.databricks.model_sets.model_set_project import load_project_model_set

    values, _ = _project(tmp_path, workflow_config)
    _enable(
        tmp_path, publication={"mode": "separate_views", "combined_view": "other.db.results_dev"}
    )
    with pytest.raises(ValueError, match="output_schema"):
        load_project_model_set(values)


def test_source_correction_policy_reaches_the_score_notebook(
    tmp_path, workflow_config, monkeypatch
):
    """The deployed factory must pass correction permission to the actual batch service."""
    from tests.integration.platforms.test_model_set_batch import _saved_set

    from skyulf.integrations.databricks.model_sets import model_set_batch
    from skyulf.integrations.databricks.model_sets import model_set_project as module
    from skyulf.integrations.mlflow.models import model_set

    values, _ = _project(tmp_path, workflow_config)
    _enable(tmp_path, source_change_policy="rebuild_on_change")
    artifact, model, _ = _saved_set(tmp_path / "artifact")
    values["score_model_version"] = "1"
    monkeypatch.setattr(module, "validate_job_parameters", lambda values: None)
    monkeypatch.setattr(module, "resolve_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(model_set, "load_registered_model_set", lambda *args, **kwargs: artifact)
    batch = Mock(return_value=model_set_batch.ModelSetBatchResult(1, 2, 2, 0, {}, False))
    monkeypatch.setattr(model_set_batch, "run_model_set_batch", batch)
    dbutils = SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values), notebook=Mock())
    output = module.run_model_set_score_notebook(None, dbutils, exit_notebook=False)
    assert batch.call_args.kwargs["source_change_policy"] == "rebuild_on_change"
    assert '"source_change_policy": "rebuild_on_change"' in output


@pytest.mark.parametrize(
    "policy_name", ["workspace.models.reveneu_dev", "workspace.models.revenue_dev"]
)
def test_component_performance_policy_fails_before_scoring_write(
    tmp_path, workflow_config, monkeypatch, policy_name
):
    """Unknown component names and missing labels must fail before publishing predictions."""
    from skyulf.integrations.databricks.model_sets import model_set_batch
    from skyulf.integrations.databricks.model_sets import model_set_project as module
    from skyulf.integrations.mlflow.models import model_set

    values, _ = _project(tmp_path, workflow_config)
    _enable(tmp_path)
    values.update(
        score_model_version="1",
        monitoring_catalog="ops",
        monitoring_schema="monitoring",
        monitoring_environment="prod",
        monitoring_project="risk",
        monitoring_performance_policies=json.dumps(
            {
                policy_name: {
                    "mode": "report",
                    "metric": "f1_weighted",
                    "direction": "higher",
                    "baseline": {"kind": "training_holdout", "model_version": "1"},
                    "tolerance": 0.05,
                    "tolerance_mode": "absolute",
                    "window_hours": 24,
                    "label_delay_hours": 6,
                    "minimum_labeled_rows": 20,
                    "minimum_label_coverage": 0.8,
                    "consecutive_windows": 3,
                }
            }
        ),
    )
    resolved = SimpleNamespace(name="workspace.models.example_set_dev", version="1")
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            components=[
                SimpleNamespace(
                    reference=SimpleNamespace(name="workspace.models.revenue_dev", version="1"),
                    branch="revenue",
                )
            ]
        )
    )
    monkeypatch.setattr(module, "resolve_model", Mock(return_value=resolved))
    monkeypatch.setattr(model_set, "load_registered_model_set", Mock(return_value=artifact))
    batch = Mock(side_effect=AssertionError("batch started before performance preflight"))
    monkeypatch.setattr(model_set_batch, "run_model_set_batch", batch)
    config = module.read_notebook_config(values)
    with pytest.raises(ValueError, match="component|label_table"):
        module.score_model_set_payload(None, config, values)
    batch.assert_not_called()


def test_set_factory_resolves_owned_names_and_preserves_legacy(tmp_path, workflow_config):
    """Existing projects remain train only until a factory explicitly returns a set."""
    from skyulf.integrations.databricks.model_sets.model_set_project import load_project_model_set

    values, _ = _project(tmp_path, workflow_config)
    assert load_project_model_set(values) is None
    _enable(tmp_path)
    result = load_project_model_set(values)
    assert result is not None
    assert result["model_name"] == "workspace.models.example_set_dev"
    assert result["prediction_table"] == "workspace.outputs.example_set_scores_dev"


@pytest.mark.parametrize("name", ["other.models.set_dev", "workspace.models.set"])
def test_set_factory_rejects_escaping_names(tmp_path, workflow_config, name):
    """A set cannot bypass catalog ownership or the active deployment suffix."""
    from skyulf.integrations.databricks.model_sets.model_set_project import load_project_model_set

    values, _ = _project(tmp_path, workflow_config)
    _enable(tmp_path, model_name=name)
    with pytest.raises(ValueError, match="metadata_schema|resource_suffix"):
        load_project_model_set(values)


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_set_operator_never_loads_training_or_composition(
    tmp_path, workflow_config, monkeypatch, action
):
    """Saved-set operations must work after editable training and rule source disappear."""
    from skyulf.integrations.databricks.jobs.training.branch_notebook import (
        run_branch_training_notebook,
    )
    from skyulf.integrations.databricks.model_sets import model_set_project as module

    values, _ = _project(tmp_path, workflow_config)
    _enable(tmp_path)
    Path(tmp_path / "src/modeling/branches.py").unlink()
    values.update(lifecycle_action=action, candidate_version="2", expected_champion_version="none")
    if action == "reject":
        values["rejection_reason"] = "Business review failed"
    operation = Mock(return_value={"action": action, "model_set_version": "2"})
    monkeypatch.setattr(module, "run_model_set_operator", operation)
    dbutils = SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values), notebook=Mock())
    result = run_branch_training_notebook(None, dbutils, exit_notebook=False)
    assert f'"action": "{action}"' in result
    assert operation.call_args.args[2]["model_name"] == "workspace.models.example_set_dev"


def test_spark_approval_limits_only_functional_probe_after_global_key_check(monkeypatch):
    """Large inference populations must not become local model-set approval inputs."""
    from skyulf.inference import model_set_partition_safety
    from skyulf.integrations.databricks.model_sets import model_set_project as module
    from skyulf.integrations.databricks.scoring.batch import spark_scoring

    spark = Mock()
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            input_schema=[SimpleNamespace(name="id"), SimpleNamespace(name="x")],
            record_key_columns=("id",),
        )
    )
    gate = Mock()
    monkeypatch.setattr(model_set_partition_safety, "require_partition_safe_model_set", gate)
    monkeypatch.setattr(module, "latest_source_version", lambda *args: {"version": 17})
    selected = Mock()
    distributed = Mock(return_value=SimpleNamespace(frame=selected, row_count=4096))
    monkeypatch.setattr(spark_scoring, "read_distributed_rows", distributed)
    bounded = Mock(return_value="probe")
    monkeypatch.setattr(module, "bounded_frame", bounded)
    result = module.approval_frame(
        spark,
        artifact,
        {
            "score_source_table": "a.b.source",
            "max_rows": 80,
            "max_input_mb": 4,
            "inference_mode": "spark",
        },
    )
    gate.assert_called_once_with(artifact)
    distributed.assert_called_once_with(
        spark.read.option.return_value.table.return_value, ("id", "x"), ("id",)
    )
    selected.orderBy.assert_called_once_with("id")
    selected.orderBy.return_value.limit.assert_called_once_with(80)
    assert bounded.call_args.args[0] is selected.orderBy.return_value.limit.return_value
    assert result == "probe"


def test_spark_approval_rejects_unsafe_set_before_source_access(monkeypatch):
    """Functional sampling is safe only for a certified partition-independent set."""
    from skyulf.inference import model_set_partition_safety
    from skyulf.integrations.databricks.model_sets import model_set_project as module

    spark = Mock()
    monkeypatch.setattr(
        model_set_partition_safety,
        "require_partition_safe_model_set",
        Mock(side_effect=ValueError("unsafe set")),
    )
    with pytest.raises(ValueError, match="unsafe set"):
        module.approval_frame(spark, object(), {"inference_mode": "spark"})
    assert spark.mock_calls == []


def test_set_key_schema_accepts_only_portable_types():
    """Floating keys are unsuitable for stable joins and must fail before packaging."""
    from skyulf.integrations.databricks.model_sets.model_set_project import _record_key_schema

    source = SimpleNamespace(dtypes=[("id", "bigint"), ("label", "string"), ("active", "boolean")])
    schema = _record_key_schema(source, ("id", "label", "active"))
    assert [column.dtype for column in schema] == ["int64", "string", "bool"]
    with pytest.raises(ValueError, match="record key"):
        _record_key_schema(SimpleNamespace(dtypes=[("id", "double")]), ("id",))


def test_approval_frame_pins_integer_version_and_saved_columns(monkeypatch):
    """Delta history rows must be unpacked before configuring versionAsOf."""
    from skyulf.integrations.databricks.model_sets import model_set_project as module

    spark = Mock()
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            input_schema=[SimpleNamespace(name="id"), SimpleNamespace(name="x")],
            record_key_columns=("id",),
        )
    )
    monkeypatch.setattr(
        module, "latest_source_version", lambda *args: {"version": 17, "userMetadata": None}
    )
    bounded = Mock(return_value="frame")
    monkeypatch.setattr(module, "bounded_frame", bounded)
    result = module.approval_frame(
        spark, artifact, {"score_source_table": "a.b.source", "max_rows": 80, "max_input_mb": 4}
    )
    spark.read.option.assert_called_once_with("versionAsOf", 17)
    assert bounded.call_args.args[1:4] == (("id", "x"), ("id",), 80)
    assert result == "frame"


@pytest.mark.parametrize(
    "model_type", ["linear_regression", "logistic_regression", "voting_regressor"]
)
def test_component_tags_preserve_long_names_and_tuned_model_type(model_type):
    """Inspection tags must preserve full model names and identify the tuned base estimator."""
    from typing import Any, cast

    from skyulf.integrations.databricks.model_sets.model_set_project import _component_tags

    name = "catalog." + "s" * 250 + ".revenue"
    branch = SimpleNamespace(
        name="revenue",
        pipeline={
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {"type": model_type},
            }
        },
    )
    outcome = SimpleNamespace(
        components={"revenue": SimpleNamespace(model_name=name, model_version="7")}
    )
    tags = _component_tags(cast(Any, (branch,)), cast(Any, outcome))
    assert tags["model_set_revenue_name"] + tags["model_set_revenue_name_2"] == name
    assert tags["model_set_revenue_type"] == model_type
    assert tags["model_set_revenue_version"] == "7"
    assert all(len(value.encode()) <= 256 for value in tags.values())


def test_completed_training_packages_original_registered_components(
    tmp_path, workflow_config, monkeypatch
):
    """Real completed training must become a frozen set with no component alias changes."""
    mlflow = pytest.importorskip("mlflow")
    from test_local_branches import _configs, _data

    from skyulf.inference.model_set_scoring import predict_model_set
    from skyulf.integrations.databricks.model_sets.model_set_project import (
        package_training_model_set,
    )
    from skyulf.integrations.databricks.training import local_branches
    from skyulf.integrations.databricks.training.fitting import local_retraining
    from skyulf.integrations.mlflow.models.model_set import load_registered_model_set

    store = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=store, registry_uri=store)
    client.create_experiment("branches", artifact_location=(tmp_path / "runs").as_uri())
    configs = _configs(workflow_config, store=store)
    configs.pop("ensemble")
    configs["amount"]["cv_enabled"] = False
    monkeypatch.setattr(
        local_retraining,
        "read_training_snapshot",
        lambda spark, spec: _data().loc[:, list(spec.source_columns)].copy(),
    )
    branches = local_branches.prepare_training_branches(None, configs)
    outcome = local_branches.train_local_branches(
        None,
        branches,
        tracking_uri=store,
        registry_uri=store,
        experiment_name="branches",
        artifact_path=tmp_path / "fit",
    )
    spark = Mock()
    spark.read.option.return_value.table.return_value = SimpleNamespace(dtypes=[("id", "bigint")])
    settings = {"model_name": "workspace.test.set", "composition_config": {"outputs": []}}
    options = {"composition_source": "", "tracking_uri": store, "registry_uri": store}
    with pytest.raises(ValueError, match="complete branch"):
        package_training_model_set(
            spark, branches, replace(outcome, components={}), settings, **options
        )
    changed = (replace(branches[0], model_name="workspace.test.other"), *branches[1:])
    with pytest.raises(ValueError, match="training plan"):
        package_training_model_set(spark, changed, outcome, settings, **options)
    resolved = package_training_model_set(spark, branches, outcome, settings, **options)
    tags = client.get_model_version(resolved.name, resolved.version).tags
    assert tags["model_set_model_count"] == "2"
    for branch in branches:
        result = outcome.components[branch.name]
        assert tags[f"model_set_{branch.name}_name"] == result.model_name
        assert tags[f"model_set_{branch.name}_version"] == result.model_version
    assert tags["model_set_amount_type"] == "linear_regression"
    assert tags["model_set_category_type"] == "logistic_regression"
    saved = load_registered_model_set(resolved, tracking_uri=store, registry_uri=store)
    for name, result in outcome.components.items():
        original = tmp_path / "fit" / name / "pipeline.pkl"
        assert (
            original.read_bytes()
            == (saved.directory / "components" / name / "pipeline.pkl").read_bytes()
        )
        assert client.get_registered_model(result.model_name).aliases == {}
    predictions = predict_model_set(_data()[["id", "x"]], saved)
    assert len(predictions) == 60
    assert {"amount__prediction", "category__prediction"} <= set(predictions)
    assert client.get_registered_model(resolved.name).aliases == {}
    revised = {
        **settings,
        "composition_config": {
            "outputs": [
                {
                    "name": "value",
                    "version": "2",
                    "function": "value",
                    "params": {},
                    "columns": [{"name": "amount_value", "dtype": "float64"}],
                    "required_components": ["amount"],
                }
            ]
        },
    }
    revised_options = {
        **options,
        "composition_source": "import pandas as pd\ndef value(inputs, predictions, params):\n"
        "    return pd.DataFrame({'amount_value': predictions['amount__prediction']})\n",
    }
    newer = package_training_model_set(spark, branches, outcome, revised, **revised_options)
    assert client.get_model_version(newer.name, newer.version).tags == tags
    assert newer.digest != resolved.digest
    original_again = load_registered_model_set(resolved, tracking_uri=store, registry_uri=store)
    replay = predict_model_set(_data()[["id", "x"]], original_again)
    assert replay.equals(predictions)


@pytest.mark.parametrize("override", ["", "3"])
def test_score_entrypoint_pins_set_and_uses_saved_artifact(
    tmp_path, workflow_config, monkeypatch, override
):
    """Scoring must pin at most once and never reopen editable training or business code."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.databricks.model_sets import model_set_batch
    from skyulf.integrations.databricks.model_sets import model_set_project as module
    from skyulf.integrations.mlflow.models import model_set
    from skyulf.integrations.mlflow.registration.registry import ResolvedModel

    values, _ = _project(tmp_path, workflow_config)
    _enable(tmp_path)
    (tmp_path / "src/modeling/branches.py").unlink()
    values.update(score_model_version=override, lifecycle_action="approve", candidate_version="99")
    champion = Mock(return_value="2")
    resolved = ResolvedModel(
        "workspace.models.example_set_dev", override or "2", "models:/set/2", None, "a" * 64
    )
    resolve = Mock(return_value=resolved)
    saved = object()
    monkeypatch.setattr(module, "controlled_champion_version", champion)
    monkeypatch.setattr(module, "resolve_model", resolve)
    monkeypatch.setattr(model_set, "load_registered_model_set", Mock(return_value=saved))
    batch = Mock(return_value=model_set_batch.ModelSetBatchResult(1, 2, 2, 0, {}, False))
    monkeypatch.setattr(model_set_batch, "run_model_set_batch", batch)
    dbutils = SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values), notebook=Mock())
    result = module.run_model_set_score_notebook("spark", dbutils, exit_notebook=False)
    assert champion.call_count == (0 if override else 1)
    assert resolve.call_args.kwargs["version"] == (override or "2")
    assert batch.call_args.args == ("spark", resolved, saved)
    assert batch.call_args.kwargs["prediction_table"] == "workspace.outputs.example_set_scores_dev"
    assert '"output_count": 2' in result
    assert json.loads(result)["source_table"] == batch.call_args.kwargs["source_table"]
