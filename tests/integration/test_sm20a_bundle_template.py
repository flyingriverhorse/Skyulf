"""The generated local Bundle delegates to Skyulf's verified services."""

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / "skyulf-core/templates/databricks/template/{{.project_name}}/src/workflow.py"
)


def _workflow():
    """Load the source copied unchanged into every generated project."""
    spec = importlib.util.spec_from_file_location("sm20a_workflow", WORKFLOW)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config():
    """Keep a concrete, bounded source and model selection in each test."""
    return {
        "engine": "polars",
        "training_table": "workspace.test.training",
        "training_version": 0,
        "score_source_table": "workspace.test.source",
        "prediction_table": "workspace.test.predictions",
        "model_name": "workspace.test.model",
        "model_version": "1",
        "champion_version": "1",
        "row_keys": ["entity_id"],
        "input_columns": ["x"],
        "target_column": "target",
        "event_column": "event_time",
        "label_time_column": "label_at",
        "start": "2026-01-01T00:00:00+00:00",
        "holdout_start": "2026-02-01T00:00:00+00:00",
        "cutoff": "2026-03-01T00:00:00+00:00",
        "max_rows": 100,
        "max_bytes": 100_000,
        "metric": "heldout_rmse",
        "min_improvement": 0.0,
        "quality_threshold": None,
        "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
    }


def test_train_preserves_selected_polars_engine_and_never_promotes(monkeypatch, tmp_path):
    """A training job must only produce a candidate with the chosen fit engine."""
    workflow = _workflow()
    train = Mock(return_value=SimpleNamespace(model_version="2"))
    monkeypatch.setattr(workflow, "train_local_candidate", train)
    result = workflow.run_action(
        object(),
        _config(),
        "train",
        experiment_name="/Users/test/experiment",
        artifact_path=tmp_path / "artifact",
    )
    assert result.model_version == "2"
    assert train.call_args.kwargs["engine"] == "polars"
    assert train.call_args.args[1].version == 0


def test_score_uses_incremental_service_without_period_or_source_version(monkeypatch):
    """Recurring scoring must derive new inserts from committed Delta receipts."""
    workflow = _workflow()
    prepare = Mock(return_value=object())
    score = Mock(return_value=SimpleNamespace(noop=False, input_count=2))
    monkeypatch.setattr(workflow, "prepare_local_workflow", prepare)
    monkeypatch.setattr(workflow, "run_incremental_local_batch", score)
    monkeypatch.setattr(workflow, "provision_prediction_table", Mock())
    result = workflow.run_action(object(), _config(), "score")
    assert result.input_count == 2
    assert score.call_args.kwargs["row_keys"] == ("entity_id",)
    assert "period_start" not in score.call_args.kwargs
    assert "source_version" not in score.call_args.kwargs
    assert prepare.call_args.args[0].engine == "polars"
    assert isinstance(score.call_args.kwargs["admission"], workflow.SingleWriterAdmission)


def test_lifecycle_actions_are_not_bundle_entry_points():
    """The minimal Bundle must not accidentally mutate registry aliases."""
    workflow = _workflow()
    for action in ("compare", "stage", "promote", "setup"):
        with pytest.raises(ValueError, match="Unknown workflow action"):
            workflow.run_action(object(), _config(), action)


def test_target_binding_separates_test_and_prod_outputs():
    """One generated workflow must never reuse a test prediction table in prod."""
    workflow = _workflow()
    config = _config()
    config.update(
        training_table="{catalog}.{input_schema}.labeled_events",
        score_source_table="shared.raw.events",
        prediction_table="{catalog}.{output_schema}.predictions{resource_suffix}",
        model_name="{catalog}.{metadata_schema}.model{resource_suffix}",
    )
    shared = {
        "input_schema": "refined",
        "output_schema": "mlresult",
        "metadata_schema": "metadata",
    }
    test_config = workflow.resolve_target_config(
        config, {**shared, "catalog": "test_catalog", "resource_suffix": "_murat"}
    )
    syst_config = workflow.resolve_target_config(
        config, {**shared, "catalog": "syst_catalog", "resource_suffix": ""}
    )
    prod_config = workflow.resolve_target_config(
        config, {**shared, "catalog": "prod_catalog", "resource_suffix": ""}
    )
    assert test_config["prediction_table"] == "test_catalog.mlresult.predictions_murat"
    assert syst_config["prediction_table"] == "syst_catalog.mlresult.predictions"
    assert prod_config["prediction_table"] == "prod_catalog.mlresult.predictions"
    assert prod_config["score_source_table"] == "shared.raw.events"
    assert test_config["model_name"] != prod_config["model_name"]
    assert (
        len(
            {
                test_config["prediction_table"],
                syst_config["prediction_table"],
                prod_config["prediction_table"],
            }
        )
        == 3
    )
    assert config["prediction_table"] == "{catalog}.{output_schema}.predictions{resource_suffix}"


def test_target_binding_rejects_unsafe_catalog_before_any_job():
    """A malformed target parameter must fail before it reaches a UC operation."""
    workflow = _workflow()
    with pytest.raises(ValueError, match="catalog"):
        workflow.resolve_target_config(
            _config(),
            {
                "catalog": "prod;DROP TABLE x",
                "input_schema": "refined",
                "output_schema": "mlresult",
                "metadata_schema": "metadata",
                "resource_suffix": "",
            },
        )


def test_target_binding_rejects_cross_environment_output():
    """A prod job cannot write a model or prediction into a test catalog."""
    workflow = _workflow()
    config = _config()
    config.update(
        model_name="prod_catalog.metadata.model",
    )
    config["prediction_table"] = "test_catalog.mlresult.predictions"
    with pytest.raises(ValueError, match="prediction_table"):
        workflow.resolve_target_config(
            config,
            {
                "catalog": "prod_catalog",
                "input_schema": "refined",
                "output_schema": "mlresult",
                "metadata_schema": "metadata",
                "resource_suffix": "",
            },
        )


def test_target_binding_rejects_cross_environment_model():
    """A prod train job cannot register a model in a test catalog."""
    workflow = _workflow()
    config = _config()
    config.update(
        prediction_table="prod_catalog.mlresult.predictions",
    )
    config["model_name"] = "test_catalog.metadata.model"
    with pytest.raises(ValueError, match="model_name"):
        workflow.resolve_target_config(
            config,
            {
                "catalog": "prod_catalog",
                "input_schema": "refined",
                "output_schema": "mlresult",
                "metadata_schema": "metadata",
                "resource_suffix": "",
            },
        )


def test_generated_config_keeps_company_output_in_each_target():
    """The actual JSON template must bind its outputs separately in every target."""
    workflow = _workflow()
    template = WORKFLOW.parents[1] / "config/workflow.json.tmpl"
    config = json.loads(
        template.read_text(encoding="utf-8")
        .replace("{{.project_name}}", "customer_model")
        .replace("{{.engine}}", "pandas")
        .replace("{{.row_key}}", "entity_id")
    )
    outputs = {}
    for target, catalog, suffix in (
        ("test", "test_catalog", "_murat"),
        ("syst", "syst_catalog", ""),
        ("prod", "prod_catalog", ""),
    ):
        bound = workflow.resolve_target_config(
            config,
            {
                "catalog": catalog,
                "input_schema": "dsp_refined",
                "output_schema": "dsp_mlresult",
                "metadata_schema": "dsp_metadata",
                "resource_suffix": suffix,
            },
        )
        assert bound["score_source_table"] == f"{catalog}.dsp_refined.customer_model_source"
        assert bound["model_name"] == f"{catalog}.dsp_metadata.customer_model_model{suffix}"
        outputs[target] = bound["prediction_table"]
    assert outputs == {
        "test": "test_catalog.dsp_mlresult.customer_model_predictions_murat",
        "syst": "syst_catalog.dsp_mlresult.customer_model_predictions",
        "prod": "prod_catalog.dsp_mlresult.customer_model_predictions",
    }


def test_minimal_generated_config_uses_one_existing_source():
    """Default training and scoring should reference one existing source table."""
    workflow = _workflow()
    template = WORKFLOW.parents[1] / "config/workflow.json.tmpl"
    config = json.loads(
        template.read_text(encoding="utf-8")
        .replace("{{.project_name}}", "customer_model")
        .replace("{{.engine}}", "pandas")
        .replace("{{.row_key}}", "entity_id")
    )
    bound = workflow.resolve_target_config(
        config,
        {
            "catalog": "test_catalog",
            "input_schema": "input_schema",
            "output_schema": "output_schema",
            "metadata_schema": "metadata_schema",
            "resource_suffix": "",
        },
    )
    assert bound["training_table"] == bound["score_source_table"]


def test_score_provision_rejects_missing_source_without_creating_tables():
    """A wrong source reference must not create partial prediction resources."""
    workflow = _workflow()
    spark = Mock()
    spark.catalog.tableExists.return_value = False
    with pytest.raises(ValueError, match="source"):
        workflow.provision_prediction_table(spark, _config(), object())
    spark.sql.assert_not_called()


def test_score_does_not_require_labeled_training_table(monkeypatch):
    """A retired training snapshot must not prevent scoring a live source."""
    workflow = _workflow()
    config = _config()
    config["training_table"] = "workspace.test.retired_training"
    spark = Mock()
    spark.catalog.tableExists.side_effect = lambda name: name != config["training_table"]
    spark.sql.return_value.first.return_value = {
        "properties": {"delta.enableChangeDataFeed": "true"}
    }
    monkeypatch.setattr(workflow, "_prediction_columns", Mock(return_value=()))
    monkeypatch.setattr(workflow, "_check_existing_table", Mock())
    prepared = SimpleNamespace(preflight=SimpleNamespace(ready=True))
    assert workflow.provision_prediction_table(spark, config, prepared) is False


def test_score_replay_validates_existing_output_without_recounting_source(monkeypatch):
    """A later score must accept a growing source without recreating its output."""
    workflow = _workflow()
    config = _config()
    source = Mock()
    source.columns = ["entity_id", "x"]
    source.schema = {
        "entity_id": SimpleNamespace(dataType=SimpleNamespace(typeName=lambda: "string"))
    }
    spark = Mock()
    spark.catalog.tableExists.return_value = True
    spark.table.return_value = source
    spark.sql.return_value.first.return_value = {
        "properties": {"delta.enableChangeDataFeed": "true"}
    }
    check_target = Mock()
    monkeypatch.setattr(workflow, "_check_existing_table", check_target)
    prepared = SimpleNamespace(
        artifact=SimpleNamespace(manifest=SimpleNamespace(input_columns=("x",))),
        preflight=SimpleNamespace(
            ready=True,
            output_schema=(SimpleNamespace(name="prediction", dtype="float64"),),
        ),
    )
    result = workflow.provision_prediction_table(spark, config, prepared)
    check_target.assert_called_once()
    source.select.assert_not_called()
    assert result is False
    assert all(not call.args[0].startswith("CREATE TABLE") for call in spark.sql.call_args_list)


def test_first_score_provisions_prediction_before_incremental_write(monkeypatch):
    """The scheduled score job must handle first-run setup without another job."""
    workflow = _workflow()
    events = []
    prepared = object()
    monkeypatch.setattr(workflow, "prepare_local_workflow", Mock(return_value=prepared))
    monkeypatch.setattr(
        workflow,
        "provision_prediction_table",
        Mock(side_effect=lambda spark, config, model: events.append("provision")),
        raising=False,
    )
    monkeypatch.setattr(
        workflow,
        "run_incremental_local_batch",
        Mock(side_effect=lambda *args, **kwargs: events.append("score")),
    )
    workflow.run_action(object(), _config(), "score")
    assert events == ["provision", "score"]


def test_prediction_provision_creates_no_control_table(monkeypatch):
    """First scoring must create only the output table, with no lock table."""
    workflow = _workflow()
    config = _config()
    source = Mock()
    source.columns = ["entity_id", "x"]
    source.schema = {
        "entity_id": SimpleNamespace(dataType=SimpleNamespace(typeName=lambda: "string"))
    }
    source.select.return_value.limit.return_value.count.return_value = 2
    spark = Mock()
    spark.catalog.tableExists.side_effect = lambda name: (
        name
        in {
            config["training_table"],
            config["score_source_table"],
        }
    )
    spark.table.return_value = source
    spark.sql.return_value.first.return_value = {
        "properties": {"delta.enableChangeDataFeed": "true"}
    }
    prepared = SimpleNamespace(
        artifact=SimpleNamespace(manifest=SimpleNamespace(input_columns=("x",))),
        preflight=SimpleNamespace(
            ready=True,
            output_schema=(SimpleNamespace(name="prediction", dtype="float64"),),
        ),
    )
    created = workflow.provision_prediction_table(spark, config, prepared)
    statements = [call.args[0] for call in spark.sql.call_args_list]
    assert created is True
    assert len([statement for statement in statements if statement.startswith("CREATE TABLE")]) == 1
    assert all("admission" not in statement for statement in statements)


def test_generated_config_has_no_admission_or_alias_state():
    """The starting Bundle must not ask users to provision coordination tables."""
    template = WORKFLOW.parents[1] / "config/workflow.json.tmpl"
    config = json.loads(
        template.read_text(encoding="utf-8")
        .replace("{{.project_name}}", "customer_model")
        .replace("{{.engine}}", "pandas")
        .replace("{{.row_key}}", "entity_id")
    )
    assert "score_admission_table" not in config
    assert "alias_admission_table" not in config
    assert "include_lifecycle" not in config


def test_init_row_key_becomes_prediction_table_key():
    """A chosen source identity must be carried into the generated output schema."""
    root = WORKFLOW.parents[3]
    schema = json.loads((root / "databricks_template_schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["row_key"]["default"] == "entity_id"

    template = WORKFLOW.parents[1] / "config/workflow.json.tmpl"
    config = json.loads(
        template.read_text(encoding="utf-8")
        .replace("{{.project_name}}", "customer_model")
        .replace("{{.engine}}", "pandas")
        .replace("{{.row_key}}", "customer_id")
    )
    source = SimpleNamespace(
        columns=["customer_id", "feature_value"],
        schema={
            "customer_id": SimpleNamespace(dataType=SimpleNamespace(typeName=lambda: "string"))
        },
    )
    prepared = SimpleNamespace(
        artifact=SimpleNamespace(manifest=SimpleNamespace(input_columns=("feature_value",))),
        preflight=SimpleNamespace(
            output_schema=(SimpleNamespace(name="prediction", dtype="float64"),)
        ),
    )
    columns = _workflow()._prediction_columns(config, prepared, source)
    assert config["row_keys"] == ["customer_id"]
    assert columns[0] == ("customer_id", "string", "STRING")


def test_generated_bundle_has_only_train_and_serialized_score_jobs():
    """A new project must not silently bring back setup or control jobs."""
    resource = WORKFLOW.parents[1] / "resources/workflow.jobs.yml.tmpl"
    template = resource.read_text(encoding="utf-8")
    assert re.findall(r"^    ([a-z_]+):$", template, flags=re.MULTILINE) == ["train", "score"]
    assert re.search(
        r"^    score:\n      name:.*\n      max_concurrent_runs: 1$", template, re.MULTILINE
    )
