"""The generated local Bundle delegates to Skyulf's verified services."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / "templates/databricks/template/{{.project_name}}/src/workflow.py"
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
        "score_admission_table": "workspace.test.score_admission",
        "alias_admission_table": "workspace.test.alias_admission",
        "model_name": "workspace.test.model",
        "model_version": "1",
        "champion_version": "1",
        "candidate_version": "2",
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
    monkeypatch.setattr(workflow, "promote_candidate", Mock(side_effect=AssertionError("promoted")))
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
    monkeypatch.setattr(workflow, "DeltaTableAdmission", Mock(return_value=object()))
    result = workflow.run_action(object(), _config(), "score")
    assert result.input_count == 2
    assert score.call_args.kwargs["row_keys"] == ("entity_id",)
    assert "period_start" not in score.call_args.kwargs
    assert "source_version" not in score.call_args.kwargs
    assert prepare.call_args.args[0].engine == "polars"


def test_compare_is_read_only_and_stage_does_not_promote(monkeypatch):
    """An eligible comparison must still need a separate staging and promotion job."""
    workflow = _workflow()
    report = SimpleNamespace(eligible=True, reason="eligible")
    heldout = object()
    monkeypatch.setattr(workflow, "_comparison", Mock(return_value=(report, heldout)))
    stage = Mock(return_value=SimpleNamespace(kind="challenger"))
    promote = Mock(side_effect=AssertionError("promoted without approval"))
    monkeypatch.setattr(workflow, "stage_challenger", stage)
    monkeypatch.setattr(workflow, "promote_candidate", promote)
    monkeypatch.setattr(workflow, "DeltaAliasAdmission", Mock(return_value=object()))
    assert workflow.run_action(object(), _config(), "compare") is report
    promote.assert_not_called()
    result = workflow.run_action(object(), _config(), "stage")
    assert result.kind == "challenger"
    assert stage.call_args.kwargs["expected_champion_version"] == "1"
    promote.assert_not_called()


def test_ineligible_candidate_cannot_reach_alias_writer(monkeypatch):
    """An inferior candidate cannot stage or promote through the Bundle entry point."""
    workflow = _workflow()
    monkeypatch.setattr(
        workflow,
        "_comparison",
        Mock(return_value=(SimpleNamespace(eligible=False, reason="inferior"), object())),
    )
    promote = Mock()
    monkeypatch.setattr(workflow, "promote_candidate", promote)
    with pytest.raises(ValueError, match="not eligible"):
        workflow.run_action(object(), _config(), "promote")
    promote.assert_not_called()


def test_target_binding_separates_test_and_prod_outputs():
    """One generated workflow must never reuse a test prediction table in prod."""
    workflow = _workflow()
    config = _config()
    config.update(
        training_table="{catalog}.{input_schema}.labeled_events",
        score_source_table="shared.raw.events",
        prediction_table="{catalog}.{output_schema}.predictions{resource_suffix}",
        model_name="{catalog}.{metadata_schema}.model{resource_suffix}",
        score_admission_table="{catalog}.{metadata_schema}.score_control{resource_suffix}",
        alias_admission_table="{catalog}.{metadata_schema}.alias_control{resource_suffix}",
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
        score_admission_table="prod_catalog.metadata.score_control",
        alias_admission_table="prod_catalog.metadata.alias_control",
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
        score_admission_table="prod_catalog.metadata.score_control",
        alias_admission_table="prod_catalog.metadata.alias_control",
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
