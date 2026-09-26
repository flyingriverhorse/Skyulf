"""The generated local Bundle delegates to Skyulf's verified services."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / "templates/databricks/template/{{.project_name}}/src/workflow.py"
)


def _workflow():
    """Use the public library that generated notebooks delegate to."""
    from skyulf.integrations.databricks import local_workflow

    return local_workflow


def _output():
    """Inspect output publication independently of notebook widgets."""
    from skyulf.integrations.databricks import prediction_output

    return prediction_output


def _render_default_config(record_key="entity_id", risk_category=""):
    """Resolve default branches for offline checks; real CLI tests cover Go rendering."""
    template = WORKFLOW.parents[1] / "config/workflow.json.tmpl"
    schema = json.loads(
        (WORKFLOW.parents[3] / "databricks_template_schema.json").read_text(encoding="utf-8")
    )
    values = {key: spec["default"] for key, spec in schema["properties"].items()}
    values.update(
        project_name="customer_model",
        record_key_columns_json=json.dumps([record_key]),
        risk_category=risk_category,
    )
    content = template.read_text(encoding="utf-8")
    content = content[content.index("{\n") :].replace("{{$window}}", "full_snapshot")
    for name in (
        "risk_category",
        "event_time_format",
        "event_time_timezone",
        "result_time_format",
        "result_time_timezone",
        "event_column",
        "result_available_at_column",
        "start",
        "holdout_start",
        "cutoff",
        "result_cutoff",
    ):
        content = content.replace(
            "{{if ." + name + '}}"{{.' + name + '}}"{{else}}null{{end}}',
            json.dumps(values[name] or None),
        )
    content = (
        content.replace(
            '{{if eq .metric "auto"}}{{if eq .task "classification"}}heldout_accuracy'
            "{{else}}heldout_rmse{{end}}{{else}}{{.metric}}{{end}}",
            "heldout_rmse",
        )
        .replace(
            '{{if eq .task "classification"}}logistic_regression{{else}}linear_regression{{end}}',
            "linear_regression",
        )
        .replace(
            "{{if .source_table_name}}{{.source_table_name}}{{else}}{{.project_name}}_source{{end}}",
            "customer_model_source",
        )
        .replace(
            "{{if .score_source_table_name}}{{.score_source_table_name}}"
            "{{else}}customer_model_source{{end}}",
            "customer_model_source",
        )
    )
    content = (
        content.replace(
            '{{if eq $window "rolling_calendar"}}{{.monthly_lookback_months}}{{else}}null{{end}}',
            "null",
        )
        .replace(
            '{{if eq $window "rolling_calendar"}}"{{.window_timezone}}"{{else}}null{{end}}',
            "null",
        )
        .replace(
            '{{if eq .model_type "auto"}}linear_regression{{else}}{{.model_type}}{{end}}',
            "linear_regression",
        )
    )
    content = re.sub(
        r'{{if eq \.preprocessing "numeric_impute_scale"}}.*?{{end}}',
        "",
        content,
        flags=re.DOTALL,
    )
    for name, value in values.items():
        content = content.replace("{{." + name + "}}", str(value))
    return json.loads(content)


@pytest.mark.parametrize("risk_category", ["", "Low", "High"])
def test_bundle_risk_category_is_optional_and_configurable(risk_category):
    """Initialization must preserve the chosen business label without inventing a default risk."""
    root = WORKFLOW.parents[3]
    schema = json.loads((root / "databricks_template_schema.json").read_text())
    choice = schema["properties"]["risk_category"]
    assert choice["default"] == ""
    assert re.fullmatch(choice["pattern"], risk_category)
    assert not re.fullmatch(choice["pattern"], 'Low"invalid')
    assert _render_default_config(risk_category=risk_category)["risk_category"] == (
        risk_category or None
    )


def test_company_cost_tag_example_keeps_the_generic_cluster_choice():
    """A company setup must select PayingRegNo without changing the generic template default."""
    root = WORKFLOW.parents[3]
    schema = json.loads((root / "databricks_template_schema.json").read_text())
    example = json.loads((root / "examples/paying-reg-no-init.example.json").read_text())
    assert schema["properties"]["cost_tag_key"]["default"] == "CostCenter"
    assert "PayingRegNo" in schema["properties"]["cost_tag_key"]["description"]
    assert example["cost_tag_key"] == "PayingRegNo"
    assert example["compute_mode"] == "policy_cluster"
    assert example["cost_tag_value"] == "REPLACE_WITH_PAYING_REG_NO"


def test_generated_config_keeps_company_output_in_each_target():
    """The actual JSON template must bind its outputs separately in every target."""
    workflow = _workflow()
    config = _render_default_config()
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
    config = _render_default_config()
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


def test_generated_date_rules_do_not_assume_a_source_timezone():
    """Source timestamps need explicit interpretation independent of the job schedule."""
    config = _render_default_config()
    for name in ("event_time_parsing", "result_time_parsing"):
        assert config[name] == {"format": None, "timezone": None, "date_only": "reject"}


def test_generated_config_has_no_admission_or_alias_state():
    """The starting Bundle must not ask users to provision coordination tables."""
    config = _render_default_config()
    assert "score_admission_table" not in config
    assert "alias_admission_table" not in config
    assert "include_lifecycle" not in config


def test_init_record_key_becomes_prediction_table_key():
    """A chosen source identity must be carried into the generated output schema."""
    root = WORKFLOW.parents[3]
    schema = json.loads((root / "databricks_template_schema.json").read_text(encoding="utf-8"))
    assert json.loads(schema["properties"]["record_key_columns_json"]["default"]) == ["entity_id"]

    config = _render_default_config("customer_id")
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
    columns = _output()._prediction_columns(config, prepared, source)
    assert config["record_key_columns"] == ["customer_id"]
    assert columns[0] == ("customer_id", "string", "STRING")


def test_generated_bundle_has_only_train_and_serialized_score_jobs():
    """A new project must not silently bring back setup or control jobs."""
    resource = WORKFLOW.parents[1] / "resources/workflow.jobs.yml.tmpl"
    template = resource.read_text(encoding="utf-8")
    assert re.findall(r"^    ([a-z_]+):$", template, flags=re.MULTILINE) == ["train", "score"]
    assert re.search(
        r"^    score:\n      name:.*\n      max_concurrent_runs: 1$", template, re.MULTILINE
    )
    assert re.search(
        r"^    train:\n      name:.*\n      max_concurrent_runs: 1$", template, re.MULTILINE
    )
    assert 'if eq .retraining_mode "monthly_paused"' in template
    assert "pause_status: PAUSED" in template
    assert 'default: {{if eq .retraining_mode "monthly_paused"}}train_monthly' in template
    assert "quartz_cron_expression: ${var.retraining_cron_expression}" in template
    assert "timezone_id: ${var.retraining_timezone_id}" in template


def test_init_exposes_both_model_change_modes_and_editable_training_cadence():
    """The generated Bundle must let users choose scoring semantics and cron."""
    root = WORKFLOW.parents[3]
    schema = json.loads((root / "databricks_template_schema.json").read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert properties["model_change_mode"]["enum"] == ["incremental_append", "full_rebuild"]
    assert properties["retraining_cron_expression"]["default"] == "0 0 3 3 * ?"
    assert properties["retraining_timezone_id"]["default"] == "UTC"
    bundle = (WORKFLOW.parents[1] / "databricks.yml.tmpl").read_text(encoding="utf-8")
    assert 'default: "{{.retraining_cron_expression}}"' in bundle
    assert 'default: "{{.retraining_timezone_id}}"' in bundle


def test_auto_champion_init_exposes_metric_gates_and_reuses_score_job():
    """Automatic selection must be explicit and call the serialized score job."""
    root = WORKFLOW.parents[3]
    schema = json.loads((root / "databricks_template_schema.json").read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert properties["score_model_selection"]["enum"] == [
        "pinned_version",
        "champion",
    ]
    assert properties["promotion_policy"]["enum"] == ["manual_approval", "automatic"]
    assert "heldout_rmse" in properties["metric"]["enum"]
    assert "heldout_f1" in properties["metric"]["enum"]
    assert properties["quality_threshold"]["type"] == "string"
    assert properties["quality_threshold"]["default"] == "null"
    config = (WORKFLOW.parents[1] / "config/workflow.json.tmpl").read_text(encoding="utf-8")
    assert '"score_model_selection": "{{.score_model_selection}}"' in config
    assert _render_default_config()["metric"] == "heldout_rmse"
    jobs = (WORKFLOW.parents[1] / "resources/workflow.jobs.yml.tmpl").read_text(encoding="utf-8")
    assert "job_id: ${resources.jobs.score.id}" in jobs
    assert "task_key: score_after_lifecycle" in jobs
    assert jobs.count("queue:\n        enabled: true") == 2


def test_generated_default_training_does_not_require_dates():
    """Ordinary labeled tables must initialize without invented observation or result dates."""
    config = _render_default_config()
    assert config["split_strategy"] == "random"
    assert config["test_size"] == 0.2
    assert config["random_state"] == 42
    assert config["stratify"] is False
    assert config["filter_unavailable_results"] is False
    assert all(
        config[key] is None
        for key in (
            "event_column",
            "result_available_at_column",
            "start",
            "holdout_start",
            "cutoff",
            "result_cutoff",
            "monthly_lookback_months",
        )
    )


def test_generated_input_limit_uses_readable_megabytes():
    """The default must remain 64 MiB after replacing the byte-valued Bundle field."""
    config = _render_default_config()
    assert config["max_input_mb"] == 64
    assert "max_bytes" not in config
