"""The generated local Bundle delegates to Skyulf's verified services."""

import json
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "templates/databricks/template/{{.project_name}}/src/jobs/initialize_run.py"
)


def test_inference_choice_precedes_training_engine_and_rejects_conflicts():
    """The setup wizard must reject an explicit Polars/Spark contradiction."""
    from jsonschema import Draft7Validator

    schema = json.loads((WORKFLOW.parents[4] / "databricks_template_schema.json").read_text())
    properties = schema["properties"]
    assert properties["inference_mode"]["order"] < properties["engine"]["order"]
    assert Draft7Validator(schema).is_valid({"inference_mode": "spark", "engine": "pandas"})
    assert not Draft7Validator(schema).is_valid({"inference_mode": "spark", "engine": "polars"})
    assert Draft7Validator(properties["engine"]["skip_prompt_if"]).is_valid(
        {"inference_mode": "spark"}
    )


def test_cdf_recovery_wizard_is_visible_and_disabled_by_default():
    """Full-rescore authorization requires an explicit guided choice in every layout."""
    from jsonschema import Draft7Validator

    schema = json.loads((WORKFLOW.parents[4] / "databricks_template_schema.json").read_text())
    setting = schema["properties"]["auto_rebuild_on_cdf_expiry"]
    assert setting["default"] == "false"
    assert setting["enum"] == ["false", "true"]
    assert "skip_prompt_if" not in setting
    assert Draft7Validator(setting).is_valid("true")
    assert not Draft7Validator(setting).is_valid(True)
    assert not Draft7Validator(setting).is_valid("yes")


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_initializer_only_shows_task_specific_models_and_metrics(task):
    """Changing task must change visible menus while keeping them aligned with Core."""
    from jsonschema import Draft7Validator

    from skyulf.integrations.mlflow.lifecycle.validation import _CLASSIFICATION, _REGRESSION
    from skyulf.modeling.base import BaseModelCalculator
    from skyulf.registry import NodeRegistry

    schema = json.loads((WORKFLOW.parents[4] / "databricks_template_schema.json").read_text())
    properties = schema["properties"]
    values = {name: field["default"] for name, field in properties.items()}
    values["task"] = task
    other = "regression" if task == "classification" else "classification"
    for suffix in ("model", "metric"):
        assert not Draft7Validator(properties[f"{task}_{suffix}"]["skip_prompt_if"]).is_valid(
            values
        )
        assert Draft7Validator(properties[f"{other}_{suffix}"]["skip_prompt_if"]).is_valid(values)
    expected_models = set()
    for name in NodeRegistry.get_all_metadata():
        calculator = NodeRegistry.get_calculator(name)
        if issubclass(calculator, BaseModelCalculator) and calculator().problem_type == task:
            expected_models.add(name)
    assert set(properties[f"{task}_model"]["enum"]) == expected_models
    assert set(properties[f"{task}_metric"]["enum"]) == (
        _CLASSIFICATION if task == "classification" else _REGRESSION
    )
    assert "preprocessing" not in properties


def _workflow():
    """Use the public library that generated notebooks delegate to."""
    from skyulf.integrations.databricks.lifecycle import local_workflow

    return local_workflow


@pytest.mark.parametrize("prefix", ["event", "result"])
@pytest.mark.parametrize(
    "kind,text_kind,details",
    [
        ("timestamp", "local_datetime", set()),
        ("local_timestamp", "local_datetime", {"time_timezone"}),
        ("date", "local_datetime", {"time_timezone", "date_only"}),
        ("text", "offset_datetime", {"text_kind", "time_format"}),
        ("text", "local_datetime", {"text_kind", "time_format", "time_timezone"}),
        ("text", "date", {"text_kind", "time_format", "time_timezone", "date_only"}),
    ],
)
def test_date_questions_follow_declared_representation(prefix, kind, text_kind, details):
    """Users should only answer parsing questions relevant to their declared column format."""
    from jsonschema import Draft7Validator

    properties = json.loads((WORKFLOW.parents[4] / "databricks_template_schema.json").read_text())[
        "properties"
    ]
    values = {key: spec["default"] for key, spec in properties.items()}
    values.update(split_strategy="temporal", filter_unavailable_results="true")
    values.update({f"{prefix}_time_kind": kind, f"{prefix}_text_kind": text_kind})
    assert f"{prefix}_time_kind" in properties
    visible = {
        suffix
        for suffix in ("text_kind", "time_format", "time_timezone", "date_only")
        if not Draft7Validator(properties[f"{prefix}_{suffix}"]["skip_prompt_if"]).is_valid(values)
    }
    assert visible == details


@pytest.mark.parametrize("prefix", ["event", "result"])
def test_unused_dates_hide_all_followup_questions(prefix):
    """A date-free workflow must never ask about date types or parsing details."""
    from jsonschema import Draft7Validator

    properties = json.loads((WORKFLOW.parents[4] / "databricks_template_schema.json").read_text())[
        "properties"
    ]
    values = {key: spec["default"] for key, spec in properties.items()}
    for suffix in ("time_kind", "text_kind", "time_format", "time_timezone", "date_only"):
        assert Draft7Validator(properties[f"{prefix}_{suffix}"]["skip_prompt_if"]).is_valid(values)


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_setup_sections_hide_their_details_until_selected(enabled):
    """Scheduling and cluster details appear only when the user selects them."""
    from jsonschema import Draft7Validator

    properties = json.loads((WORKFLOW.parents[4] / "databricks_template_schema.json").read_text())[
        "properties"
    ]
    values = {key: spec["default"] for key, spec in properties.items()}
    if enabled:
        values.update(cv_enabled="true", retraining_mode="scheduled", compute_mode="policy_cluster")
    for name in (
        "retraining_cron_expression",
        "retraining_timezone_id",
        "cluster_policy_name",
        "spark_version",
        "node_type_id",
        "cost_tag_key",
        "cost_tag_value",
    ):
        hidden = Draft7Validator(properties[name]["skip_prompt_if"]).is_valid(values)
        assert hidden is not enabled


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_data_settings_use_defaults_without_extra_prompts(task):
    """CV keeps advanced defaults hidden while source-window selection remains explicit."""
    from jsonschema import Draft7Validator

    properties = json.loads((WORKFLOW.parents[4] / "databricks_template_schema.json").read_text())[
        "properties"
    ]
    values = {key: spec["default"] for key, spec in properties.items()}
    values.update(task=task, cv_enabled="true", training_sample_rows="1000")
    for name in (
        "training_version",
        "test_size",
        "random_state",
        "stratify",
        "training_sample_rows",
        "training_sample_seed",
        "min_improvement",
        "risk_category",
    ):
        assert "default" in properties[name]
        assert Draft7Validator(properties[name].get("skip_prompt_if", False)).is_valid(values)
    assert properties["training_window_mode"]["default"] == "auto"
    assert not Draft7Validator(properties["training_window_mode"]["skip_prompt_if"]).is_valid(
        values
    )
    values["training_layout"] = "multi_target"
    assert Draft7Validator(properties["training_window_mode"]["skip_prompt_if"]).is_valid(values)


@pytest.mark.parametrize("train_mode", ["manual", "scheduled"])
@pytest.mark.parametrize("score_mode", ["manual", "scheduled"])
def test_schedule_questions_follow_each_independent_mode(train_mode, score_mode):
    """Selecting a scoring clock must not expose or enable an unrelated training clock."""
    from jsonschema import Draft7Validator

    properties = json.loads((WORKFLOW.parents[4] / "databricks_template_schema.json").read_text())[
        "properties"
    ]
    values = {key: spec["default"] for key, spec in properties.items()}
    values.update(retraining_mode=train_mode, scoring_mode=score_mode)
    for prefix, mode in (("retraining", train_mode), ("scoring", score_mode)):
        assert properties[f"{prefix}_pause_status"]["default"] == "UNPAUSED"
        for suffix in ("cron_expression", "timezone_id", "pause_status"):
            assert Draft7Validator(properties[f"{prefix}_{suffix}"]["skip_prompt_if"]).is_valid(
                values
            ) == (mode == "manual")


@pytest.mark.parametrize("window", ["auto", "full_snapshot", "fixed_window", "rolling_calendar"])
@pytest.mark.parametrize("strategy", ["random", "temporal"])
@pytest.mark.parametrize("availability", ["false", "true"])
def test_window_control_questions_follow_active_data_policies(window, strategy, availability):
    """Holdout and result maturity prompts apply only to the policies that use them."""
    from jsonschema import Draft7Validator

    properties = json.loads((WORKFLOW.parents[4] / "databricks_template_schema.json").read_text())[
        "properties"
    ]
    values = {key: spec["default"] for key, spec in properties.items()}
    values.update(
        training_window_mode=window,
        split_strategy=strategy,
        filter_unavailable_results=availability,
    )
    holdout_hidden = Draft7Validator(properties["holdout_months"]["skip_prompt_if"]).is_valid(
        values
    )
    lag_hidden = Draft7Validator(
        properties["result_availability_lag_hours"]["skip_prompt_if"]
    ).is_valid(values)
    assert holdout_hidden == (strategy != "temporal" or window not in ("auto", "rolling_calendar"))
    assert lag_hidden == (availability == "false")


@pytest.mark.parametrize("window", ["auto", "full_snapshot", "fixed_window", "rolling_calendar"])
@pytest.mark.parametrize("strategy", ["random", "temporal"])
def test_explicit_boundary_questions_only_apply_to_fixed_windows(window, strategy):
    """Rolling runs derive boundaries, so setup must not ask for dates it will replace."""
    from jsonschema import Draft7Validator

    properties = json.loads((WORKFLOW.parents[4] / "databricks_template_schema.json").read_text())[
        "properties"
    ]
    values = {key: spec["default"] for key, spec in properties.items()}
    values.update(training_window_mode=window, split_strategy=strategy)
    for field in ("start", "holdout_start", "cutoff"):
        visible = window == "fixed_window" and (field != "holdout_start" or strategy == "temporal")
        hidden = Draft7Validator(properties[field]["skip_prompt_if"]).is_valid(values)
        assert hidden is not visible


def _output():
    """Inspect output publication independently of notebook widgets."""
    from skyulf.integrations.databricks.scoring.shared import prediction_output

    return prediction_output


def _render_default_config(
    record_key="entity_id", risk_category="", inference_mode="local", compute_mode="serverless"
):
    """Resolve default branches for offline checks; real CLI tests cover Go rendering."""
    template = WORKFLOW.parents[2] / "config/workflow.json.tmpl"
    schema = json.loads(
        (WORKFLOW.parents[4] / "databricks_template_schema.json").read_text(encoding="utf-8")
    )
    values = {key: spec["default"] for key, spec in schema["properties"].items()}
    values.update(
        project_name="customer_model",
        record_key_columns=record_key,
        risk_category=risk_category,
        inference_mode=inference_mode,
        compute_mode=compute_mode,
    )
    content = template.read_text(encoding="utf-8")
    # These optional fields are absent for this helper's default full-snapshot
    # configuration. Actual CLI cases cover their selected, non-default forms.
    for name in (
        "lookback_days",
        "holdout_days",
        "window_timezone",
        "holdout_months",
        "result_availability_lag_hours",
        "cv_group_column",
        "cv_test_size",
        "cv_max_train_size",
        "monthly_lookback_months",
        "event_column",
        "result_available_at_column",
        "start",
        "holdout_start",
        "cutoff",
        "result_cutoff",
    ):
        content = re.sub(r'{{if [^{}]+}}  "' + name + r'": [^\n]*,\n{{end}}', "", content)
    content = re.sub(r'{{if eq \.shap_enabled "true"}}.*?{{end}}', "", content, flags=re.DOTALL)
    content = (
        content[content.index("{\n") :]
        .replace("{{$window}}", "full_snapshot")
        .replace("{{$split}}", "random")
        .replace("{{$cv_enabled}}", "false")
    )
    content = content.replace(
        '{{if $competition}}  "competition_max_trials": {{.competition_max_trials}},\n'
        '  "competition_max_candidates": 8,\n{{end}}',
        "",
    )
    content = content.replace(
        '{{if or (eq .cv_type "time_series_split") '
        '(and (eq .cv_type "nested_cv") (eq .cv_nested_type "time_series_split"))}}false'
        '{{else if eq .cv_type "shuffle_split"}}true{{else}}{{.cv_shuffle}}{{end}}',
        "true",
    )
    worker_environment = (
        "local" if inference_mode == "spark" and compute_mode == "serverless" else "virtualenv"
    )
    content = content.replace(
        '{{if and (eq .inference_mode "spark") (eq .compute_mode "serverless")}}'
        "local{{else}}virtualenv{{end}}",
        worker_environment,
    )
    # This lightweight resolver checks non-model defaults. The actual Go CLI
    # exercises the conditional Basic/Advanced modeling block separately.
    content = re.sub(
        r'    "modeling": .*?(?=\n  }\n}\s*$)',
        '    "modeling": {"type": "linear_regression", "params": {}}',
        content,
        flags=re.DOTALL,
    )
    content = content.replace(
        '{{if eq $window "rolling_days"}}{{.lookback_days}}{{else}}null{{end}}',
        "null",
    ).replace(
        '{{if and (eq $window "rolling_days") (eq $split "temporal")}}'
        "{{.holdout_days}}{{else}}null{{end}}",
        "null",
    )
    content = content.replace(
        '{{if and (eq $window "rolling_calendar") (eq $split "temporal")}}'
        "{{.holdout_months}}{{else}}null{{end}}",
        "null",
    ).replace(
        '{{if eq .filter_unavailable_results "true"}}'
        "{{.result_availability_lag_hours}}{{else}}null{{end}}",
        "null",
    )
    content = content.replace(
        "{{if .prediction_table_name}}{{.prediction_table_name}}{{else}}{{.project_name}}_predictions{{end}}",
        "{{.project_name}}_predictions",
    )
    for name in (
        "risk_category",
        "event_time_format",
        "event_time_timezone",
        "result_time_format",
        "result_time_timezone",
        "event_column",
        "cv_group_column",
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
            '{{if eq .task "classification"}}{{.classification_metric}}{{else}}{{.regression_metric}}{{end}}',
            "heldout_rmse",
        )
        .replace(
            '{{if eq .task "classification"}}{{.classification_model}}{{else}}{{.regression_model}}{{end}}',
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
    content = content.replace(
        '{{if eq $window "rolling_calendar"}}{{.monthly_lookback_months}}{{else}}null{{end}}',
        "null",
    ).replace(
        '{{if eq $window "rolling_calendar"}}"{{.window_timezone}}"{{else}}null{{end}}',
        "null",
    )
    for name in ("record_key_columns", "input_columns"):
        expression = (
            '[{{range $i, $column := (regexp "[A-Za-z_][A-Za-z0-9_]*").FindAllString .'
            + name
            + ' -1}}{{if $i}}, {{end}}"{{$column}}"{{end}}]'
        )
        content = content.replace(
            expression, json.dumps([v.strip() for v in values[name].split(",")])
        )
    content = content.replace(
        '{{if .cv_inner_folds}}  "cv_inner_folds": {{.cv_inner_folds}},\n{{end}}',
        f'  "cv_inner_folds": {values["cv_inner_folds"]},' if values["cv_inner_folds"] else "",
    )
    for name, value in values.items():
        content = content.replace("{{." + name + "}}", str(value))
    return json.loads(content)


@pytest.mark.parametrize("inference_mode", ["local", "spark"])
@pytest.mark.parametrize("compute_mode", ["serverless", "policy_cluster"])
def test_generated_worker_environment_follows_compute_and_inference_mode(
    inference_mode, compute_mode
):
    """Serverless Spark must use its declared task environment while other routes stay isolated."""
    config = _render_default_config(inference_mode=inference_mode, compute_mode=compute_mode)
    expected = (
        "local" if inference_mode == "spark" and compute_mode == "serverless" else "virtualenv"
    )
    assert config["spark_udf_env_manager"] == expected


@pytest.mark.parametrize("risk_category", ["", "Low", "High"])
def test_bundle_risk_category_is_optional_and_configurable(risk_category):
    """Initialization must preserve the chosen business label without inventing a default risk."""
    root = WORKFLOW.parents[4]
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
    root = WORKFLOW.parents[4]
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
    root = WORKFLOW.parents[4]
    schema = json.loads((root / "databricks_template_schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["record_key_columns"]["default"] == "entity_id"

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
    resources = WORKFLOW.parents[2] / "resources"
    template = "\n".join(
        (resources / f"{name}.job.yml.tmpl").read_text(encoding="utf-8")
        for name in ("train", "score")
    )
    assert re.findall(r"^    ([a-z_]+):$", template, flags=re.MULTILINE) == ["train", "score"]
    for name in ("train", "score"):
        job = (resources / f"{name}.job.yml.tmpl").read_text(encoding="utf-8")
        assert re.search(r"^      max_concurrent_runs: 1$", job, re.MULTILINE)
    assert 'if eq .retraining_mode "scheduled"' in template
    assert "pause_status: ${var.retraining_pause_status}" in template
    assert "          default: train\n" in template
    assert "train_monthly" not in template
    assert "quartz_cron_expression: ${var.retraining_cron_expression}" in template
    assert "timezone_id: ${var.retraining_timezone_id}" in template


def test_preview_cli_exposes_one_training_action(monkeypatch, capsys):
    """Offline preview must present the same training action for every trigger."""
    monkeypatch.setattr(sys, "argv", ["preview.py", "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(WORKFLOW.parent.parent / "tools/preview.py.tmpl"), run_name="__main__")
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert "{score,train}" in output
    assert "train_monthly" not in output


def test_init_exposes_both_model_change_modes_and_editable_training_cadence():
    """The generated Bundle must let users choose scoring semantics and cron."""
    root = WORKFLOW.parents[4]
    schema = json.loads((root / "databricks_template_schema.json").read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert properties["model_change_mode"]["enum"] == ["incremental_append", "full_rebuild"]
    assert properties["retraining_cron_expression"]["default"] == "0 0 3 3 * ?"
    assert properties["retraining_timezone_id"]["default"] == "UTC"
    bundle = (WORKFLOW.parents[2] / "deployment/variables.yml.tmpl").read_text(encoding="utf-8")
    assert 'default: "{{.retraining_cron_expression}}"' in bundle
    assert 'default: "{{.retraining_timezone_id}}"' in bundle


def test_auto_champion_init_exposes_metric_gates_and_reuses_score_job():
    """Automatic selection must be explicit and call the serialized score job."""
    root = WORKFLOW.parents[4]
    schema = json.loads((root / "databricks_template_schema.json").read_text(encoding="utf-8"))
    properties = schema["properties"]
    assert properties["score_model_selection"]["enum"] == [
        "pinned_version",
        "champion",
    ]
    assert properties["promotion_policy"]["enum"] == ["manual_approval", "automatic"]
    assert "heldout_rmse" in properties["regression_metric"]["enum"]
    assert "heldout_f1" in properties["classification_metric"]["enum"]
    assert properties["quality_threshold"]["type"] == "string"
    assert properties["quality_threshold"]["default"] == "null"
    config = (WORKFLOW.parents[2] / "config/workflow.json.tmpl").read_text(encoding="utf-8")
    assert '"score_model_selection": "{{.score_model_selection}}"' in config
    assert _render_default_config()["metric"] == "heldout_rmse"
    jobs = "\n".join(
        (WORKFLOW.parents[2] / f"resources/{name}.job.yml.tmpl").read_text(encoding="utf-8")
        for name in ("train", "score")
    )
    assert "job_id: ${resources.jobs.score.id}" in jobs
    assert "task_key: run_batch_scoring" in jobs
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
        key not in config
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
