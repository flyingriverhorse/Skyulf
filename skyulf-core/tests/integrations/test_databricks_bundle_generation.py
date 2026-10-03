"""Opt-in real CLI generation checks for the deployable two-job operator graph."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROFILE = os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE")
CLI = shutil.which("databricks")
pytestmark = pytest.mark.skipif(
    not PROFILE or not CLI,
    reason="Set SKYULF_BUNDLE_CLI_TEST_PROFILE to explicitly opt into installed CLI generation.",
)


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
@pytest.mark.parametrize("enabled,shap", [("false", "false"), ("true", "false"), ("true", "true")])
def test_evaluation_charts_are_an_optional_independent_leaf(
    tmp_path, layout, compute, enabled, shap
):
    """The opt-in adds one plotting task without changing promotion dependencies or duplicating packages."""
    project = _generate_project(
        tmp_path,
        training_layout=layout,
        compute_mode=compute,
        evaluation_charts_enabled=enabled,
        shap_enabled=shap,
    )
    config = _read_validated_config(project)
    assert config["evaluation_charts"]["enabled"] is (enabled == "true")
    jobs = _read_jobs(project)
    tasks = {task["task_key"]: task for task in jobs["train"]["tasks"]}
    assert ("generate_charts" in tasks) is (enabled == "true")
    assert not any(
        dependency["task_key"] == "generate_charts"
        for task in tasks.values()
        for dependency in task.get("depends_on", [])
    )
    dependencies = (
        jobs["train"].get("environments", [{}])[0].get("spec", {}).get("dependencies", [])
        if compute == "serverless"
        else [item.get("pypi", {}).get("package") for item in tasks["initialize_run"]["libraries"]]
    )
    assert dependencies.count("matplotlib==3.10.0") == int(enabled == "true" or shap == "true")
    if enabled == "true":
        predecessor = "evaluate_model_set" if layout == "multi_target" else "evaluate_model"
        task = tasks["generate_charts"]
        assert task["depends_on"] == [{"task_key": predecessor}]
        assert task["max_retries"] == 0
        assert task["notebook_task"]["base_parameters"]["workspace_host"] == "${workspace.host}"
        notebook = (project / "src/jobs/generate_charts.py").read_text()
        assert 'run_evaluation_charts_notebook(globals()["dbutils"])' in notebook
        assert "displayHTML" not in notebook
        assert (
            task["notebook_task"]["base_parameters"]["reference_json"]
            == "{{tasks." + predecessor + ".values.reference_json}}"
        )
    assert "generate_charts" not in {task["task_key"] for task in jobs["score"]["tasks"]}


def _initialize_project(tmp_path, *, omit_fields=(), **overrides):
    """Render the actual template through the installed CLI without cloud writes."""
    root = Path(__file__).resolve().parents[2] / "templates/databricks"
    schema = json.loads((root / "databricks_template_schema.json").read_text())
    values = {key: spec["default"] for key, spec in schema["properties"].items()}
    values.update(project_name="sm33_generated", **overrides)
    for name in omit_fields:
        values.pop(name)
    inputs = tmp_path / "init.json"
    inputs.write_text(json.dumps(values), encoding="utf-8")
    generated = subprocess.run(
        [
            str(CLI),
            "bundle",
            "init",
            str(root),
            "--config-file",
            str(inputs),
            "--output-dir",
            str(tmp_path / "output"),
            "--profile",
            str(PROFILE),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return generated, tmp_path / "output" / "sm33_generated"


def _generate_project(tmp_path, **overrides):
    """Return a successfully rendered project for its behavior checks."""
    generated, project = _initialize_project(tmp_path, **overrides)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return project


def _read_bundle(project):
    """Read the root and deployment includes without emulating CLI variable resolution."""
    bundle = yaml.safe_load((project / "databricks.yml").read_text())
    for name in ("variables", "targets"):
        path = project / "deployment" / f"{name}.yml"
        if path.is_file():
            bundle.update(yaml.safe_load(path.read_text()))
    return bundle


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
@pytest.mark.parametrize("enabled", ["false", "true"])
def test_cdf_recovery_generates_conditional_score_graph(tmp_path, layout, compute, enabled):
    """Opt-in recovery stays in the score job and both successful branches reach its report."""
    project = _generate_project(
        tmp_path,
        training_layout=layout,
        compute_mode=compute,
        auto_rebuild_on_cdf_expiry=enabled,
    )
    config = _read_validated_config(project)
    assert config["auto_rebuild_on_cdf_expiry"] is (enabled == "true")
    jobs = _read_jobs(project)
    assert set(jobs) == {"train", "score"}
    tasks = {task["task_key"]: task for task in jobs["score"]["tasks"]}
    if enabled == "false":
        assert set(tasks) == {
            "score",
            "monitoring_allowed",
            "monitor_model",
            "drift_report",
            "check_retraining",
            "retraining_needed",
            "retrain_on_drift",
            "retraining_skipped",
        }
        return
    assert set(tasks) == {
        "score",
        "recovery_needed",
        "recover_predictions",
        "scoring_report",
        "monitoring_allowed",
        "monitor_model",
        "drift_report",
        "retrain_on_drift",
        "check_retraining",
        "retraining_needed",
        "retraining_skipped",
    }
    assert tasks["recovery_needed"]["depends_on"] == [{"task_key": "score"}]
    assert tasks["recovery_needed"]["condition_task"] == {
        "op": "EQUAL_TO",
        "left": "{{tasks.score.values.recovery_required}}",
        "right": "true",
    }
    assert tasks["recover_predictions"]["depends_on"] == [
        {"task_key": "recovery_needed", "outcome": "true"}
    ]
    report = tasks["scoring_report"]
    assert report["run_if"] == "NONE_FAILED"
    assert report["depends_on"] == [
        {"task_key": "recovery_needed", "outcome": "false"},
        {"task_key": "recover_predictions"},
    ]
    for key in ("score", "recover_predictions", "scoring_report"):
        task = tasks[key]
        parameters = task["notebook_task"]["base_parameters"]
        assert parameters["workflow_contract"] == "3"
        assert parameters["deployed_auto_rebuild_on_cdf_expiry"] == enabled
        for name in (
            "config_path",
            "catalog",
            "input_schema",
            "output_schema",
            "metadata_schema",
            "resource_suffix",
        ):
            assert parameters[name] == tasks["score"]["notebook_task"]["base_parameters"][name]
        assert task["timeout_seconds"] == "${var.score_task_timeout_seconds}"
        assert task["max_retries"] == "${var.score_max_retries}"
        assert task["min_retry_interval_millis"] == "${var.score_min_retry_interval_millis}"
        assert task["retry_on_timeout"] is False
        if compute == "serverless":
            assert task["disable_auto_optimization"] is True
            assert task["environment_key"] == "skyulf"
        else:
            assert task["job_cluster_key"] == "skyulf"
            assert task["libraries"] == tasks["score"]["libraries"]
    for key, entrypoint in (
        ("recover_predictions", "run_cdf_recovery_notebook"),
        ("scoring_report", "run_scoring_report_notebook"),
    ):
        notebook = (project / "src/jobs" / f"{key}.py").read_text()
        assert entrypoint in notebook
        assert "exit_notebook=False" in notebook
        assert 'display_html=globals().get("displayHTML")' in notebook
        assert notebook.index("# COMMAND ----------") < notebook.index(".notebook.exit(output)")


def test_deployment_defaults_are_ci_independent(tmp_path):
    """Shared environments require explicit selection without company automation."""
    project = _generate_project(tmp_path)
    root = yaml.safe_load((project / "databricks.yml").read_text())
    assert "deployment/targets.yml" in root["include"]
    assert "deployment/variables.yml" in root["include"]
    bundle = _read_bundle(project)
    assert set(bundle["targets"]) == {"test", "syst", "prod"}
    assert not any(v.get("default") for v in bundle["targets"].values())
    assert not (project / ".github/workflows").exists()
    for env in ("test", "syst", "prod"):
        target = bundle["targets"][env]
        assert target["mode"] == "production"
        assert target["permissions"] == []
        assert target["variables"]["resource_suffix"] == ""
        assert target["workspace"]["root_path"] == f"${{var.{env}_root_path}}"
        path = bundle["variables"][f"{env}_root_path"]["default"]
        assert path.startswith("/Workspace/Projects/")
        assert "current_user" not in path
        assert "resources" not in target  # Identity/ACL ownership remains external by default.


@pytest.mark.parametrize("prefill", [False, True])
def test_catalog_and_schema_generate_without_questions(tmp_path, prefill):
    """CI initialization must generate editable environment bindings without prompting."""
    root = Path(__file__).resolve().parents[2] / "templates/databricks"
    properties = json.loads((root / "databricks_template_schema.json").read_text())["properties"]
    for key in ("catalog", "schema"):
        assert properties[key]["skip_prompt_if"] == {}
    if prefill:
        project = _generate_project(tmp_path, catalog="existing_catalog", schema="existing_schema")
    else:
        project = _generate_project(tmp_path, omit_fields=("catalog", "schema"))
    bundle = _read_bundle(project)
    expected_catalog = "existing_catalog" if prefill else "REPLACE_TEST_CATALOG"
    expected_schema = "existing_schema" if prefill else "REPLACE_TEST_SCHEMA"
    assert bundle["variables"]["test_catalog"]["default"] == expected_catalog
    for key in ("input_schema", "output_schema", "metadata_schema"):
        assert bundle["variables"][f"test_{key}"]["default"] == expected_schema
        assert bundle["targets"]["test"]["variables"][key] == f"${{var.test_{key}}}"
    assert bundle["targets"]["test"]["variables"]["catalog"] == "${var.test_catalog}"


@pytest.mark.parametrize(
    "identity", ["deployer", "shared_service_principal", "separate_service_principals"]
)
def test_personal_targets_keep_outputs_and_identity_separate(tmp_path, identity):
    """Personal trials must not inherit shared writers, output names or enabled schedules."""
    project = _generate_project(
        tmp_path,
        personal_development_targets="true",
        deployment_identity=identity,
        manage_job_permissions="true",
        retraining_mode="scheduled",
        scoring_mode="scheduled",
        retraining_pause_status="UNPAUSED",
        scoring_pause_status="UNPAUSED",
    )
    bundle = _read_bundle(project)
    assert list(bundle["targets"]) == [
        "test",
        "test_development",
        "syst",
        "syst_development",
        "prod",
        "prod_development",
    ]
    assert [key for key, value in bundle["targets"].items() if value.get("default")] == [
        "test_development"
    ]
    for env in ("test", "syst", "prod"):
        shared = bundle["targets"][env]
        personal = bundle["targets"][f"{env}_development"]
        assert personal["workspace"]["host"] == shared["workspace"]["host"]
        assert personal["mode"] == "development"
        assert shared["permissions"] == personal["permissions"] == []
        for key in ("input_schema", "output_schema", "metadata_schema"):
            variable = f"{env}_development_{key}"
            assert personal["variables"][key] == f"${{var.{variable}}}"
            assert variable in bundle["variables"]
            assert personal["variables"][key] != shared["variables"][key]
        assert "${workspace.current_user.userName}" in personal["workspace"]["root_path"]
        assert (
            personal["variables"]["resource_suffix"] == f"_{env}_dev_${{workspace.current_user.id}}"
        )
        assert personal["variables"]["retraining_pause_status"] == "PAUSED"
        assert personal["variables"]["scoring_pause_status"] == "PAUSED"
        assert personal["resources"]["jobs"] == {job: {"name": job} for job in ("train", "score")}
        assert "run_as" not in personal
        for job in ("train", "score"):
            resource = shared["resources"]["jobs"][job]
            assert resource["permissions"] == f"${{var.{job}_permissions}}"
            if identity == "deployer":
                assert "run_as" not in resource
            else:
                role = "shared" if identity == "shared_service_principal" else job
                assert resource["run_as"] == {
                    "service_principal_name": f"${{var.{role}_service_principal}}"
                }


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_smoke_renders_without_cloud_operations(tmp_path, layout):
    """Every layout must retain runnable offline checks without cloud operations."""
    project = _generate_project(tmp_path, training_layout=layout)
    result = subprocess.run(
        [sys.executable, "src/tools/smoke.py"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "passed"
    assert report["project_hooks_executed"] is False
    assert report["remote_operations"] is False


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
@pytest.mark.parametrize("enabled", ["false", "true"])
def test_shap_generation_matches_training_dependencies(tmp_path, layout, compute, enabled):
    """Every layout must opt in consistently without adding explanation packages to scoring."""
    import runpy

    project = _generate_project(
        tmp_path,
        training_layout=layout,
        compute_mode=compute,
        shap_enabled=enabled,
        shap_max_samples="4",
        shap_max_display_samples="2",
    )
    workflow = json.loads((project / "config/workflow.json").read_text())
    pipelines = [workflow["pipeline"]]
    if layout == "multi_target":
        module = runpy.run_path(str(project / "src/modeling/multi_model.py"))
        pipelines.extend(item["workflow"]["pipeline"] for item in module["MODELS"].values())
    for pipeline in pipelines:
        if enabled == "true":
            assert pipeline["explainability"] == {
                "method": "shap",
                "max_samples": 4,
                "max_features": 30,
                "max_display_samples": 2,
            }
        else:
            assert "explainability" not in pipeline
    job = yaml.safe_load((project / "resources/train.job.yml").read_text())
    serialized = json.dumps(job)
    assert ("shap==0.49.1" in serialized) == (enabled == "true")
    assert ("matplotlib==3.10.0" in serialized) == (enabled == "true")
    assert "shap==" not in (project / "resources/score.job.yml").read_text()


@pytest.mark.parametrize("mode", ["all", "combined_only", "separate_views"])
def test_multi_target_output_selection_renders_custom_destinations(tmp_path, mode):
    """Initializer choices must produce callable settings with usable custom consumer names."""
    from skyulf.inference.project_code import load_project_module
    from skyulf.integrations.databricks.model_set_project import (
        capture_set_rules,
        load_project_model_set,
    )

    project = _generate_project(
        tmp_path,
        training_layout="multi_target",
        model_set_name="business_models",
        source_change_policy="rebuild_on_change",
        model_set_output_mode=mode,
        model_set_table_name="business_predictions",
        model_view_prefix="estimates",
        combined_view_name="profit_results",
    )
    values = {
        "config_path": str(project / "config/workflow.json"),
        "catalog": "workspace",
        "input_schema": "default",
        "output_schema": "default",
        "metadata_schema": "default",
        "resource_suffix": "_dev",
    }
    settings = load_project_model_set(
        values, json.loads((project / "config/workflow.json").read_text())
    )
    assert settings is not None
    assert settings["model_name"] == "workspace.default.business_models_dev"
    assert settings["source_change_policy"] == "rebuild_on_change"
    assert settings["prediction_table"] == "workspace.default.business_predictions_dev"
    assert settings["publication"]["mode"] == mode
    captured, source = capture_set_rules(values, settings)
    assert captured["composition_config"] == {"outputs": []}
    assert load_project_module(source).build_scoring()["reuse_pre_split"] is True
    if mode == "separate_views":
        assert (
            settings["publication"]["model_view_template"]
            == "workspace.default.estimates_{branch}_dev"
        )
        assert settings["publication"]["combined_view"] == "workspace.default.profit_results_dev"
    assert not (project / "src/composition").exists()


def test_cli_multi_target_setup_defaults_omitted_branch_settings(tmp_path):
    """Explicit target selections must initialize model settings from their branch defaults."""
    from test_databricks_layout_prompts import SHARED_FIELDS

    root = Path(__file__).resolve().parents[2] / "templates/databricks"
    properties = json.loads((root / "databricks_template_schema.json").read_text())["properties"]
    project = _generate_project(
        tmp_path,
        training_layout="multi_target",
        omit_fields={
            name
            for name in properties
            if name not in SHARED_FIELDS and not name.startswith("branch_")
        },
    )
    config = _read_validated_config(project, action="train")
    assert config["task"] == "regression"
    assert config["target_column"] == "target"
    assert config["input_columns"] == ["feature_value"]
    assert config["split_strategy"] == "random" and config["cv_enabled"] is False
    assert config["promotion_policy"] == "manual_approval"
    assert config["score_handoff"] == "disabled"
    jobs = _read_jobs(project)
    task_names = {task["task_key"] for task in jobs["train"]["tasks"]}
    assert {
        "initialize_run",
        "choose_action",
        "register_model_set",
        "evaluate_model_set",
        "model_decision",
        "training_report",
    } <= task_names
    assert len([name for name in task_names if name.startswith("train_")]) == 2
    assert (
        jobs["score"]["tasks"][0]["notebook_task"]["notebook_path"] == "../src/jobs/score_models.py"
    )
    assert "schedule" not in jobs["score"]


def _read_jobs(project):
    """Keep each job independently parseable with stable resource keys and no duplicates."""
    jobs = {}
    resources = project / "resources"
    assert {path.name for path in resources.glob("*.yml")} == {"train.job.yml", "score.job.yml"}
    for name in ("train", "score"):
        document = yaml.safe_load((resources / f"{name}.job.yml").read_text())
        resource_jobs = document["resources"]["jobs"]
        assert set(resource_jobs) == {name}
        jobs.update(resource_jobs)
    return jobs


def _synced_sources(project, bundle):
    """Expand source patterns so notebook and custom-module coverage stays explicit."""
    sources = set()
    for pattern in bundle["sync"]["include"]:
        if pattern.startswith("src/"):
            matches = list(project.glob(pattern))
            assert matches and all(path.is_file() for path in matches), pattern
            sources.update(path.relative_to(project).as_posix() for path in matches)
    return sources


def test_generated_source_layout_connects_both_custom_recipes(tmp_path):
    """Generated relative imports, preview, hooks and all job paths must agree."""
    from skyulf.integrations.databricks.project import load_project_workflow
    from skyulf.preprocessing.function_steps import FITTED_STEP

    project = _generate_project(tmp_path)
    assert {path.name for path in (project / "src").iterdir()} == {
        "jobs",
        "features",
        "modeling",
        "tools",
    }
    for job in _read_jobs(project).values():
        for task in job["tasks"]:
            if "notebook_task" in task:
                path = project / "resources" / task["notebook_task"]["notebook_path"]
                assert path.is_file(), path
    features = project / "src/features"
    config = _read_validated_config(project)
    baseline = load_project_workflow(config, features)
    assert baseline["pre_split_steps"] == baseline["pipeline"]["preprocessing"] == []
    for filename, factory, shown, configured in (
        (
            "pre_split.py",
            "minimum_completeness",
            'minimum_completeness(columns=["field_a", "field_b", "field_c"], min_present=2)',
            'minimum_completeness(columns=["quality_a", "quality_b"], min_present=1)',
        ),
        (
            "preprocessing.py",
            "frequency_encoding",
            'frequency_encoding(columns=["category"])',
            'frequency_encoding(columns=["category"])',
        ),
    ):
        path = features / filename
        text = path.read_text(encoding="utf-8")
        module = filename.removesuffix(".py") + "_custom"
        text = text.replace(
            f"# from .custom.{module} import {factory}", f"from .custom.{module} import {factory}"
        )
        assert f"# {shown}," in text
        path.write_text(text.replace(f"# {shown},", f"{configured},"), encoding="utf-8")
    # Default scoring reuses pre-split checks, so their inputs must be declared too.
    config["input_columns"] = ["category", "amount", "quality_a", "quality_b"]
    (project / "config/workflow.json").write_text(json.dumps(config), encoding="utf-8")
    enabled = load_project_workflow(config, features)
    assert enabled["pre_split_steps"][0]["pre_split"]["effect"] == "filter"
    frequency_step = enabled["pipeline"]["preprocessing"][0]
    assert frequency_step["transformer"] == FITTED_STEP
    assert frequency_step["params"]["output"] == ["category"]
    assert frequency_step["params"]["params"]["columns"] == ["category"]
    result = subprocess.run(
        [sys.executable, str(project / "src/tools/preview.py"), "--action", "train"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_basic_setup_defaults_match_explicit_advanced_settings(tmp_path):
    """Skipped tuning prompts must use defaults without losing explicit init-file values."""
    basic = tmp_path / "basic"
    explicit = tmp_path / "explicit"
    basic.mkdir()
    explicit.mkdir()
    advanced = (
        "training_version",
        "test_size",
        "random_state",
        "stratify",
        "training_window_mode",
        "training_sample_rows",
        "training_sample_seed",
        "cv_folds",
        "cv_type",
        "cv_shuffle",
        "cv_random_state",
        "min_improvement",
        "risk_category",
    )
    defaults = _generate_project(basic, omit_fields=advanced, cv_enabled="true")
    supplied = _generate_project(explicit, cv_enabled="true")
    assert _read_validated_config(defaults) == _read_validated_config(supplied)


def test_cli_hidden_advanced_overrides_survive_initialization(tmp_path):
    """Config-file values must take precedence even when their questions are hidden."""
    expected = {
        "training_version": 12,
        "test_size": 0.3,
        "random_state": 7,
        "stratify": True,
        "training_window_mode": "fixed_window",
        "training_sample_rows": 1000,
        "training_sample_seed": 11,
        "cv_folds": 3,
        "cv_type": "stratified_k_fold",
        "cv_shuffle": False,
        "cv_random_state": 13,
        "min_improvement": 0.1,
        "risk_category": "high",
    }
    overrides = {
        key: value if isinstance(value, str) or key == "min_improvement" else json.dumps(value)
        for key, value in expected.items()
    }
    project = _generate_project(
        tmp_path,
        **overrides,
        task="classification",
        cv_enabled="true",
        event_column="observed_at",
        start="2026-06-01T00:00:00+00:00",
        cutoff="2026-09-01T00:00:00+00:00",
    )
    config = _read_validated_config(project)
    assert {key: config[key] for key in expected} == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_rows", 0),
        ("max_input_mb", -1),
        ("max_rows", "0"),
        ("max_input_mb", "-1"),
        ("max_rows", "1.5"),
        ("max_input_mb", "1e3"),
        ("record_key_columns", "\fid"),
        ("input_columns", "x,\fy"),
        ("record_key_columns", 'id, "task": "classification"'),
    ],
)
def test_cli_rejects_invalid_limits_and_column_text(tmp_path, field, value):
    """Invalid init values must fail before writing malformed or unusable config."""
    generated, _ = _initialize_project(tmp_path, **{field: value})
    assert generated.returncode != 0
    assert field in generated.stderr


def _read_modeling(project):
    """Resolve the generated Python settings through the actual project loader."""
    from skyulf.integrations.databricks.project import load_project_workflow

    config = json.loads((project / "config/workflow.json").read_text())
    return load_project_workflow(config, project / "src/features")["pipeline"]["modeling"]


def _load_project_config(project, config):
    """Resolve the model file before checking the training contract, as notebooks do."""
    from skyulf.integrations.databricks.project import load_project_workflow

    return load_project_workflow(config, project / "src/features")


def _read_validated_config(project, *, action="score"):
    """Check generated settings against the same preflight used by the notebook."""
    from skyulf.integrations.databricks.local_workflow import resolve_target_config
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    config = json.loads((project / "config/workflow.json").read_text())
    resolved = resolve_target_config(
        config,
        {
            "catalog": "test_catalog",
            "input_schema": "inputs",
            "output_schema": "outputs",
            "metadata_schema": "metadata",
            "resource_suffix": "_dev",
        },
    )
    from skyulf.integrations.databricks.project import load_project_workflow

    loaded = load_project_workflow(resolved, project / "src/features")
    validate_workflow_config(loaded, action=action)
    return config


@pytest.mark.parametrize("train_mode", ["manual", "scheduled"])
@pytest.mark.parametrize("score_mode", ["manual", "scheduled"])
def test_cli_independent_schedules_preserve_shared_two_job_graph(tmp_path, train_mode, score_mode):
    """Each clock is optional and target-overridable without duplicating score handoff."""
    project = _generate_project(
        tmp_path,
        retraining_mode=train_mode,
        scoring_mode=score_mode,
        retraining_cron_expression="0 0 3 1 1,7 ?",
        scoring_cron_expression="0 15 * * * ?",
        retraining_timezone_id="Europe/Vilnius",
        scoring_timezone_id="America/New_York",
        retraining_pause_status="PAUSED",
        scoring_pause_status="UNPAUSED",
    )
    variables = _read_bundle(project)["variables"]
    jobs = _read_jobs(project)
    config = _read_validated_config(project)
    assert set(jobs) == {"train", "score"}
    bundle = yaml.safe_load((project / "databricks.yml").read_text())
    synced_notebooks = _synced_sources(project, bundle)
    assert all((project / path).is_file() for path in synced_notebooks)
    defaults = {entry["name"]: entry["default"] for entry in jobs["train"]["parameters"]}
    assert defaults["lifecycle_action"] == "train"
    for job, mode, prefix, cron, zone, pause in (
        ("train", train_mode, "retraining", "0 0 3 1 1,7 ?", "Europe/Vilnius", "PAUSED"),
        ("score", score_mode, "scoring", "0 15 * * * ?", "America/New_York", "UNPAUSED"),
    ):
        assert jobs[job]["max_concurrent_runs"] == 1
        assert jobs[job]["queue"]["enabled"] is True
        if mode == "manual":
            assert "schedule" not in jobs[job]
            assert f"{prefix}_pause_status" not in variables
        else:
            assert jobs[job]["schedule"] == {
                "quartz_cron_expression": "${var." + prefix + "_cron_expression}",
                "timezone_id": "${var." + prefix + "_timezone_id}",
                "pause_status": "${var." + prefix + "_pause_status}",
            }
            for suffix, expected in (
                ("cron_expression", cron),
                ("timezone_id", zone),
                ("pause_status", pause),
            ):
                assert variables[f"{prefix}_{suffix}"]["default"] == expected
                assert f"{prefix}_{suffix}" not in config
    tasks = {task["task_key"]: task for task in jobs["train"]["tasks"]}
    readme = (project / "README.md").read_text(encoding="utf-8")
    documented_contract = re.search(r'workflow_contract: "(\d+)"', readme)
    assert documented_contract is not None
    for task in jobs["train"]["tasks"] + jobs["score"]["tasks"]:
        assert task["task_key"] in readme
        if "notebook_task" in task:
            assert task["notebook_task"]["base_parameters"][
                "workflow_contract"
            ] == documented_contract.group(1)
    assert tasks["run_batch_scoring"]["run_job_task"]["job_id"] == "${resources.jobs.score.id}"
    assert tasks["train_and_tune"]["max_retries"] == 0
    assert jobs["score"]["tasks"][0]["max_retries"] == "${var.score_max_retries}"
    assert _read_bundle(project)["variables"]["score_max_retries"]["default"] == 0


@pytest.mark.parametrize("strategy,holdout", [("random", None), ("temporal", 2)])
def test_cli_emits_integer_window_controls_only_when_active(tmp_path, strategy, holdout):
    """Initializer numeric text must render as integer policy values or inactive nulls."""
    project = _generate_project(
        tmp_path,
        training_window_mode="rolling_calendar",
        split_strategy=strategy,
        event_column="observed_at",
        holdout_months="2",
        monthly_lookback_months="4",
        filter_unavailable_results="true",
        result_available_at_column="confirmed_at",
        result_availability_lag_hours="48",
    )
    config = _read_validated_config(project)
    assert config["holdout_months"] == holdout
    assert config["result_availability_lag_hours"] == 48
    assert type(config["result_availability_lag_hours"]) is int


def test_cli_default_window_controls_are_inactive(tmp_path):
    """Date-free default projects must not activate irrelevant holdout or maturity policies."""
    config = _read_validated_config(_generate_project(tmp_path))
    assert config["holdout_months"] is None
    assert config["result_availability_lag_hours"] is None


def test_cli_plain_column_lists_preserve_order_and_empty_preprocessing(tmp_path):
    """Operators can enter comma-separated names without inserting JSON syntax."""
    project = _generate_project(
        tmp_path,
        record_key_columns="customer_id, observation_id",
        input_columns="income, age, balance",
        regression_model="random_forest_regressor",
        regression_metric="heldout_mae",
    )
    config = _read_validated_config(project)
    assert config["record_key_columns"] == ["customer_id", "observation_id"]
    assert config["input_columns"] == ["income", "age", "balance"]
    assert config["pipeline"]["preprocessing"] == []
    modeling = _read_modeling(project)
    assert modeling["type"] == "hyperparameter_tuner"
    assert modeling["base_model"] == {"type": "random_forest_regressor", "params": {}}
    assert modeling["search_space"]
    assert config["metric"] == "heldout_mae"


def test_cli_named_prediction_output_is_separate_from_prediction_input(tmp_path):
    """Choosing the output name must not replace the input table or its target bindings."""
    project = _generate_project(
        tmp_path,
        score_source_table_name="customers_to_score",
        prediction_table_name="customer_predictions",
    )
    config = _read_validated_config(project)
    assert config["score_source_table"] == "{catalog}.{input_schema}.customers_to_score"
    assert (
        config["prediction_table"]
        == "{catalog}.{output_schema}.customer_predictions{resource_suffix}"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("record_key_columns", "customer_id,,event_id"),
        ("input_columns", 'income, age"'),
        ("input_columns", "income,"),
        ("regression_model", "random_forest_classifier"),
        ("classification_metric", "heldout_rmse"),
    ],
)
def test_cli_rejects_invalid_plain_columns_and_cross_task_choices(tmp_path, field, value):
    """The friendlier input must still reject malformed names and wrong-task selections."""
    generated, _ = _initialize_project(tmp_path, **{field: value})
    assert generated.returncode != 0
    assert field in generated.stderr


def test_cli_six_month_schedule_keeps_data_window_independent(tmp_path):
    """A six-month job cadence must not silently impose a six-month training window."""
    project = _generate_project(
        tmp_path,
        retraining_mode="scheduled",
        retraining_cron_expression="0 0 3 1 1,7 ?",
        retraining_timezone_id="Europe/Copenhagen",
        training_window_mode="full_snapshot",
    )
    bundle = _read_bundle(project)
    jobs = _read_jobs(project)
    assert bundle["variables"]["retraining_cron_expression"]["default"] == "0 0 3 1 1,7 ?"
    assert bundle["variables"]["retraining_timezone_id"]["default"] == "Europe/Copenhagen"
    assert jobs["train"]["schedule"]["pause_status"] == "${var.retraining_pause_status}"
    assert _read_validated_config(project)["training_window_mode"] == "full_snapshot"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_cli_guided_pipeline_cv_and_offline_preview(tmp_path, engine, task):
    """The real initializer must preserve selected Core steps/model/CV on both engines."""
    import sys

    import numpy as np
    import pandas as pd
    import polars as pl

    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.local_pipeline import load_local_pipeline, predict_local_pipeline
    from skyulf.integrations.databricks.local_batch import fit_local_workflow

    model = "random_forest_classifier" if task == "classification" else "random_forest_regressor"
    project = _generate_project(
        tmp_path,
        engine=engine,
        task=task,
        training_version="0",
        **{f"{task}_model": model},
        cv_enabled="true",
        cv_folds="3",
        cv_type="k_fold",
        search_n_trials="2",
        training_sample_rows="500",
        training_sample_seed="19",
    )
    config = _read_validated_config(project)
    jobs = _read_jobs(project)
    for role, job in jobs.items():
        for entry in job["tasks"]:
            if "notebook_task" in entry:
                assert entry["max_retries"] == (
                    "${var.score_max_retries}" if role == "score" else 0
                )
                assert entry["disable_auto_optimization"] is True
    modeling = _read_modeling(project)
    assert modeling["type"] == "hyperparameter_tuner"
    assert modeling["base_model"] == {"type": model, "params": {}}
    assert modeling["n_trials"] == 2
    assert config["pipeline"]["preprocessing"] == []
    # Operators add their own steps after initialization; test that edited path.
    steps = [
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"columns": ["feature_value"], "strategy": "mean"},
        },
        {
            "name": "scale",
            "transformer": "StandardScaler",
            "params": {"columns": ["feature_value"]},
        },
    ]
    source_path = project / "src/features/preprocessing.py"
    with source_path.open("a", encoding="utf-8") as stream:
        stream.write("\n\ndef build_preprocessing():\n    return " + repr(steps) + "\n")
    from skyulf.integrations.databricks.project import load_project_workflow

    config = load_project_workflow(config, source_path.parent)
    assert config["cv_enabled"] is True and config["cv_folds"] == 3
    assert config["training_sample_rows"] == 500 and config["training_sample_seed"] == 19
    preview = subprocess.run(
        [sys.executable, str(project / "src/tools/preview.py"), "--action", "train"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=project,
    )
    assert preview.returncode == 0, preview.stdout + preview.stderr
    assert f"Engine: {engine}" in preview.stdout
    assert model in preview.stdout
    assert "training partition only" in preview.stdout
    frame = pd.DataFrame(
        {
            "feature_value": [float(i) if i % 7 else np.nan for i in range(40)],
            "target": [i % 2 if task == "classification" else i * 2.0 for i in range(40)],
        }
    )
    data = pl.from_pandas(frame) if engine == "polars" else frame
    artifact = tmp_path / "pipeline.pkl"
    fit_local_workflow(
        config["pipeline"],
        SplitDataset(train=data, test=data[:0]),
        target_column="target",
        artifact_path=artifact,
        max_rows=100,
        max_bytes=1048576,
    )
    incoming = pd.DataFrame({"feature_value": [None, 5.0, 8.0]})
    if engine == "polars":
        incoming = pl.from_pandas(incoming)
    predictions = predict_local_pipeline(incoming, load_local_pipeline(artifact))
    assert len(predictions) == 3
    assert np.isfinite(np.asarray(predictions)).all()


def test_cli_random_window_and_non_utc_source_rules(tmp_path):
    """Random holdout and source calendar selection are independent initializer choices."""
    project = _generate_project(
        tmp_path,
        training_window_mode="fixed_window",
        split_strategy="random",
        event_column="observed_at",
        event_time_kind="text",
        event_text_kind="local_datetime",
        event_time_format="%d/%m/%Y %H:%M",
        event_time_timezone="Europe/Copenhagen",
        training_version="0",
        start="2026-06-01T00:00:00+02:00",
        cutoff="2026-09-01T00:00:00+02:00",
        filter_unavailable_results="true",
        result_available_at_column="confirmed_at",
        result_time_kind="text",
        result_text_kind="date",
        result_time_format="%Y-%m-%d",
        result_time_timezone="Europe/Vilnius",
        result_date_only="midnight",
        result_cutoff="2026-09-15T00:00:00+03:00",
    )
    config = _read_validated_config(project)
    assert config["training_window_mode"] == "fixed_window"
    assert config["monthly_lookback_months"] is None and config["window_timezone"] is None
    assert config["event_time_parsing"] == {
        "format": "%d/%m/%Y %H:%M",
        "timezone": "Europe/Copenhagen",
        "date_only": "reject",
    }
    assert config["result_time_parsing"]["date_only"] == "midnight"


@pytest.mark.parametrize(
    ("task", "model", "metric"),
    [
        ("regression", "linear_regression", "heldout_rmse"),
        ("classification", "logistic_regression", "heldout_accuracy"),
    ],
)
def test_cli_initializes_task_without_stale_training_inputs(tmp_path, task, model, metric):
    """A generated task must select compatible defaults without pinning example data."""
    project = _generate_project(tmp_path, task=task)
    config = _read_validated_config(project)
    assert config["config_version"] == 1
    assert config["task"] == task
    assert _read_modeling(project)["type"] == "hyperparameter_tuner"
    assert _read_modeling(project)["base_model"] == {"type": model, "params": {}}
    assert config["metric"] == metric
    assert config["training_version"] is None
    assert all(config[key] is None for key in ("start", "holdout_start", "cutoff"))
    assert config["record_key_columns"] == ["entity_id"]
    assert config["input_columns"] == ["feature_value"]
    assert config["training_table"] == "{catalog}.{input_schema}.sm33_generated_source"
    assert config["score_source_table"] == config["training_table"]


def test_cli_preserves_existing_sources_composite_keys_and_explicit_split(tmp_path):
    """Real source mappings and limits must survive initialization without invented columns."""
    project = _generate_project(
        tmp_path,
        task="classification",
        source_table_name="labeled_customers",
        score_source_table_name="new_customers",
        record_key_columns=" customer_id,\tobservation_id ",
        input_columns="\tincome, age ",
        target_column="churn",
        split_strategy="temporal",
        training_window_mode="fixed_window",
        filter_unavailable_results="true",
        event_column="observed_at",
        result_available_at_column="labeled_at",
        training_version="17",
        start="2026-06-01T00:00:00+00:00",
        holdout_start="2026-07-01T00:00:00+00:00",
        cutoff="2026-08-01T00:00:00+00:00",
        result_cutoff="2026-09-01T00:00:00+00:00",
        max_rows="500",
        max_input_mb="1",
        classification_metric="heldout_f1",
        quality_threshold="0.8",
    )
    config = _read_validated_config(project)
    assert config["training_table"] == "{catalog}.{input_schema}.labeled_customers"
    assert config["score_source_table"] == "{catalog}.{input_schema}.new_customers"
    assert config["record_key_columns"] == ["customer_id", "observation_id"]
    assert config["input_columns"] == ["income", "age"]
    assert config["target_column"] == "churn"
    assert config["event_column"] == "observed_at"
    assert config["result_available_at_column"] == "labeled_at"
    assert config["training_version"] == 17
    assert config["start"] == "2026-06-01T00:00:00+00:00"
    assert config["holdout_start"] == "2026-07-01T00:00:00+00:00"
    assert config["cutoff"] == "2026-08-01T00:00:00+00:00"
    assert config["result_cutoff"] == "2026-09-01T00:00:00+00:00"
    assert config["split_strategy"] == "temporal"
    assert config["filter_unavailable_results"] is True
    assert config["max_rows"] == 500 and config["max_input_mb"] == 1
    assert config["metric"] == "heldout_f1" and config["quality_threshold"] == 0.8


def test_cli_reuses_named_training_source_and_existing_single_key(tmp_path):
    """An omitted scoring table and composite key must preserve the simpler setup."""
    project = _generate_project(
        tmp_path, source_table_name="customers", record_key_columns="customer_id"
    )
    config = _read_validated_config(project)
    assert config["record_key_columns"] == ["customer_id"]
    assert config["training_table"] == "{catalog}.{input_schema}.customers"
    assert config["score_source_table"] == config["training_table"]


@pytest.mark.parametrize("selection", ["pinned_version", "champion"])
@pytest.mark.parametrize("policy", ["manual_approval", "automatic"])
@pytest.mark.parametrize("handoff", ["disabled", "after_alias_change"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
def test_cli_emits_independent_policies_and_serialized_operator_graph(
    tmp_path, selection, policy, handoff, compute
):
    """Render real Go templates so broken dynamic references or job dependencies cannot hide."""
    monthly = handoff == "after_alias_change"
    project = _generate_project(
        tmp_path,
        score_model_selection=selection,
        promotion_policy=policy,
        score_handoff=handoff,
        compute_mode=compute,
        engine="polars" if policy == "automatic" else "pandas",
        quality_threshold="100.0",
        retraining_mode="scheduled" if monthly else "manual",
    )
    jobs = _read_jobs(project)
    config = _read_validated_config(project)
    assert set(jobs) == {"train", "score"}
    for job in jobs.values():
        assert job["max_concurrent_runs"] == 1 and job["queue"]["enabled"] is True
    bundle = yaml.safe_load((project / "databricks.yml").read_text())
    synced_notebooks = _synced_sources(project, bundle)
    assert all((project / path).is_file() for path in synced_notebooks)
    tasks = {task["task_key"]: task for task in jobs["train"]["tasks"]}
    chain = [
        "load_data",
        "prepare_dataset",
        "train_and_tune",
        "validate_model",
        "register_model",
        "evaluate_model",
    ]
    assert set(tasks) == {
        "initialize_run",
        "choose_action",
        *chain,
        "model_decision",
        "training_report",
        "monitoring_allowed",
        "register_monitor",
        "scoring_requested",
        "run_batch_scoring",
    }
    assert tasks["choose_action"]["depends_on"] == [{"task_key": "initialize_run"}]
    assert tasks["choose_action"]["condition_task"] == {
        "op": "EQUAL_TO",
        "left": "{{tasks.initialize_run.values.training_requested}}",
        "right": "true",
    }
    assert tasks["load_data"]["depends_on"] == [{"task_key": "choose_action", "outcome": "true"}]
    for previous, following in zip(chain, chain[1:], strict=False):
        assert tasks[following]["depends_on"] == [{"task_key": previous}]
    assert tasks["model_decision"]["run_if"] == "NONE_FAILED"
    assert tasks["model_decision"]["depends_on"] == [
        {"task_key": "choose_action", "outcome": "false"},
        {"task_key": "evaluate_model"},
    ]
    assert tasks["training_report"]["depends_on"] == [{"task_key": "model_decision"}]
    assert tasks["training_report"]["run_if"] == "ALL_DONE"
    completion = tasks["training_report"]["notebook_task"]["base_parameters"]
    assert completion["decision_result_state"] == "{{tasks.model_decision.result_state}}"
    assert tasks["monitoring_allowed"]["depends_on"] == [{"task_key": "training_report"}]
    assert tasks["register_monitor"]["depends_on"] == [
        {"task_key": "monitoring_allowed", "outcome": "true"}
    ]
    assert tasks["scoring_requested"]["run_if"] == "NONE_FAILED"
    assert tasks["scoring_requested"]["depends_on"] == [
        {"task_key": "monitoring_allowed", "outcome": "false"},
        {"task_key": "register_monitor"},
    ]
    assert tasks["scoring_requested"]["condition_task"] == {
        "op": "EQUAL_TO",
        "left": "{{tasks.training_report.values.score_requested}}",
        "right": "true",
    }
    assert tasks["run_batch_scoring"]["depends_on"] == [
        {"task_key": "scoring_requested", "outcome": "true"}
    ]
    assert tasks["run_batch_scoring"]["run_job_task"] == {
        "job_id": "${resources.jobs.score.id}",
        "job_parameters": {"score_model_version": ""},
    }
    defaults = {item["name"]: item["default"] for item in jobs["train"]["parameters"]}
    assert defaults["lifecycle_action"] == "train"
    assert set(defaults) == {
        "lifecycle_action",
        "candidate_version",
        "expected_champion_version",
        "rejection_reason",
        "promotion_receipt_json",
    }
    parameters = tasks["initialize_run"]["notebook_task"]["base_parameters"]
    for name in defaults:
        assert parameters[name] == "{{job.parameters." + name + "}}"
    score_task = jobs["score"]["tasks"][0]
    assert score_task["notebook_task"]["notebook_path"] == "../src/jobs/score.py"
    assert "lifecycle_action" not in score_task["notebook_task"]["base_parameters"]
    assert jobs["score"]["parameters"] == [{"name": "score_model_version", "default": ""}]
    score_parameters = score_task["notebook_task"]["base_parameters"]
    assert score_parameters["score_model_version"] == "{{job.parameters.score_model_version}}"
    for notebook_parameters in (parameters, score_parameters):
        assert notebook_parameters["workflow_contract"] == "3"
        assert notebook_parameters["deployed_score_handoff"] == handoff
    assert (project / "src/jobs/score.py").is_file()
    assert config["score_model_selection"] == selection
    assert config["promotion_policy"] == policy
    assert config["score_handoff"] == handoff
    assert config["quality_threshold"] == 100.0  # Manual gates must not be erased.
    assert "model_selection_mode" not in config
    if monthly:
        assert jobs["train"]["schedule"]["pause_status"] == "${var.retraining_pause_status}"
    else:
        assert "schedule" not in jobs["train"]
    if compute == "serverless":
        assert tasks["train_and_tune"]["environment_key"] == "skyulf"
        assert "job_clusters" not in jobs["train"]
    else:
        assert tasks["train_and_tune"]["job_cluster_key"] == "skyulf"
        assert "environments" not in jobs["train"]
    for task in tasks.values():
        assert task["max_retries"] == 0
        if "notebook_task" in task:
            notebook = task["notebook_task"]
            assert (project / "resources" / notebook["notebook_path"]).is_file()
            assert notebook["notebook_path"].removeprefix("../") in synced_notebooks
            runtime = notebook["base_parameters"]
            assert runtime["job_id"] == "{{job.id}}"
            assert runtime["job_run_id"] == "{{job.run_id}}"
            assert runtime["repair_count"] == "{{job.repair_count}}"
            assert runtime["execution_count"] == "{{task.execution_count}}"
            if task["task_key"] != "initialize_run":
                assert runtime["tracking_uri"] == "{{tasks.initialize_run.values.tracking_uri}}"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("availability", [False, True])
def test_cli_generates_date_free_training_contract(tmp_path, engine, task, availability):
    """The real initializer must support date-free training and independent delayed results."""
    from skyulf.integrations.databricks.local_workflow import resolve_target_config
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    project = _generate_project(
        tmp_path,
        engine=engine,
        task=task,
        training_version="7",
        filter_unavailable_results="true" if availability else "false",
        result_available_at_column="confirmed_at" if availability else "",
        result_cutoff="2026-09-01T00:00:00+00:00" if availability else "",
        stratify="true" if task == "classification" else "false",
    )
    config = _read_validated_config(project)
    resolved = resolve_target_config(
        config,
        {
            "catalog": "workspace",
            "input_schema": "inputs",
            "output_schema": "outputs",
            "metadata_schema": "metadata",
            "resource_suffix": "",
        },
    )
    validate_workflow_config(_load_project_config(project, resolved), action="train")
    assert config["split_strategy"] == "random"
    assert config["training_window_mode"] == "full_snapshot"
    assert config["window_timezone"] is None
    assert config["cv_enabled"] is False and config["cv_folds"] == 5
    assert config["training_sample_rows"] is None
    assert config["engine"] == engine
    assert config["test_size"] == 0.2 and config["random_state"] == 42
    assert config["stratify"] == (task == "classification")
    assert config["event_column"] is None
    assert all(
        config[key] is None
        for key in ("start", "holdout_start", "cutoff", "monthly_lookback_months")
    )
    assert config["filter_unavailable_results"] == availability
    assert config["result_available_at_column"] == ("confirmed_at" if availability else None)


def test_cli_preserves_conflicting_fields_for_preflight_rejection(tmp_path):
    """Initialization must not silently discard a supplied date mapping in random mode."""
    from skyulf.integrations.databricks.local_workflow import resolve_target_config
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    project = _generate_project(tmp_path, event_column="observed_at", training_version="7")
    config = json.loads((project / "config/workflow.json").read_text())
    assert config["event_column"] == "observed_at"
    resolved = resolve_target_config(
        config,
        {
            "catalog": "workspace",
            "input_schema": "inputs",
            "output_schema": "outputs",
            "metadata_schema": "metadata",
            "resource_suffix": "",
        },
    )
    with pytest.raises(ValueError, match="event_column|[Rr]andom"):
        validate_workflow_config(_load_project_config(project, resolved), action="train")


@pytest.mark.parametrize(
    "filename",
    [
        "date-free-init.example.json",
        "random-delayed-results-init.example.json",
        "temporal-delayed-results-init.example.json",
        "guided-classification-init.example.json",
        "random-window-init.example.json",
        "serverless-init.example.json",
        "policy-init.example.json",
        "paying-reg-no-init.example.json",
    ],
)
def test_cli_training_examples_pass_manual_preflight(tmp_path, filename):
    """Published examples must render and validate as usable manual training policies."""
    from skyulf.integrations.databricks.local_workflow import resolve_target_config
    from skyulf.integrations.databricks.workflow_config import validate_workflow_config

    root = Path(__file__).resolve().parents[2] / "templates/databricks/examples"
    inputs = json.loads((root / filename).read_text())
    inputs.pop("project_name")
    project = _generate_project(tmp_path, **inputs)
    config = _read_validated_config(project)
    resolved = resolve_target_config(
        config,
        {
            "catalog": "workspace",
            "input_schema": "inputs",
            "output_schema": "outputs",
            "metadata_schema": "metadata",
            "resource_suffix": "",
        },
    )
    checked = validate_workflow_config(_load_project_config(project, resolved), action="train")
    assert checked["training_version"] == json.loads(inputs.get("training_version", "null"))
    assert checked["split_strategy"] == inputs.get("split_strategy", "random")
    if inputs.get("start"):
        assert checked["training_window_mode"] == "fixed_window"
    preview = subprocess.run(
        [sys.executable, str(project / "src/tools/preview.py"), "--action", "train"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert preview.returncode == 0, preview.stdout + preview.stderr
    assert f"Engine: {inputs['engine']}" in preview.stdout


def test_generated_nested_search_preserves_separate_inner_folds(tmp_path):
    """The real CLI must carry inner and outer fold counts into the effective recipe."""
    project = _generate_project(
        tmp_path, cv_enabled="true", cv_type="nested_cv", cv_folds="4", cv_inner_folds="2"
    )
    config = json.loads((project / "config/workflow.json").read_text(encoding="utf-8"))
    from skyulf.integrations.databricks.local_cv import LocalCVSpec
    from skyulf.integrations.databricks.local_search import prepare_search_pipeline

    config = _load_project_config(project, config)
    cv = LocalCVSpec.from_workflow(config)
    recipe = prepare_search_pipeline(
        config["pipeline"], cv, target_column=config["target_column"], event_column=None
    )
    assert recipe["modeling"]["cv_folds"] == 4
    assert recipe["modeling"]["cv_inner_folds"] == 2


@pytest.mark.parametrize("method", ["time_series_split", "nested_cv"])
def test_temporal_cv_initializes_a_usable_default_holdout(tmp_path, method):
    """Temporal selection must generate a chronological window without hidden manual pins."""
    project = _generate_project(
        tmp_path,
        omit_fields=("split_strategy", "cv_shuffle", "cv_random_state"),
        cv_enabled="true",
        cv_type=method,
        cv_nested_type="time_series_split" if method == "nested_cv" else "auto",
        event_column="observed_at",
    )
    config = _read_validated_config(project)
    assert config["split_strategy"] == "temporal"
    assert config["training_window_mode"] == "rolling_calendar"
    assert config["event_column"] == "observed_at"
    assert config["cv_shuffle"] is False
    assert config["holdout_months"] == 1


@pytest.mark.parametrize("policy", ["time_series_split", "stratified_group_k_fold"])
def test_generated_nested_policies_preserve_controls_and_two_jobs(tmp_path, policy):
    """Real template expansion must carry split metadata and threshold opt-in into Core."""
    from skyulf.integrations.databricks.local_cv import LocalCVSpec
    from skyulf.integrations.databricks.local_search import prepare_search_pipeline

    settings = {"cv_enabled": "true", "cv_type": "nested_cv", "cv_nested_type": policy}
    temporal = policy == "time_series_split"
    if temporal:
        settings.update(
            split_strategy="temporal",
            event_column="event",
            cv_gap="2",
            cv_test_size="4",
            cv_max_train_size="30",
        )
    else:
        settings.update(
            task="classification",
            classification_model="logistic_regression",
            cv_group_column="customer",
            search_tune_threshold="true",
        )
    project = _generate_project(tmp_path, **settings)
    config = _read_validated_config(project)
    config = _load_project_config(project, config)
    cv = LocalCVSpec.from_workflow(config)
    recipe = prepare_search_pipeline(
        config["pipeline"],
        cv,
        target_column=config["target_column"],
        event_column=config.get("event_column"),
    )
    assert set(_read_jobs(project)) == {"train", "score"}
    assert recipe["modeling"]["cv_nested_type"] == policy
    if temporal:
        assert recipe["modeling"]["cv_time_column"] == "event"
        assert recipe["modeling"]["cv_gap"] == 2
        assert recipe["modeling"]["cv_shuffle"] is False
    else:
        assert recipe["modeling"]["cv_group_column"] == "customer"
        assert recipe["modeling"]["tune_threshold"] is True


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("enabled", ["false", "true"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_generated_model_weight_declarations(tmp_path, layout, enabled, task):
    """Real CLI output must run all layouts without a separate weight hook."""
    import runpy

    from skyulf.integrations.databricks.project import load_project_workflow

    overrides = {
        "training_layout": layout,
        "task": task,
        "sample_weight_enabled": enabled,
        "weight_column": "importance",
        "branch_1_sample_weight_enabled": enabled,
        "branch_1_weight_column": "importance",
        "branch_1_task": task,
    }
    project = _generate_project(tmp_path, **overrides)
    modeling = project / "src/modeling"
    assert not (modeling / "weights.py").exists()
    expected = "importance" if enabled == "true" else None
    if layout == "multi_target":
        entries = runpy.run_path(str(modeling / "multi_model.py"))["build_training_branches"]()
        assert entries["branch_1"]["workflow"]["weight_column"] == expected
        assert entries["branch_2"]["workflow"]["weight_column"] is None
    else:
        config = json.loads((project / "config/workflow.json").read_text())
        loaded = load_project_workflow(config, project / "src/features")
        assert loaded["weight_column"] == expected
        assert loaded["reserved_weight_columns"] == ([] if expected is None else [expected])
        filename = "single_model.py" if layout == "single_model" else "model_competition.py"
        assert loaded["weights_python_source"] == (modeling / filename).read_bytes().decode()


@pytest.mark.parametrize(
    "model", ["voting_classifier", "stacking_classifier", "voting_regressor", "stacking_regressor"]
)
def test_weighted_ensemble_uses_selected_members_and_search_space(tmp_path, model):
    """Weighted menus must control both resolved members and their generated tuning axes."""
    from skyulf.integrations.databricks.project import load_project_workflow
    from skyulf.modeling.capabilities import ensure_model_sample_weight_support

    task = "classification" if model.endswith("classifier") else "regression"
    project = _generate_project(
        tmp_path,
        task=task,
        sample_weight_enabled="true",
        **{
            f"{task}_model_weighted": model,
            f"single_ensemble_{task}_base_count": "2",
            f"single_ensemble_{task}_base_1_weighted": "random_forest",
            f"single_ensemble_{task}_base_2_weighted": "decision_tree",
        },
    )
    config = json.loads((project / "config/workflow.json").read_text())
    loaded = load_project_workflow(config, project / "src/features")
    modeling = loaded["pipeline"]["modeling"]
    params = modeling["base_model"]["params"]
    assert modeling["base_model"]["type"] == model
    assert params["base_estimators"] == ["random_forest", "decision_tree"]
    ensure_model_sample_weight_support(model, params)
    assert any(key.startswith("decision_tree__") for key in modeling["search_space"])
    assert not any(key.startswith("logistic_regression__") for key in modeling["search_space"])
