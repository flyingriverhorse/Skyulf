"""Optional serving jobs render paused and delegate to verified runtime services."""

import json
import os
import runpy
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"
TEMPLATE = ROOT / "template/{{.project_name}}"
OPTIONAL = {
    "daily_rollout": ("resources/daily_rollout.job.yml", "src/jobs/daily_rollout.py"),
    "online_publication": (
        "resources/online_publication.job.yml",
        "src/jobs/online_publication.py",
    ),
}
CLI = pytest.mark.skipif(
    not shutil.which("databricks")
    or not (
        os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE")
        or os.environ.get("SKYULF_BUNDLE_OFFLINE_CLI") == "1"
    ),
    reason="Explicit offline or authenticated CLI generation opt-in required.",
)


def test_serving_opt_ins_default_off_in_topic_and_generated_schema():
    """Existing initialization answers must not silently add automated writers."""
    source = ROOT / "schema/serving.json"
    assert source.is_file(), "Serving job opt-in schema is not implemented"
    topic = json.loads(source.read_text(encoding="utf-8"))
    generated = json.loads((ROOT / "databricks_template_schema.json").read_text(encoding="utf-8"))
    assert "include_online_serving" not in topic
    assert "include_online_serving" not in generated["properties"]
    for name in (
        "include_daily_rollout",
        "include_online_publication",
        "include_online_feature_lookup",
    ):
        assert topic[name]["default"] == "false"
        assert topic[name]["enum"] == ["false", "true"]
        assert generated["properties"][name]["default"] == "false"


def test_endpoint_compute_answers_are_hidden_with_compatible_defaults():
    """Answer files expose endpoint tuning without adding mandatory setup questions."""
    topic = json.loads((ROOT / "schema/serving.json").read_text(encoding="utf-8"))
    for name, default in {
        "workload_type": "CPU",
        "workload_size": "Small",
        "scale_to_zero_enabled": "true",
        "ready_timeout_s": "3600",
    }.items():
        field = topic["serving_" + name]
        assert field["default"] == default
        assert field["skip_prompt_if"] == {}


def test_online_setup_budget_allows_preparation_and_default_ready_wait():
    """The task must not expire before its own one-hour readiness timeout can finish."""
    source = (TEMPLATE / "resources/online_feature_lookup_setup.job.yml.tmpl").read_text()
    resource = yaml.safe_load(source)["resources"]["jobs"]["online_feature_lookup_setup"]
    assert resource["tasks"][0]["timeout_seconds"] == "${var.monitoring_timeout_seconds}"


@CLI
@pytest.mark.parametrize(
    "kind,size,zero,timeout",
    [
        ("CPU", "Small", "true", "3600"),
        ("CPU_MEDIUM", "Medium", "false", "120"),
        ("CPU_LARGE", "Large", "true", "7200"),
    ],
)
def test_endpoint_settings_render_as_typed_yaml(tmp_path, kind, size, zero, timeout):
    """Operators' explicit answers must reach serving.yml with boolean and integer types."""
    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(
        tmp_path,
        include_online_feature_lookup="true",
        serving_workload_type=kind,
        serving_workload_size=size,
        serving_scale_to_zero_enabled=zero,
        serving_ready_timeout_s=timeout,
    )
    endpoint = yaml.safe_load((project / "config/serving.yml").read_text())[
        "online_feature_lookup"
    ]["endpoint"]
    assert endpoint["workload_type"] == kind
    assert endpoint["workload_size"] == size
    assert endpoint["scale_to_zero_enabled"] is (zero == "true")
    assert type(endpoint["ready_timeout_s"]) is int
    assert endpoint["ready_timeout_s"] == int(timeout)


def test_publication_mode_prompt_is_conditional_and_defaults_to_triggered():
    """Existing initialization answers keep their paused hourly publisher behavior."""
    topic = json.loads((ROOT / "schema/serving.json").read_text(encoding="utf-8"))
    mode = topic["online_publication_mode"]
    assert mode["default"] == "TRIGGERED"
    assert mode["enum"] == ["TRIGGERED", "CONTINUOUS"]
    assert mode["skip_prompt_if"] == {
        "properties": {"include_online_publication": {"const": "false"}}
    }


@CLI
@pytest.mark.parametrize("serving", ["false", "true"])
@pytest.mark.parametrize("mode", ["TRIGGERED", "CONTINUOUS"])
def test_real_cli_publication_mode_controls_schedule_and_preserves_independent_setup(
    tmp_path, serving, mode
):
    """Continuous publishers must have explicit controls and no repeating sync schedule."""
    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(
        tmp_path,
        include_online_publication="true",
        include_online_feature_lookup=serving,
        online_publication_mode=mode,
    )
    config = yaml.safe_load((project / "config/serving.yml").read_text())
    assert config["online_publication"]["publish_mode"] == mode
    resource = yaml.safe_load((project / "resources/online_publication.job.yml").read_text())
    job = resource["resources"]["jobs"]["online_publication"]
    assert ("schedule" in job) is (mode == "TRIGGERED")
    assert job["parameters"] == [{"name": "publication_action", "default": "start"}]
    assert (
        job["tasks"][0]["notebook_task"]["base_parameters"]["publication_action"]
        == "{{job.parameters.publication_action}}"
    )
    assert (project / "resources/online_feature_lookup_setup.job.yml").is_file() is (
        serving == "true"
    )
    assert not (project / "resources/daily_rollout.job.yml").exists()


@CLI
@pytest.mark.parametrize("job", list(OPTIONAL))
def test_optional_jobs_have_paused_serial_schedules_and_pinned_runtime(tmp_path, job):
    """Native job serialization must retain the guard against automatic retries."""
    jobs = pytest.importorskip("databricks.sdk.service.jobs")
    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(tmp_path, **{"include_" + job: "true"})
    path = project / OPTIONAL[job][0]
    assert path.is_file(), "Optional serving job resource is not implemented"
    text = path.read_text(encoding="utf-8")
    resource = yaml.safe_load(text)["resources"]["jobs"][job]
    assert resource["max_concurrent_runs"] == 1
    assert resource["schedule"]["pause_status"] == "PAUSED"
    assert resource["schedule"]["timezone_id"] == "UTC"
    assert resource["schedule"]["quartz_cron_expression"] == (
        "0 0 4 * * ?" if job == "daily_rollout" else "0 0 * * * ?"
    )
    task = resource["tasks"][0]
    assert task["max_retries"] == 0
    assert jobs.Task.from_dict(task).as_dict().get("disable_auto_optimization") is True
    assert task["disable_auto_optimization"] is True
    assert "disable_auto_optimization" not in task["notebook_task"]
    parameters = {"serving_config_path": "${workspace.file_path}/config/serving.yml"}
    if job == "online_publication":
        parameters["publication_action"] = "{{job.parameters.publication_action}}"
    if job == "daily_rollout":  # rollout evidence names the target that ran it
        parameters.update(
            bundle_target="${bundle.target}",
            job_run_id="{{job.run_id}}",
            task_key="daily_rollout",
            workspace_host="${workspace.host}",
        )
    assert task["notebook_task"]["base_parameters"] == parameters
    assert resource["environments"][0]["spec"]["dependencies"] == [
        "-r ${workspace.file_path}/deployment/serving-requirements.txt",
    ]
    requirements = (TEMPLATE / "deployment/serving-requirements.txt.tmpl").read_text(
        encoding="utf-8"
    )
    assert "-r requirements.txt" in requirements
    assert "databricks-feature-engineering==0.18.1" in requirements


@pytest.mark.parametrize("job", list(OPTIONAL))
def test_notebooks_delegate_to_serving_runtime_and_emit_result(job, monkeypatch):
    """The generated notebooks must call runtime validation before exiting with evidence."""
    notebook = TEMPLATE / OPTIONAL[job][1]
    assert notebook.is_file(), "Serving notebook is not implemented"
    module = ModuleType("skyulf.integrations.databricks.jobs.serving_job")
    called = []
    output = {"status": "HOLD", "receipt_id": "verified"}

    def runner(spark, dbutils):
        """Record the generated notebook's explicit runtime handoff."""
        called.append((spark, dbutils))
        return output

    setattr(module, f"run_{job}_notebook", runner)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    from skyulf.integrations.databricks.jobs.shared import notebook_diagnostics

    monkeypatch.setattr(notebook_diagnostics, "notebook_task", lambda name, dbutils: nullcontext())
    exited = []
    spark = object()
    dbutils = SimpleNamespace(notebook=SimpleNamespace(exit=exited.append))
    runpy.run_path(
        str(notebook), run_name="__main__", init_globals={"spark": spark, "dbutils": dbutils}
    )
    assert called == [(spark, dbutils)]
    assert [json.loads(value) for value in exited] == [output]


@CLI
@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_default_generated_tree_keeps_only_existing_jobs(tmp_path, layout):
    """Opt-outs must omit resources, notebooks and configuration rather than leave dead files."""
    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(tmp_path, training_layout=layout)
    assert {path.name for path in (project / "resources").glob("*.yml")} == {
        "train.job.yml",
        "score.job.yml",
        "monitoring.job.yml",
    }
    for paths in OPTIONAL.values():
        assert all(not (project / path).exists() for path in paths)
    assert not (project / "config/serving.yml").exists()
    assert not (project / "deployment/serving-requirements.txt").exists()


@CLI
@pytest.mark.parametrize("daily,online", [("true", "false"), ("false", "true"), ("true", "true")])
def test_real_cli_renders_only_requested_serving_jobs(tmp_path, daily, online):
    """Each opt-in independently selects a runnable paused job without arbitrary verdict input."""
    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(
        tmp_path, include_daily_rollout=daily, include_online_publication=online
    )
    for job, selected in (("daily_rollout", daily), ("online_publication", online)):
        assert all((project / path).is_file() is (selected == "true") for path in OPTIONAL[job])
    config = yaml.safe_load((project / "config/serving.yml").read_text(encoding="utf-8"))
    assert set(config) == {"rollout", "online_publication"}
    assert all(
        section["enabled"] is False and section["exclusive_writer"] is False
        for section in config.values()
    )
    assert config["rollout"]["run_id"] == ""
    assert config["rollout"]["traffic_step_percent"] == 10
    assert config["rollout"]["stage_minimum_hours"] == 24.0
    assert config["rollout"]["evidence_max_age_hours"] == 24.0
    assert config["rollout"]["smoke_records"] == []
    assert config["online_publication"]["source_table_id"] == ""
    assert "verdict" not in json.dumps(config) and "PASS" not in json.dumps(config)


@CLI
@pytest.mark.parametrize(
    "publication,serving",
    [("false", "false"), ("true", "false"), ("false", "true"), ("true", "true")],
)
def test_online_setup_generation_is_independent_of_publication_and_rollout(
    tmp_path, publication, serving
):
    """External publication users need the same training and setup runtime as publishers."""
    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(
        tmp_path, include_online_publication=publication, include_online_feature_lookup=serving
    )
    enabled = serving == "true"
    assert (project / "resources/online_feature_lookup_setup.job.yml").is_file() is enabled
    assert (project / "src/jobs/online_feature_lookup_setup.py").is_file() is enabled
    assert not (project / "resources/online_serving_setup.job.yml").exists()
    assert not (project / "src/jobs/online_serving_setup.py").exists()
    assert not (project / "resources/daily_rollout.job.yml").exists()
    assert (project / "resources/online_publication.job.yml").is_file() is (publication == "true")
    if enabled:
        config = yaml.safe_load((project / "config/serving.yml").read_text())
        assert "online_serving" not in config
        assert config["online_feature_lookup"]["enabled"] is False
        assert config["online_feature_lookup"]["publications"][0]["source_table_id"] == ""
        training = yaml.safe_load((project / "config/training.yml").read_text())["defaults"]
        assert training["online_features"]["required_features"] == []
        assert training["online_features"]["freshness"][0]["max_age_seconds"] is None
        assert training["feature_lookup"]["lookups"][0]["feature_names"] == []
        from skyulf.integrations.databricks.projects.yaml_config import read_training_config

        parsed = read_training_config(project / "config")
        assert parsed is not None
        assert parsed["defaults"]["online_features"] == training["online_features"]
        requirements = (project / "deployment/requirements.txt").read_text()
        assert "databricks-feature-engineering==0.18.1" in requirements
        resource = yaml.safe_load(
            (project / "resources/online_feature_lookup_setup.job.yml").read_text()
        )
        job = resource["resources"]["jobs"]["online_feature_lookup_setup"]
        assert "schedule" not in job
        assert job["max_concurrent_runs"] == 1
        assert job["tasks"][0]["max_retries"] == 0
        assert (project / "deployment/serving-requirements.txt").is_file()
