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
    for name in ("include_daily_rollout", "include_online_publication"):
        assert topic[name]["default"] == "false"
        assert topic[name]["enum"] == ["false", "true"]
        assert generated["properties"][name]["default"] == "false"


@pytest.mark.parametrize("job", list(OPTIONAL))
def test_optional_jobs_have_paused_serial_schedules_and_pinned_runtime(job):
    """Native job serialization must retain the guard against automatic retries."""
    jobs = pytest.importorskip("databricks.sdk.service.jobs")
    path = TEMPLATE / (OPTIONAL[job][0] + ".tmpl")
    assert path.is_file(), "Optional serving job resource is not implemented"
    resource = yaml.safe_load(path.read_text(encoding="utf-8"))["resources"]["jobs"][job]
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
    assert task["notebook_task"]["base_parameters"] == {
        "serving_config_path": "${workspace.file_path}/config/serving.yml"
    }
    assert resource["environments"][0]["spec"]["dependencies"] == [
        "../dist/skyulf/*.whl",
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
    assert config["rollout"]["smoke_records"] == []
    assert config["online_publication"]["source_table_id"] == ""
    assert "verdict" not in json.dumps(config) and "PASS" not in json.dumps(config)
