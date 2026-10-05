"""Optional native dashboard refresh must preserve the producer's existing graph."""

import os
import runpy
from pathlib import Path

import pytest
import yaml

TEMPLATE = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"
DASHBOARD_ID = "01f1c090238e1b6da5d633032ad9960b"
WAREHOUSE_ID = "d047a4d9aa276958"
URL = f"https://example.cloud.databricks.com/dashboardsv3/{DASHBOARD_ID}/published"


def _configure():
    """Load the shipped dependency-free project configuration helper."""
    return runpy.run_path(str(TEMPLATE / "src/tools/configure_monitoring_dashboard.py"))[
        "configure"
    ]


def test_refresh_overlay_preserves_jobs_and_uses_existing_dashboard_url(tmp_path):
    """Opt-in must append a native task without rewriting policy, schedule or compute."""
    resources = tmp_path / "resources"
    resources.mkdir()
    original = resources / "monitoring.job.yml"
    original.write_text("# operator's job, schedules and policies\n", encoding="utf-8")
    path = _configure()(tmp_path, dashboard_url=URL, warehouse_id=WAREHOUSE_ID)
    tasks = yaml.safe_load(path.read_text())["targets"]["test"]["resources"]["jobs"]["monitoring"][
        "tasks"
    ]
    assert original.read_text() == "# operator's job, schedules and policies\n"
    assert tasks[0]["task_key"] == "dashboard_refresh_enabled"
    assert tasks[0]["depends_on"] == [{"task_key": "evaluate_retraining"}]
    assert tasks[0]["condition_task"] == {
        "op": "EQUAL_TO",
        "left": "${bundle.mode}:${var.monitoring_enabled}",
        "right": "production:true",
    }
    assert tasks[1]["task_key"] == "refresh_monitoring_dashboard"
    assert tasks[1]["depends_on"] == [{"task_key": "dashboard_refresh_enabled", "outcome": "true"}]
    assert tasks[1]["dashboard_task"] == {
        "dashboard_id": DASHBOARD_ID,
        "warehouse_id": WAREHOUSE_ID,
    }
    assert "environment_key" not in tasks[1]
    assert "subscription" not in tasks[1]["dashboard_task"]


def test_unconfigured_and_disabled_dashboard_has_no_placeholder_task(tmp_path):
    """Disabling removes only the owned overlay and leaves no invalid empty task."""
    (tmp_path / "resources").mkdir()
    configure = _configure()
    path = configure(tmp_path)
    assert not path.exists()
    configure(tmp_path, dashboard_id=DASHBOARD_ID)
    task = yaml.safe_load(path.read_text())["targets"]["test"]["resources"]["jobs"]["monitoring"][
        "tasks"
    ][1]
    assert task["dashboard_task"] == {"dashboard_id": DASHBOARD_ID}
    configure(tmp_path)
    assert not path.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dashboard_id": "bad:id"},
        {"dashboard_url": "http://example.com/dashboardsv3/" + DASHBOARD_ID},
        {"dashboard_url": "https://example.com/unrelated/" + DASHBOARD_ID},
        {"dashboard_id": DASHBOARD_ID, "warehouse_id": "not-a-warehouse"},
        {"dashboard_id": DASHBOARD_ID, "dashboard_url": URL},
        {"warehouse_id": WAREHOUSE_ID},
        {"dashboard_id": DASHBOARD_ID, "targets": ()},
        {"dashboard_id": DASHBOARD_ID, "targets": ("invalid:target",)},
    ],
)
def test_invalid_refresh_configuration_does_not_replace_existing_overlay(tmp_path, kwargs):
    """Bad IDs or URLs must fail before changing a deployed project's configuration."""
    (tmp_path / "resources").mkdir()
    configure = _configure()
    path = configure(tmp_path, dashboard_id=DASHBOARD_ID)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        configure(tmp_path, **kwargs)
    assert path.read_bytes() == before


def test_refresh_helper_does_not_overwrite_or_remove_unowned_overlay(tmp_path):
    """The reserved filename must not erase operator-managed resource configuration."""
    resources = tmp_path / "resources"
    resources.mkdir()
    path = resources / "monitoring_dashboard.job.yml"
    path.write_text("# manual configuration\n", encoding="utf-8")
    configure = _configure()
    for kwargs in ({}, {"dashboard_id": DASHBOARD_ID}):
        with pytest.raises(ValueError, match="not generated"):
            configure(tmp_path, **kwargs)
    assert path.read_text() == "# manual configuration\n"


def test_refresh_overlay_only_configures_selected_targets(tmp_path):
    """Target overrides merge with base tasks without adding refresh to other environments."""
    (tmp_path / "resources").mkdir()
    path = _configure()(tmp_path, dashboard_id=DASHBOARD_ID, targets=("test", "prod", "test"))
    overlay = yaml.safe_load(path.read_text())
    assert set(overlay) == {"targets"}
    assert set(overlay["targets"]) == {"test", "prod"}
    assert overlay["targets"]["test"] == overlay["targets"]["prod"]


def test_refresh_accepts_a_copied_published_dashboard_page_url(tmp_path):
    """Operators can reuse a browser page link without manually extracting its dashboard ID."""
    (tmp_path / "resources").mkdir()
    path = _configure()(tmp_path, dashboard_url=URL + "/pages/performance?o=123")
    overlay = yaml.safe_load(path.read_text())
    tasks = overlay["targets"]["test"]["resources"]["jobs"]["monitoring"]["tasks"]
    assert tasks[1]["dashboard_task"]["dashboard_id"] == DASHBOARD_ID


def test_refresh_preserves_numeric_target_names_as_strings(tmp_path):
    """Generated YAML must not coerce an explicitly named target into a number."""
    (tmp_path / "resources").mkdir()
    path = _configure()(tmp_path, dashboard_id=DASHBOARD_ID, targets=("123",))
    assert set(yaml.safe_load(path.read_text())["targets"]) == {"123"}


@pytest.mark.skipif(not os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE"), reason="CLI opt-in")
@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("compute", ["serverless", "policy_cluster"])
def test_generated_projects_ship_optional_native_dashboard_refresh(tmp_path, layout, compute):
    """Every real CLI layout keeps refresh absent until explicitly configured."""
    from test_databricks_bundle_generation import _generate_project

    project = _generate_project(tmp_path, training_layout=layout, compute_mode=compute)
    jobs = yaml.safe_load((project / "resources/monitoring.job.yml").read_text())
    assert len(jobs["resources"]["jobs"]["monitoring"]["tasks"]) == 3
    assert not (project / "resources/monitoring_dashboard.job.yml").exists()
    configure = runpy.run_path(str(project / "src/tools/configure_monitoring_dashboard.py"))[
        "configure"
    ]
    path = configure(project, dashboard_url=URL, warehouse_id=WAREHOUSE_ID)
    overlay = yaml.safe_load(path.read_text())
    assert len(overlay["targets"]["test"]["resources"]["jobs"]["monitoring"]["tasks"]) == 2
