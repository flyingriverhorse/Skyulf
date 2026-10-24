"""The wizard sets up the test dashboard; deployment/targets.yml explains syst and prod."""

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft7Validator
from test_databricks_bundle_generation import (
    CLI,
    OFFLINE_CLI,
    PROFILE,
    _generate_project,
    _initialize_project,
)

TEMPLATE_ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"
WAREHOUSE_ID = "d047a4d9aa276958"
DASHBOARD_ID = "01f1c090238e1b6da5d633032ad9960b"
URL = f"https://dbc-45604623-c18b.cloud.databricks.com/dashboardsv3/{DASHBOARD_ID}/published?o=1"
STORE = {"monitoring": "true", "monitoring_schema": "taxi_monitoring"}
needs_cli = pytest.mark.skipif(
    not (PROFILE or OFFLINE_CLI) or not CLI,
    reason="Set SKYULF_BUNDLE_OFFLINE_CLI=1 to render the template with the installed CLI.",
)


def _properties():
    """Read the generated initializer questions exactly as the CLI receives them."""
    return json.loads((TEMPLATE_ROOT / "databricks_template_schema.json").read_text())["properties"]


def _visible(name, values):
    """Apply the CLI skip predicate to one answer set."""
    return not Draft7Validator(_properties()[name].get("skip_prompt_if", False)).is_valid(values)


def _dashboards(project):
    """Parse the targets file, which holds every target's dashboard settings."""
    return yaml.safe_load((project / "deployment/targets.yml").read_text(encoding="utf-8"))


def _with_dashboard(targets):
    """Targets that create a dashboard or add its refresh task to the monitoring job."""
    return [
        name
        for name, target in targets.items()
        if "dashboards" in (target.get("resources") or {})
        or "tasks" in (((target.get("resources") or {}).get("jobs") or {}).get("monitoring") or {})
    ]


def _refresh_task(target):
    """Return the refresh task, checking it waits for evaluate_retraining and monitoring_enabled."""
    gate, task = target["resources"]["jobs"]["monitoring"]["tasks"]
    assert gate["task_key"] == "dashboard_refresh_enabled"
    assert gate["depends_on"] == [{"task_key": "evaluate_retraining"}]
    assert gate["condition_task"] == {
        "op": "EQUAL_TO",
        "left": "${var.monitoring_enabled}",
        "right": "true",
    }
    assert task["task_key"] == "refresh_monitoring_dashboard"
    assert task["depends_on"] == [{"task_key": "dashboard_refresh_enabled", "outcome": "true"}]
    return task


def _examples(project):
    """Uncomment the hand-copy examples so their YAML is checked like real configuration."""
    text = (project / "deployment/targets.yml").read_text(encoding="utf-8")
    blocks = text.split("# Example: ")[1:]
    parsed = {}
    for block in blocks:
        title, _, body = block.partition("\n")
        lines = [line[1:] for line in body.splitlines() if line.startswith("#  ")]
        parsed[title.split(" ")[0]] = yaml.safe_load("\n".join(lines))
    return parsed


def test_dashboard_questions_follow_the_chosen_mode():
    """create asks for a warehouse, connect for a URL; a missing store is a generation guard."""
    assert _properties()["monitoring_dashboard"]["default"] == "none"
    assert _visible("monitoring_dashboard", STORE)
    for mode, warehouse, url in (
        ("none", False, False),
        ("create", True, False),
        ("connect", True, True),
    ):
        answers = {**STORE, "monitoring_dashboard": mode}
        assert _visible("monitoring_dashboard_warehouse_id", answers) is warehouse
        assert _visible("monitoring_dashboard_url", answers) is url


@pytest.mark.parametrize(
    ("name", "value", "valid"),
    [
        ("monitoring_dashboard_warehouse_id", WAREHOUSE_ID, True),
        ("monitoring_dashboard_warehouse_id", "", True),
        ("monitoring_dashboard_warehouse_id", "d047a4d9", False),
        ("monitoring_dashboard_url", URL, True),
        ("monitoring_dashboard_url", "", True),
        ("monitoring_dashboard_url", f"http://host/dashboardsv3/{DASHBOARD_ID}", False),
        ("monitoring_dashboard_url", "https://host/dashboardsv3/not-an-id/published", False),
        ("monitoring_dashboard_url", URL + '"', False),
        (
            "monitoring_dashboard_url",
            f'https://bad"host/dashboardsv3/{DASHBOARD_ID}/published',
            False,
        ),
        (
            "monitoring_dashboard_url",
            f"https://host/dashboardsv3/{DASHBOARD_ID}/published/pages/a b",
            False,
        ),
    ],
)
def test_answer_patterns_match_the_helper(name, value, valid):
    """The wizard must refuse values the helper would refuse, before rendering."""
    assert Draft7Validator({"pattern": _properties()[name]["pattern"]}).is_valid(value) is valid


@needs_cli
def test_default_configures_no_target_but_keeps_examples(tmp_path):
    """none leaves every target without a dashboard; both examples still parse when uncommented."""
    project = _generate_project(tmp_path, **STORE)
    assert not _with_dashboard(_dashboards(project)["targets"])
    examples = _examples(project)
    assert set(examples) == {"create", "connect", "duration"}
    assert examples["duration"]["prod"]["resources"]["jobs"]["train"]["health"]["rules"]
    create = examples["create"]["prod"]
    assert create["resources"]["dashboards"]["monitoring_dashboard"]["warehouse_id"] == (
        "REPLACE_WITH_WAREHOUSE_ID"
    )
    assert _refresh_task(create)["dashboard_task"] == {
        "dashboard_id": "${resources.dashboards.monitoring_dashboard.id}"
    }
    assert _refresh_task(examples["connect"]["prod"])["dashboard_task"] == {
        "dashboard_id": "REPLACE_DASHBOARD_ID"
    }
    assert not (project / "src/tools/configure_monitoring_dashboard.py").exists()


@needs_cli
def test_create_configures_only_the_test_target(tmp_path):
    """create owns one dashboard in test; syst and prod stay for the user to set by hand."""
    project = _generate_project(
        tmp_path,
        **STORE,
        monitoring_dashboard="create",
        monitoring_dashboard_warehouse_id=WAREHOUSE_ID,
    )
    targets = _dashboards(project)["targets"]
    assert _with_dashboard(targets) == ["test"]
    test = targets["test"]
    dashboard = test["resources"]["dashboards"]["monitoring_dashboard"]
    assert dashboard["warehouse_id"] == WAREHOUSE_ID
    assert dashboard["file_path"] == "../src/monitoring/monitoring.lvdash.json"
    assert (dashboard["dataset_catalog"], dashboard["dataset_schema"]) == (
        "${var.monitoring_catalog}",
        "${var.monitoring_schema}",
    )
    assert dashboard["embed_credentials"] is False
    assert _refresh_task(test)["dashboard_task"] == {
        "dashboard_id": "${resources.dashboards.monitoring_dashboard.id}"
    }
    assert test["variables"]["monitoring_dashboard_url"] == (
        "${workspace.host}/dashboardsv3/${resources.dashboards.monitoring_dashboard.id}/published"
    )
    assert _examples(project)["create"]["prod"]["resources"]["jobs"] == test["resources"]["jobs"]


@needs_cli
@pytest.mark.parametrize("warehouse", ["", WAREHOUSE_ID])
def test_connect_refreshes_the_existing_dashboard(tmp_path, warehouse):
    """connect takes the ID from the URL, links the URL and never defines a dashboard."""
    project = _generate_project(
        tmp_path,
        **STORE,
        monitoring_dashboard="connect",
        monitoring_dashboard_url=URL,
        monitoring_dashboard_warehouse_id=warehouse,
    )
    test = _dashboards(project)["targets"]["test"]
    assert "dashboards" not in test["resources"]
    expected = {"dashboard_id": DASHBOARD_ID} | ({"warehouse_id": warehouse} if warehouse else {})
    assert _refresh_task(test)["dashboard_task"] == expected
    assert test["variables"]["monitoring_dashboard_url"] == URL
    assert (
        "deployment/monitoring.yml"
        not in yaml.safe_load((project / "databricks.yml").read_text(encoding="utf-8"))["include"]
    )


@needs_cli
@pytest.mark.parametrize(
    ("guard", "answers"),
    [
        (
            "ERROR_monitoring_dashboard_create_requires_warehouse_id",
            {**STORE, "monitoring_dashboard": "create"},
        ),
        (
            "ERROR_monitoring_dashboard_connect_requires_published_url",
            {**STORE, "monitoring_dashboard": "connect"},
        ),
        (
            "ERROR_monitoring_dashboard_requires_monitoring_store",
            {
                "monitoring": "false",
                "monitoring_dashboard": "create",
                "monitoring_dashboard_warehouse_id": WAREHOUSE_ID,
            },
        ),
    ],
)
def test_incomplete_dashboard_answers_fail_generation(tmp_path, guard, answers):
    """A dashboard choice without its required value must stop generation, not deploy half."""
    generated, _ = _initialize_project(tmp_path, **answers)
    assert generated.returncode != 0
    assert guard in generated.stdout + generated.stderr
