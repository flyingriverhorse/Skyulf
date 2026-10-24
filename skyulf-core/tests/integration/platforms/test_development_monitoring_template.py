"""Optional development monitoring writes to its own named store; the default stays off."""

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

STORE = {"monitoring": "true", "monitoring_schema": "taxi_monitoring"}
DEV_TARGETS = ("test_development", "syst_development", "prod_development")
DEV_ON = {**STORE, "personal_development_targets": "true", "development_monitoring": "true"}
needs_cli = pytest.mark.skipif(
    not (PROFILE or OFFLINE_CLI) or not CLI,
    reason="Set SKYULF_BUNDLE_OFFLINE_CLI=1 to render the template with the installed CLI.",
)


def _yaml(project, relative):
    """Parse one rendered project file."""
    return yaml.safe_load((project / relative).read_text(encoding="utf-8"))


def _dev_schema_default(project):
    """Return the rendered development store name ("" when development monitoring is off)."""
    variables = _yaml(project, "deployment/variables.yml")["variables"]
    return variables["development_monitoring_schema"]["default"]


def _gate(project, job):
    """Return the condition that decides whether a job registers monitoring."""
    tasks = _yaml(project, f"resources/{job}.job.yml")["resources"]["jobs"][job]["tasks"]
    return next(t for t in tasks if t["task_key"] == "monitoring_allowed")["condition_task"]


@needs_cli
@pytest.mark.parametrize(
    "answers",
    [
        {"personal_development_targets": "true"},
        {**STORE, "personal_development_targets": "true"},
        {**STORE},
    ],
)
def test_development_monitoring_is_off_unless_chosen(tmp_path, answers):
    """Without the answer, dev targets write nothing and the dev store name stays blank."""
    project = _generate_project(tmp_path, **answers)
    targets = _yaml(project, "deployment/targets.yml")["targets"]
    for name in DEV_TARGETS:
        if name in targets:
            assert targets[name]["variables"]["monitoring_enabled"] == "false"
    assert _dev_schema_default(project) == ""


@needs_cli
def test_opt_in_passes_both_store_names_and_keeps_the_shared_schema(tmp_path):
    """Dev targets enable monitoring; every job receives the shared and the development name."""
    project = _generate_project(tmp_path, **DEV_ON)
    assert _dev_schema_default(project) == "development_monitoring"
    # Every target, monitoring included, is defined in targets.yml only.
    assert not (project / "deployment/monitoring.yml").exists()
    targets = _yaml(project, "deployment/targets.yml")["targets"]
    for name in DEV_TARGETS:
        assert targets[name]["variables"]["monitoring_enabled"] == "true"
        assert "monitoring_schema" not in targets[name]["variables"]
    for job in ("train", "score"):
        assert _gate(project, job) == {
            "op": "NOT_EQUAL",
            "left": "${bundle.mode}:${var.monitoring_enabled}",
            "right": "development:false",
        }
    text = (project / "resources/monitoring.job.yml").read_text(encoding="utf-8")
    assert text.count("monitoring_schema: ${var.monitoring_schema}") == text.count(
        "development_monitoring_schema: ${var.development_monitoring_schema}"
    )


@needs_cli
def test_development_store_name_comes_from_setup(tmp_path):
    """Any schema name can be used, for example an existing one in the workspace."""
    project = _generate_project(tmp_path, **DEV_ON, development_monitoring_schema="mlops_sandbox")
    assert _dev_schema_default(project) == "mlops_sandbox"


@needs_cli
def test_create_gives_test_development_a_dashboard_over_its_own_store(tmp_path):
    """The development dashboard reads the development store, the test one the shared store."""
    project = _generate_project(
        tmp_path,
        **DEV_ON,
        monitoring_dashboard="create",
        monitoring_dashboard_warehouse_id="d047a4d9aa276958",
    )
    targets = _yaml(project, "deployment/targets.yml")["targets"]
    assert "resources" not in targets["syst_development"]
    assert "dashboards" not in (targets["syst"].get("resources") or {})

    def schema(target):
        """Return the dataset schema of one target's dashboard."""
        return targets[target]["resources"]["dashboards"]["monitoring_dashboard"]["dataset_schema"]

    assert schema("test") == "${var.monitoring_schema}"
    assert schema("test_development") == "${var.development_monitoring_schema}"


@needs_cli
@pytest.mark.parametrize(
    ("guard", "answers"),
    [
        (
            "ERROR_development_monitoring_requires_monitoring_store",
            {"personal_development_targets": "true", "development_monitoring": "true"},
        ),
        (
            "ERROR_development_monitoring_schema_must_differ_from_monitoring_schema",
            {**DEV_ON, "development_monitoring_schema": "taxi_monitoring"},
        ),
    ],
)
def test_incomplete_development_monitoring_fails_generation(tmp_path, guard, answers):
    """A missing store, or a development store equal to the shared one, stops generation."""
    generated, _ = _initialize_project(tmp_path, **answers)
    assert generated.returncode != 0
    assert guard in generated.stdout + generated.stderr


@pytest.mark.parametrize(
    ("name", "valid"),
    [
        ("development_monitoring", True),
        ("mlops_sandbox", True),
        ("gold", True),
        ("1gold", False),
        ("bad-name", False),
    ],
)
def test_setup_accepts_any_schema_identifier(name, valid):
    """The development store name is free; it only has to be a schema identifier."""
    schema = (
        Path(__file__).resolve().parents[3] / "templates/databricks/databricks_template_schema.json"
    )
    pattern = json.loads(schema.read_text())["properties"]["development_monitoring_schema"][
        "pattern"
    ]
    assert Draft7Validator({"pattern": pattern}).is_valid(name) is valid
