"""Operator controls must reach rendered jobs and explain bounded training failures."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jsonschema import Draft7Validator
from test_databricks_bundle_generation import CLI_ONLY, _generate_project
from test_databricks_deployment import _resolve
from test_databricks_deployment import offline_workspace as offline_workspace

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"


@CLI_ONLY
def test_monitoring_notifications_resolve_in_real_bundle(tmp_path, offline_workspace):
    """Monitoring failures need the same native job notification controls as scoring."""
    project = _generate_project(tmp_path)
    path = project / "deployment/variables.yml"
    document = yaml.safe_load(path.read_text())
    settings = {
        "email_notifications": {"on_failure": ["operator@example.invalid"]},
        "webhook_notifications": {"on_failure": [{"id": "test-destination"}]},
        "health_rules": [{"metric": "RUN_DURATION_SECONDS", "op": "GREATER_THAN", "value": 1800}],
    }
    for name, value in settings.items():
        document["variables"][f"monitoring_{name}"]["default"] = value
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    job = _resolve(project, "test", offline_workspace)["resources"]["jobs"]["monitoring"]
    assert job["email_notifications"] == settings["email_notifications"]
    assert job["webhook_notifications"] == settings["webhook_notifications"]
    assert job["health"]["rules"] == settings["health_rules"]


def test_reference_cleanup_is_manual_and_dry_run_by_default(tmp_path):
    """Ordinary observation must never become a destructive retention job by default."""
    project = _generate_project(tmp_path)
    job = yaml.safe_load((project / "resources/monitoring.job.yml").read_text())["resources"][
        "jobs"
    ]["monitoring"]
    params = {item["name"]: item["default"] for item in job["parameters"]}
    assert params["monitoring_action"] == "observe"
    assert params["cleanup_confirm"] == "false"
    assert params["reference_retention_days"] == "${var.monitoring_reference_retention_days}"
    first = job["tasks"][0]["notebook_task"]["base_parameters"]
    assert first["cleanup_confirm"] == "{{job.parameters.cleanup_confirm}}"
    assert first["reference_retention_days"] == "{{job.parameters.reference_retention_days}}"


@pytest.mark.parametrize("layout", ["single_model", "multi_target"])
def test_detailed_setup_exposes_sampling_and_probe(layout):
    """Users must be able to choose sample limits and diagnostics without hidden overrides."""
    properties = json.loads((ROOT / "databricks_template_schema.json").read_text())["properties"]
    values = {name: field["default"] for name, field in properties.items()}
    values.update(setup_mode="detailed", training_layout=layout)
    sampling = (
        "branch_1_training_sample_rows" if layout == "multi_target" else "training_sample_rows"
    )
    for name in (sampling, "preprocessing_probe"):
        assert not Draft7Validator(properties[name].get("skip_prompt_if", False)).is_valid(values)


@pytest.mark.parametrize("enabled", ["false", "true"])
def test_probe_answer_reaches_training_without_builder_patch(tmp_path, enabled):
    """Normal wizard output must support the same diagnostics as prepared examples."""
    project = _generate_project(tmp_path, preprocessing_probe=enabled)
    training = yaml.safe_load((project / "config/training.yml").read_text())
    assert training["defaults"]["preprocessing_probe"] is (enabled == "true")


def test_preview_warns_about_repeated_fixed_training_and_initial_scoring(tmp_path):
    """The offline preview must disclose repeated training and initial in-sample scoring."""
    project = _generate_project(
        tmp_path,
        retraining_mode="scheduled",
        split_strategy="temporal",
        training_window_mode="fixed_window",
        event_column="event_time",
    )
    preview = runpy.run_path(str(project / "src/tools/preview.py"))
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    path = project / "config/training.yml"
    bindings = {
        "catalog": "cat",
        "input_schema": "data",
        "output_schema": "data",
        "metadata_schema": "models",
        "resource_suffix": "",
    }
    text = preview["_project_preview"](read_workflow_config(path), path, bindings, "score")
    assert "same fixed training window" in text
    assert "initial scoring includes training history" in text


def test_bounded_training_error_reports_lower_bound_and_remedies():
    """Stopping at max_rows + 1 must not pretend to know the complete table size."""
    from skyulf.integrations.databricks.training.fitting.candidate import (
        TrainingSpec,
        _materialize_training_rows,
    )

    spec = TrainingSpec(
        table="cat.data.source",
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="y",
        max_rows=4,
        max_bytes=10000,
    )
    source = SimpleNamespace(
        toLocalIterator=lambda: iter({"id": i, "x": i, "y": i} for i in range(20))
    )
    with pytest.raises(
        ValueError, match=r"at least 5.*max_rows=4.*config/training.yml.*training_sample_rows"
    ):
        _materialize_training_rows(source, spec, spec.source_columns)


def test_bounded_training_byte_error_names_public_setting():
    """Operators configure MiB in YAML even though the runtime enforces bytes."""
    from skyulf.integrations.databricks.training.fitting.candidate import (
        TrainingSpec,
        _materialize_training_rows,
    )

    spec = TrainingSpec(
        table="cat.data.source",
        version=0,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="y",
        max_rows=4,
        max_bytes=10,
    )
    source = SimpleNamespace(toLocalIterator=lambda: iter([{"id": 1, "x": "large input", "y": 1}]))
    with pytest.raises(
        ValueError, match=r"exceeds max_bytes=10.*max_input_mb.*training_sample_rows"
    ):
        _materialize_training_rows(source, spec, spec.source_columns)
