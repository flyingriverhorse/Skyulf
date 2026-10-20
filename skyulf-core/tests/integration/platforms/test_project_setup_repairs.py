"""Project setup must preserve explicit intent before any cloud work."""

import subprocess
import sys

import pytest
import yaml
from test_databricks_bundle_generation import (
    CLI_ONLY,
    TEMPLATE_ROOT,
    _generate_project,
    _initialize_project,
)
from test_project_checks import _bindings, _project
from test_project_source_validation import _config, _query

from skyulf.integrations.databricks.jobs.serving_endpoint_job import read_serving_endpoint_settings
from skyulf.integrations.databricks.projects.project_checks import check_project
from skyulf.integrations.databricks.projects.source_validation import validate_project_sources
from skyulf.integrations.databricks.projects.workflow_config import preview_workflow_config


def _endpoint_settings(tmp_path, name):
    """Load realistic target-bound settings through the deployed job's local admission."""
    path = TEMPLATE_ROOT / "examples/nyctaxi/taxi_fare_endpoint/config/serving.yml"
    document = yaml.safe_load(path.read_text())
    document["serving_endpoint"]["endpoint_name"] = name
    destination = tmp_path / "serving.yml"
    destination.write_text(yaml.safe_dump(document))
    return {
        "serving_config_path": str(destination),
        "catalog": "workspace",
        "input_schema": "taxi",
        "output_schema": "taxi",
        "metadata_schema": "taxi",
        "resource_suffix": "_1234567890123456",
        "bundle_target": "test_development",
        "bundle_project": "taxi_fare_endpoint",
        "cost_tag_key": "CostCenter",
        "cost_tag_value": "test",
    }


def test_generated_endpoint_name_is_bounded_stable_and_distinct(tmp_path):
    """Personal target suffixes must fit without merging distinct long generated names."""
    base = "taxi-fare-endpoint-with-extra-{bundle_target}-endpoint{resource_suffix}"
    values = _endpoint_settings(tmp_path, base)
    first = read_serving_endpoint_settings(values)["endpoint_name"]
    repeated = read_serving_endpoint_settings(values)["endpoint_name"]
    other = read_serving_endpoint_settings(values | {"resource_suffix": "_1234567890123457"})[
        "endpoint_name"
    ]
    assert len(first) <= 63
    assert repeated == first
    assert other != first


def test_explicit_oversized_endpoint_name_still_fails(tmp_path):
    """Only generated names may be shortened; explicit operator names keep strict admission."""
    values = _endpoint_settings(tmp_path, "a" * 64)
    with pytest.raises(ValueError, match="1-63"):
        read_serving_endpoint_settings(values)


def test_short_generated_endpoint_name_keeps_existing_identity(tmp_path):
    """Already valid endpoint names must not change during an unrelated code upgrade."""
    values = _endpoint_settings(tmp_path, "taxi-fare-{bundle_target}-endpoint{resource_suffix}")
    assert read_serving_endpoint_settings(values)["endpoint_name"] == (
        "taxi-fare-test-development-endpoint-1234567890123456"
    )


@pytest.mark.parametrize(
    "delivery,accepted", [("endpoint", True), ("batch", False), ("both", False)]
)
def test_source_cdf_requirement_follows_prediction_delivery(delivery, accepted):
    """Endpoint smoke sources do not consume CDF, while incremental batch sources do."""
    config = _config() | {"prediction_delivery": delivery}
    query, _ = _query(cdf=False)
    if not accepted:
        with pytest.raises(ValueError, match="change data feed"):
            validate_project_sources(config, query)
        return
    assert validate_project_sources(config, query)["status"] == "passed"


def test_source_preview_records_late_decimal_deduplicate_model_conversion():
    """A duplicate identity can remain exact during filtering and become numeric for fitting."""
    config = _config() | {
        "pre_split_steps": [
            {"name": "dedup", "transformer": "Deduplicate", "params": {"subset": ["x"]}}
        ]
    }
    original, _ = _query()

    def query(sql):
        """Expose the same precise duplicate key in the training and prediction schemas."""
        rows = original(sql)
        if sql.startswith("DESCRIBE TABLE"):
            return [
                row | {"data_type": "decimal(20,0)"} if row["col_name"] == "x" else row
                for row in rows
            ]
        return rows

    report = validate_project_sources(config, query)
    assert all(
        table["planned_input_conversions"]
        == [{"column": "x", "source_dtype": "decimal(20,0)", "target_dtype": "float64"}]
        for table in report["tables"].values()
    )


@pytest.mark.parametrize("delivery,business_keys", [("endpoint", False), ("both", True)])
def test_source_monitoring_uses_request_keys_for_endpoint_labels(delivery, business_keys):
    """Request labels may differ from training business keys, with both delivery checking both."""
    config = _config() | {"prediction_delivery": delivery}
    original, statements = _query()

    def query(sql):
        """Present a request-keyed label table without fabricating a business identity."""
        if sql.startswith("DESCRIBE TABLE") and "`labels`" in sql:
            columns = {
                "databricks_request_id": "string",
                "request_row_index": "bigint",
                "y": "double",
                "known_at": "timestamp",
            }
            if business_keys:
                columns["id"] = "bigint"
            return [{"col_name": name, "data_type": kind} for name, kind in columns.items()]
        return original(sql)

    report = validate_project_sources(
        config, query, label_table="cat.data.labels", label_available_at_column="known_at"
    )
    assert report["status"] == "passed"
    assert any("GROUP BY `databricks_request_id`, `request_row_index`" in sql for sql in statements)


@pytest.mark.parametrize("delivery", ["endpoint", "both"])
def test_source_monitoring_rejects_business_only_endpoint_labels(delivery):
    """Business-key labels cannot silently be bound to a request-keyed serving monitor."""
    query, _ = _query()
    with pytest.raises(ValueError, match="request"):
        validate_project_sources(
            _config() | {"prediction_delivery": delivery},
            query,
            label_table="cat.data.labels",
            label_available_at_column="known_at",
        )


@pytest.mark.parametrize(
    "request_type,index_type", [("bigint", "bigint"), ("string", "double"), ("string", "string")]
)
def test_source_monitoring_rejects_inexact_request_key_types(request_type, index_type):
    """Request identity must match serving payload types without implicit coercion."""
    original, _ = _query()

    def query(sql):
        """Expose a label table with incompatible request identity types."""
        if sql.startswith("DESCRIBE TABLE") and "`labels`" in sql:
            columns = {
                "databricks_request_id": request_type,
                "request_row_index": index_type,
                "y": "double",
                "known_at": "timestamp",
            }
            return [{"col_name": name, "data_type": kind} for name, kind in columns.items()]
        return original(sql)

    with pytest.raises(ValueError, match="request"):
        validate_project_sources(
            _config() | {"prediction_delivery": "endpoint"},
            query,
            label_table="cat.data.labels",
            label_available_at_column="known_at",
        )


def test_source_training_labels_keep_business_key_contract_with_endpoint_delivery():
    """Separate training outcomes retain business-key joins under endpoint-only delivery."""
    query, _ = _query()
    config = _config() | {
        "prediction_delivery": "endpoint",
        "training_label_table": "cat.data.labels",
        "result_available_at_column": "known_at",
    }
    assert validate_project_sources(config, query)["status"] == "passed"


@pytest.mark.parametrize(
    "field,value", [("enabled", "invalid-boolean"), ("workload_size", "INVALID")]
)
def test_static_smoke_validates_endpoint_settings(tmp_path, workflow_config, field, value):
    """Invalid endpoint controls must fail the same offline smoke command as other settings."""
    root = _project(tmp_path, workflow_config)
    document = yaml.safe_load(
        (TEMPLATE_ROOT / "examples/nyctaxi/taxi_fare_endpoint/config/serving.yml").read_text()
    )
    document["serving_endpoint"][field] = value
    (root / "config/serving.yml").write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match=field):
        check_project(root, _bindings())


def test_static_smoke_accepts_valid_placeholder_endpoint_settings(tmp_path, workflow_config):
    """Offline endpoint admission must bind generated placeholders without cloud calls."""
    root = _project(tmp_path, workflow_config)
    content = (TEMPLATE_ROOT / "examples/nyctaxi/taxi_fare_endpoint/config/serving.yml").read_text()
    (root / "config/serving.yml").write_text(content)
    result = check_project(root, _bindings())
    assert result["status"] == "passed"
    assert result["remote_operations"] is False


def test_workflow_preview_uses_public_snapshot_option(workflow_config):
    """Operators must receive the editable YAML option, rather than an internal SDK field."""
    preview = preview_workflow_config(workflow_config)
    assert "training_table_version=null pins" in preview
    assert "training_version=null pins" not in preview


@CLI_ONLY
def test_endpoint_project_default_preview_is_runnable(tmp_path):
    """The documented default preview must work in an endpoint-only generated project."""
    project = _generate_project(tmp_path, prediction_delivery="endpoint")
    completed = subprocess.run(
        [sys.executable, "src/tools/preview.py"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Endpoint smoke source:" in completed.stdout


@CLI_ONLY
@pytest.mark.parametrize("key", ["project", "target", "suffix", "managed_by"])
def test_actual_cli_rejects_reserved_cost_tag_keys(tmp_path, key):
    """The setup wizard must refuse tags reserved for endpoint ownership."""
    completed, _ = _initialize_project(tmp_path, cost_tag_key=key, prediction_delivery="endpoint")
    assert completed.returncode != 0
    assert key in completed.stderr


@CLI_ONLY
@pytest.mark.parametrize(
    "name,value",
    [
        ("inference_mode", "spark"),
        ("score_handoff", "after_alias_change"),
        ("model_change_mode", "full_rebuild"),
        ("auto_rebuild_on_cdf_expiry", "true"),
        ("preprocessing_probe", "true"),
        ("admit_probed_custom_steps", "true"),
        ("monitoring_catalog", "workspace"),
    ],
)
def test_actual_cli_refuses_removed_setup_answer_names(tmp_path, name, value):
    """Known removed answer names must fail before defaults silently change user intent."""
    completed, _ = _initialize_project(tmp_path, **{name: value})
    assert completed.returncode != 0
    assert name in completed.stderr


@CLI_ONLY
def test_actual_cli_accepts_blank_removed_monitoring_catalog(tmp_path):
    """A blank guard must preserve the current monitoring switch during initialization."""
    project = _generate_project(tmp_path, monitoring_catalog="", monitoring="true")
    variables = yaml.safe_load((project / "deployment/variables.yml").read_text())["variables"]
    assert variables["monitoring_enabled"]["default"] == "true"
    assert variables["monitoring_catalog"]["default"] == "${var.catalog}"


@CLI_ONLY
def test_serving_dispatch_templates_forward_durable_event_identity(tmp_path):
    """Training and daily promotion must pass the saved transition identity to child dispatch."""
    project = _generate_project(tmp_path, prediction_delivery="both", include_daily_rollout="true")
    for filename, upstream in (("train", "training_report"), ("daily_rollout", "daily_rollout")):
        document = yaml.safe_load((project / "resources" / f"{filename}.job.yml").read_text())
        job = next(iter(document["resources"]["jobs"].values()))
        task = next(item for item in job["tasks"] if item["task_key"] == "run_serving_endpoint")
        parameters = task["notebook_task"]["base_parameters"]
        assert (
            parameters["serving_event_id"] == "{{tasks." + upstream + ".values.serving_event_id}}"
        )
