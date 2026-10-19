"""Verified endpoint delivery enrolls immutable online identities in the shared inventory."""

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

pytest.importorskip("mlflow")

from skyulf.integrations.databricks.jobs import serving_endpoint_job as job  # noqa: E402
from skyulf.integrations.databricks.jobs import serving_enrollment as enrollment  # noqa: E402
from skyulf.integrations.databricks.jobs.shared import reference_mutations  # noqa: E402
from skyulf.integrations.databricks.observability.monitoring import (  # noqa: E402
    monitoring_store,
)
from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (  # noqa: E402
    MonitorConfig,
)
from skyulf.integrations.databricks.observability.monitoring.serving import (  # noqa: E402
    serving_registration,
)
from skyulf.integrations.databricks.serving.contracts import (  # noqa: E402
    PinnedEndpointPlan,
)


@pytest.fixture
def delivery(monkeypatch):
    """Retain the real job, parity and enrollment logic with only cloud boundaries replaced."""
    settings = {
        "enabled": True,
        "endpoint_name": "risk-endpoint",
        "model_name": "main.models.risk",
        "logging_mode": "telemetry",
        "logging_catalog": "main",
        "logging_schema": "logs",
        "logging_table_prefix": "risk",
        "sql_function": {"enabled": True, "name": "main.models.score_risk"},
        "ownership": {"project": "risk", "target": "prod"},
        "tags": {"team": "risk"},
    }
    values = {
        "monitoring_enabled": "true",
        "monitoring_catalog": "ops",
        "monitoring_schema": "monitoring",
    }
    events, inventory = [], {}

    def prepare(spec, **kwargs):
        """Return the inspected shape while the tests control registry I/O."""
        return PinnedEndpointPlan(
            spec, {}, ("value",), (("value", "float64"),), (("prediction", "float64"),)
        )

    model = SimpleNamespace(
        metadata=SimpleNamespace(metadata={"skyulf_artifact_kind": "local_pipeline"}),
        predict=lambda frame: pd.DataFrame({"prediction": frame["value"] * 2}),
    )
    monkeypatch.setattr(job, "prepare_pinned_endpoint", prepare)
    monkeypatch.setattr(job, "_load_pyfunc", lambda *args: model)
    monkeypatch.setattr(job, "sample_serving_rows", lambda *args, **kwargs: [{"value": 2.0}])
    endpoint = Mock(side_effect=lambda *args, **kwargs: events.append("endpoint") or {})
    monkeypatch.setattr(job, "ensure_owned_endpoint", endpoint)
    monkeypatch.setattr(job, "ensure_owned_sql_function", lambda *args, **kwargs: {})
    monkeypatch.setattr(job, "grant_serving_access", lambda *args, **kwargs: None)
    rest = Mock(
        side_effect=lambda *args: (
            events.append("rest") or SimpleNamespace(predictions=[{"prediction": 4.0}])
        )
    )
    sql = Mock(side_effect=lambda *args: events.append("sql") or [{"prediction": 4.0}])
    monkeypatch.setattr(job, "query_named_records", rest)
    monkeypatch.setattr(job, "_sql_predictions", sql)

    @contextmanager
    def scope(*args, **kwargs):
        """Observe use of the shared mutation guard without a live Delta authority."""
        events.append("guard")
        yield

    def enroll(spark, namespace, config, **kwargs):
        """Model the inventory's stable monitor ID upsert at the Delta boundary."""
        assert kwargs == {"preserve_activation": True}
        events.append(("enroll", config.model_version, config.enabled))
        inventory[config.monitor_id] = config.payload()

    monkeypatch.setattr(reference_mutations, "reference_mutation_scope", scope)
    monkeypatch.setattr(serving_registration, "ensure_monitoring_store", lambda *args: None)
    monkeypatch.setattr(monitoring_store, "ensure_monitoring_store", lambda *args: None)
    reference = Mock(side_effect=lambda *args, **kwargs: events.append("reference"))
    monkeypatch.setattr(serving_registration, "prepare_spark_monitoring_reference", reference)
    monkeypatch.setattr(serving_registration, "enroll_monitor", enroll)
    monkeypatch.setattr(monitoring_store, "enroll_monitor", enroll)
    monkeypatch.setattr(
        monitoring_store, "load_enrolled_models", lambda *args: list(inventory.values())
    )
    registry = Mock()
    registry.get_model_version_by_alias.return_value.version = "2"
    return SimpleNamespace(
        settings=settings,
        values=values,
        events=events,
        inventory=inventory,
        model=model,
        endpoint=endpoint,
        rest=rest,
        sql=sql,
        registry=registry,
        reference=reference,
    )


def _run(delivery, **kwargs):
    """Run the actual endpoint entry point with one concrete invocation."""
    return job.run_serving_endpoint(
        object(),
        delivery.settings,
        source_table="main.source.features",
        client=Mock(),
        registry_client=delivery.registry,
        **kwargs,
    )


def test_logging_requires_monitoring_store_before_endpoint_write(delivery):
    """Paid endpoint changes must not succeed with unconfigured automatic enrollment."""
    with pytest.raises(ValueError, match="monitoring_catalog.*monitoring_schema"):
        _run(delivery, monitoring_values={"monitoring_enabled": "true"})
    delivery.endpoint.assert_not_called()
    assert delivery.inventory == {}


def test_disabled_logging_reports_skipped_enrollment_without_inventory_access(delivery):
    """Serving without payload logging remains independent of the monitoring store."""
    delivery.settings["logging_mode"] = "disabled"
    result = _run(delivery, monitoring_values={"monitoring_enabled": "true"})
    assert result["monitoring"] == {"status": "skipped", "reason": "logging_disabled"}
    assert delivery.events == ["endpoint", "rest", "sql"]
    assert delivery.inventory == {}


@pytest.mark.parametrize("mode", ["disabled", "telemetry", "ai_gateway"])
@pytest.mark.parametrize("enabled", [None, "false"])
def test_target_monitoring_disabled_skips_all_monitor_work(delivery, monkeypatch, mode, enabled):
    """Disabled target monitoring cannot block serving or touch an existing monitor inventory."""
    delivery.settings["logging_mode"] = mode
    values = {
        "monitoring_catalog": "${var.unresolved}",
        "monitoring_schema": "",
        "monitoring_drift_thresholds": "invalid",
        "job_run_id": "invalid",
    }
    if enabled is not None:
        values["monitoring_enabled"] = enabled
    delivery.model.metadata = None
    delivery.inventory["prior"] = {"enabled": True}
    blocked = Mock(side_effect=AssertionError("Disabled monitoring performed work"))
    for module, name in (
        (enrollment, "store_namespace"),
        (enrollment, "_served_identities"),
        (reference_mutations, "notebook_reference_holder"),
        (reference_mutations, "prepare_reference_mutations"),
        (serving_registration, "enroll_serving_configs"),
        (monitoring_store, "load_enrolled_models"),
    ):
        monkeypatch.setattr(module, name, blocked)
    result = _run(delivery, monitoring_values=values)
    assert result["status"] == "READY"
    assert result["logging_mode"] == mode
    assert result["monitoring"] == {"status": "skipped", "reason": "monitoring_disabled"}
    assert delivery.events == ["endpoint", "rest", "sql"]
    assert delivery.inventory == {"prior": {"enabled": True}}
    delivery.reference.assert_not_called()
    blocked.assert_not_called()


@pytest.mark.parametrize("value", ["", "TRUE", "False", "yes", True, None])
def test_target_monitoring_flag_rejects_invalid_values_before_endpoint_write(delivery, value):
    """Malformed target flags must fail before any paid endpoint mutation."""
    delivery.values["monitoring_enabled"] = value
    with pytest.raises(ValueError, match="monitoring_enabled must be true or false"):
        _run(delivery, monitoring_values=delivery.values)
    delivery.endpoint.assert_not_called()
    assert delivery.inventory == {}


def test_target_monitoring_disabled_ignores_a_supplied_namespace(delivery):
    """Direct helper callers cannot bypass the target switch with a precomputed namespace."""
    result = enrollment.enroll_verified_serving(
        object(), "ops.monitoring", [], {"monitoring_enabled": "false"}
    )
    assert result == {"status": "skipped", "reason": "monitoring_disabled"}
    assert delivery.events == []


@pytest.mark.parametrize("mode", ["telemetry", "ai_gateway"])
def test_logged_delivery_enrolls_only_after_both_parity_checks(delivery, mode):
    """The automatic entry must use the served version and payload table after verified smoke."""
    delivery.settings["logging_mode"] = mode
    result = _run(delivery, monitoring_values=delivery.values)
    assert result["verification"] == {"sampled": 1, "rest_compared": 1, "sql_compared": 1}
    assert "smoke" not in result
    assert result["monitoring"]["status"] == "enrolled"
    assert result["monitoring"]["models"] == 1
    assert delivery.events == [
        "endpoint",
        "rest",
        "sql",
        "guard",
        "reference",
        ("enroll", "2", True),
    ]
    [config] = delivery.inventory.values()
    assert (config["model_name"], config["model_version"], config["model_alias"]) == (
        "main.models.risk",
        "2",
        None,
    )
    assert config["source_table"] == config["prediction_table"] == "main.logs.risk_payload"
    assert (config["project"], config["environment"], config["serving_endpoint"]) == (
        "risk",
        "prod",
        "risk-endpoint",
    )


@pytest.mark.parametrize("channel", ["rest", "sql"])
def test_failed_smoke_does_not_enroll_or_retire(delivery, channel):
    """An updated endpoint remains inspectable while the previous monitor stays active."""
    getattr(delivery, channel).side_effect = ValueError("failed smoke")
    with pytest.raises(ValueError, match="failed smoke"):
        _run(delivery, monitoring_values=delivery.values)
    delivery.reference.assert_not_called()
    assert delivery.inventory == {}


def test_retry_retirement_preserves_batch_other_endpoints_and_other_owners(delivery):
    """Only superseded versions of this exact online owner are disabled, never deleted."""
    prior = MonitorConfig(
        environment="prod",
        project="risk",
        model_name="main.models.risk",
        model_version="1",
        source_table="main.logs.risk_payload",
        prediction_table="main.logs.risk_payload",
        execution_engine="spark",
        reference_namespace="ops.monitoring",
        serving_endpoint="risk-endpoint",
    )
    others = [
        replace(prior, serving_endpoint=None),
        replace(prior, serving_endpoint="other-endpoint"),
        replace(prior, project="other"),
        replace(prior, environment="dev"),
        replace(prior, model_version="9", enabled=False),
    ]
    delivery.inventory.update({c.monitor_id: c.payload() for c in [prior, *others]})
    first = _run(delivery, monitoring_values=delivery.values, model_version="2")
    second = _run(delivery, monitoring_values=delivery.values, model_version="2")
    assert first["monitoring"]["retired_monitor_ids"] == [prior.monitor_id]
    assert second["monitoring"]["retired_monitor_ids"] == []
    assert len(delivery.inventory) == 7
    assert delivery.inventory[prior.monitor_id]["enabled"] is False
    assert all(delivery.inventory[c.monitor_id] == c.payload() for c in others)
    delivery.registry.get_model_version_by_alias.assert_not_called()
    assert delivery.events.index(("enroll", "2", True)) < delivery.events.index(
        ("enroll", "1", False)
    )


def test_reference_failure_keeps_previous_enrollment_active(delivery):
    """Failed reference preparation cannot retire an otherwise healthy prior version."""
    prior = MonitorConfig(
        environment="prod",
        project="risk",
        model_name="main.models.risk",
        model_version="1",
        source_table="main.logs.risk_payload",
        prediction_table="main.logs.risk_payload",
        execution_engine="spark",
        reference_namespace="ops.monitoring",
        serving_endpoint="risk-endpoint",
    )
    delivery.inventory[prior.monitor_id] = prior.payload()
    delivery.reference.side_effect = ValueError("reference unavailable")
    with pytest.raises(ValueError, match="reference unavailable"):
        _run(delivery, monitoring_values=delivery.values)
    assert delivery.inventory == {prior.monitor_id: prior.payload()}


@pytest.mark.parametrize(
    "key,value",
    [
        ("monitoring_catalog", ""),
        ("monitoring_schema", "${var.schema}"),
        ("monitoring_schema", "a.b"),
        ("monitoring_expected_interval_hours", "0"),
        ("monitoring_drift_thresholds", "[]"),
        ("monitoring_label_table", "main.labels.events"),
        ("monitoring_performance_policies", "[]"),
    ],
)
def test_invalid_monitoring_settings_fail_before_paid_writes(delivery, key, value):
    """Namespace, policy and label errors must be actionable before changing endpoint state."""
    delivery.values[key] = value
    with pytest.raises(ValueError):
        _run(delivery, monitoring_values=delivery.values)
    delivery.endpoint.assert_not_called()
    assert delivery.inventory == {}


def test_disabled_logging_ignores_unused_monitoring_configuration(delivery):
    """An unlogged endpoint does not need valid shared monitoring settings."""
    delivery.settings["logging_mode"] = "disabled"
    delivery.values.update(monitoring_schema="${var.schema}", monitoring_drift_thresholds="bad")
    result = _run(delivery, monitoring_values=delivery.values)
    assert result["monitoring"]["reason"] == "logging_disabled"
    assert delivery.inventory == {}


@pytest.fixture
def saved_set(tmp_path, monkeypatch, delivery):
    """Read real saved component bytes and real MLflow metadata through mocked registry I/O."""
    import mlflow

    from skyulf.data.dataset import SplitDataset
    from skyulf.inference._manifest import ColumnSpec
    from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
    from skyulf.inference.model_set import ComponentReference, save_model_set
    from skyulf.integrations.databricks.jobs import serving_enrollment
    from skyulf.integrations.mlflow.models import model_set
    from skyulf.integrations.mlflow.registration.registry import ResolvedModel
    from skyulf.pipeline import SkyulfPipeline

    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0], "target": [2.0, 4.0, 6.0]})
    pipeline = SkyulfPipeline({"modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    component_path = tmp_path / "component"
    save_pipeline(pipeline, component_path)
    component = load_pipeline(component_path)
    components = {
        branch: (
            ComponentReference(
                name=f"main.models.{branch}",
                version=version,
                digest=component.manifest.pipeline_sha256,
            ),
            component_path,
        )
        for branch, version in (("risk", "3"), ("cost", "8"))
    }
    artifact = save_model_set(
        tmp_path / "set",
        components,
        record_key_schema=(ColumnSpec(name="record_id", dtype="int64"),),
    )
    package_path = tmp_path / "mlflow_model"
    mlflow.pyfunc.save_model(
        path=str(package_path), **model_set.model_set_save_options(artifact, tmp_path)
    )
    package = mlflow.models.Model.load(package_path)
    delivery.model.metadata = package
    delivery.settings["model_name"] = "main.models.combined"
    resolved = ResolvedModel(
        "main.models.combined",
        "2",
        "models:/main.models.combined/2",
        None,
        artifact.manifest.set_sha256,
    )
    resolve = Mock(return_value=resolved)
    monkeypatch.setattr(serving_enrollment, "resolve_model", resolve)
    monkeypatch.setattr(model_set, "_checked_set_reference", lambda *args: object())

    @contextmanager
    def downloaded(*args):
        """Substitute the download only; manifest parsing and digest checks remain real."""
        yield package_path, package

    monkeypatch.setattr(model_set, "downloaded_registered_payload", downloaded)
    return SimpleNamespace(artifact=artifact, package=package, path=package_path, resolve=resolve)


def test_model_set_enrolls_pinned_components_and_parent_identity(delivery, saved_set):
    """Branch monitors must follow the served package even if component aliases have moved."""
    previous = [
        MonitorConfig(
            environment="prod",
            project="risk",
            model_name=component.reference.name,
            model_version=component.reference.version,
            model_set_name="main.models.combined",
            model_set_version="1",
            model_set_branch=component.branch,
            source_table="main.logs.risk_payload",
            prediction_table="main.logs.risk_payload",
            serving_endpoint="risk-endpoint",
            execution_engine="spark",
            reference_namespace="ops.monitoring",
        )
        for component in saved_set.artifact.manifest.components
    ]
    delivery.inventory.update({c.monitor_id: c.payload() for c in previous})
    result = _run(delivery, monitoring_values=delivery.values, model_version="2")
    assert result["monitoring"]["models"] == 2
    configs = sorted(
        [c for c in delivery.inventory.values() if c["enabled"]],
        key=lambda item: item["model_set_branch"],
    )
    assert [
        (
            c["model_name"],
            c["model_version"],
            c["model_set_name"],
            c["model_set_version"],
            c["model_set_branch"],
        )
        for c in configs
    ] == [
        ("main.models.cost", "8", "main.models.combined", "2", "cost"),
        ("main.models.risk", "3", "main.models.combined", "2", "risk"),
    ]
    saved_set.resolve.assert_called_once_with(
        "main.models.combined", version="2", tracking_uri="databricks", registry_uri="databricks-uc"
    )
    delivery.registry.get_model_version_by_alias.assert_not_called()
    assert delivery.events[-6:] == [
        "reference",
        "reference",
        ("enroll", "8", True),
        ("enroll", "3", True),
        ("enroll", "8", False),
        ("enroll", "3", False),
    ]
    assert result["monitoring"]["retired_monitor_ids"] == [c.monitor_id for c in previous]


@pytest.mark.parametrize("failure", ["package_digest", "component_bytes"])
def test_model_set_attribution_rejects_changed_package_before_endpoint_write(
    delivery, saved_set, failure
):
    """An enrollment cannot attribute predictions to stale metadata or damaged components."""
    if failure == "package_digest":
        saved_set.package.metadata["model_set_digest"] = "f" * 64
    else:
        from skyulf.integrations.mlflow.registration.registry import packaged_artifact_path

        directory = packaged_artifact_path(saved_set.path, saved_set.package.flavors, "model_set")
        payload = directory / "components" / "risk" / "pipeline.pkl"
        payload.write_bytes(payload.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="digest|checksum|Checksum"):
        _run(delivery, monitoring_values=delivery.values, model_version="2")
    delivery.endpoint.assert_not_called()
    assert delivery.inventory == {}


def test_enrollment_batch_limit_is_checked_before_endpoint_write(delivery, monkeypatch):
    """The shared enrollment batch limit must fail before a large set changes paid serving."""
    from skyulf.integrations.databricks.jobs import serving_enrollment

    monkeypatch.setattr(
        serving_enrollment,
        "_served_identities",
        lambda *args: [
            {"model_name": f"main.models.model{i}", "model_version": "1"} for i in range(101)
        ],
    )
    with pytest.raises(ValueError, match="100"):
        _run(delivery, monitoring_values=delivery.values)
    delivery.endpoint.assert_not_called()
    assert delivery.inventory == {}


def test_notebook_passes_monitoring_settings_and_guard_provenance(delivery, monkeypatch):
    """The generated notebook must carry store settings and real run identity to enrollment."""
    from databricks import sdk
    from mlflow import tracking

    values = {
        **delivery.values,
        "inference_config_path": "inference.yml",
        "job_timeout_s": "3900",
        "catalog": "main",
        "input_schema": "source",
        "output_schema": "scores",
        "metadata_schema": "models",
        "resource_suffix": "",
        "job_run_id": "123",
        "task_key": "serving_endpoint",
        "workspace_host": "https://example.databricks.com",
    }
    monkeypatch.setattr(job, "read_serving_endpoint_settings", lambda values: delivery.settings)
    monkeypatch.setattr(
        job, "read_yaml_mapping", lambda path: {"score_source_table": "main.source.features"}
    )
    monkeypatch.setattr(sdk, "WorkspaceClient", Mock)
    monkeypatch.setattr(tracking, "MlflowClient", lambda **kwargs: delivery.registry)
    original = reference_mutations.reference_mutation_scope
    admission = Mock(side_effect=original)
    monkeypatch.setattr(reference_mutations, "reference_mutation_scope", admission)
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(getAll=lambda: values),
        jobs=SimpleNamespace(taskValues=SimpleNamespace(set=Mock())),
    )
    spark = object()
    result = job.run_serving_endpoint_notebook(spark, dbutils)
    admission.assert_called_once_with(
        spark,
        "ops.monitoring",
        job_run_id=123,
        task_key="serving_endpoint",
        workspace_host="https://example.databricks.com",
    )
    dbutils.jobs.taskValues.set.assert_called_once_with(key="serving_result", value=result)
    assert result["monitoring"]["status"] == "enrolled"


def test_unknown_single_model_performance_policy_fails_before_endpoint_write(delivery):
    """A misspelled model key must not silently disable a configured performance policy."""
    delivery.values["monitoring_performance_policies"] = json.dumps(
        {"main.models.riks": {"mode": "off"}}
    )
    with pytest.raises(ValueError, match="Performance policy model name"):
        _run(delivery, monitoring_values=delivery.values)
    delivery.endpoint.assert_not_called()
    assert delivery.inventory == {}


@pytest.mark.parametrize("policy_name", ["main.models.combined", "main.models.unknown"])
def test_model_set_rejects_policies_outside_saved_components(delivery, saved_set, policy_name):
    """Parent and unrelated names cannot replace the component policies used for monitoring."""
    delivery.values["monitoring_performance_policies"] = json.dumps({policy_name: {"mode": "off"}})
    with pytest.raises(ValueError, match="Performance policy model name"):
        _run(delivery, monitoring_values=delivery.values, model_version="2")
    delivery.endpoint.assert_not_called()
    assert delivery.inventory == {}


def test_named_component_performance_policy_is_retained(delivery, saved_set):
    """Valid component policy subsets must survive the shared enrollment contract."""
    delivery.values["monitoring_performance_policies"] = json.dumps(
        {"main.models.risk": {"mode": "off"}}
    )
    _run(delivery, monitoring_values=delivery.values, model_version="2")
    configs = {config["model_name"]: config for config in delivery.inventory.values()}
    assert configs["main.models.risk"]["performance_policy"] == {"mode": "off"}
    assert "performance_policy" not in configs["main.models.cost"]


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, {"tracking_uri": "databricks", "registry_uri": "databricks-uc"}),
        (
            {"tracking_uri": "databricks://alternate", "registry_uri": "databricks-uc://alternate"},
            {"tracking_uri": "databricks://alternate", "registry_uri": "databricks-uc://alternate"},
        ),
    ],
)
def test_reference_preparation_keeps_served_registry_selection(
    delivery, monkeypatch, options, expected
):
    """Reference loading must read the same registry as the served package, including custom profiles."""
    from skyulf.integrations.databricks.observability.monitoring.spark import (
        spark_monitoring_reference,
    )

    delivery.settings.update(options)
    monkeypatch.setattr(
        serving_registration,
        "prepare_spark_monitoring_reference",
        spark_monitoring_reference.prepare_spark_monitoring_reference,
    )
    monkeypatch.setattr(
        spark_monitoring_reference, "ensure_monitoring_store", lambda *args: "ops.monitoring"
    )
    loader = Mock(side_effect=RuntimeError("reference registry reached"))
    monkeypatch.setattr(spark_monitoring_reference, "load_monitoring_artifact", loader)
    with pytest.raises(RuntimeError, match="reference registry reached"):
        _run(delivery, monitoring_values=delivery.values)
    assert loader.call_count == 1
    assert loader.call_args.kwargs == expected
    assert delivery.inventory == {}
