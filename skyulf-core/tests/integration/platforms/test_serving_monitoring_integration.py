"""Serving observations retain saved model schemas and event-key performance identity."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.observability.monitoring import (
    monitoring_performance as monitoring_performance,
)
from skyulf.integrations.databricks.observability.monitoring.monitoring_config import MonitorConfig


def config(**changes):
    """Build one concrete online model with the same raw source and output table."""
    values: dict[str, Any] = {
        "environment": "test",
        "project": "online",
        "model_name": "a.b.model",
        "model_version": "1",
        "source_table": "a.b.payload",
        "prediction_table": "a.b.payload",
        "execution_engine": "spark",
        "reference_namespace": "a.monitoring",
        "serving_endpoint": "endpoint",
    }
    return MonitorConfig(**(values | changes))


def artifact():
    """Use a saved regression schema without loading optional MLflow or Spark."""
    return SimpleNamespace(
        manifest=SimpleNamespace(
            input_columns=("amount", "count"),
            input_dtypes=("float64", "Int64"),
            task="regression",
            classes=(),
            classification_probabilities=False,
            pipeline_sha256="a" * 64,
        ),
        pipeline=SimpleNamespace(config={}),
    )


def test_serving_reads_late_arrivals_at_cutoff_and_attributes_parent_version(monkeypatch):
    """Log ingestion time must not replace event time or a model-set component version."""
    from skyulf.integrations.databricks.observability.monitoring.serving import serving_observation

    now = datetime(2026, 10, 6, tzinfo=UTC)
    start, end = now - timedelta(days=2), now - timedelta(days=1)
    selected = config(model_set_name="a.b.group", model_set_version="3", model_set_branch="risk")
    spark, payload, entities = Mock(), object(), object()
    spark.table.return_value = entities
    current, predictions = object(), object()
    parser = Mock(return_value=(current, predictions, {"request_count": 3}, end))
    capture_evidence = {"prediction_version": 7, "source_table_id": "payload-id"}
    read = Mock(return_value=(payload, capture_evidence))
    verify = Mock()
    monkeypatch.setattr(serving_observation, "read_serving_payload_snapshot", read)
    monkeypatch.setattr(serving_observation, "verify_serving_payload_snapshot", verify)
    monkeypatch.setattr(serving_observation, "parse_serving_payloads", parser)
    result = serving_observation.read_serving_observation(
        spark,
        selected,
        "1",
        ("databricks_request_id", "request_row_index"),
        ("amount", "count"),
        probabilities=0,
        as_of=now,
        start=start,
        end=end,
        artifact=artifact(),
    )
    read.assert_called_once_with(spark, "a.b.payload", now)
    verify.assert_called_once_with(spark, "a.b.payload", capture_evidence)
    kwargs = parser.call_args.kwargs
    assert kwargs["model_name"] == "a.b.group" and kwargs["model_version"] == "3"
    assert kwargs["input_columns"] == (("amount", "double"), ("count", "long"))
    assert kwargs["output_columns"] == (
        ("risk__prediction", "double"),
        ("risk__scoring_status", "string"),
    )
    assert result[:2] == (current, predictions)
    assert result[2]["prediction_version"] == 7
    assert result[2]["window_basis"] == "serving_request_timestamp"


def test_serving_performance_contract_separates_endpoints_and_parent_versions():
    """A production baseline from another endpoint or release cannot qualify a policy."""
    policy = {
        "mode": "report",
        "metric": "mae",
        "direction": "lower",
        "baseline": {"kind": "training_holdout", "model_version": "1"},
        "tolerance": 0.1,
        "window_hours": 24,
        "minimum_labeled_rows": 2,
        "tolerance_mode": "absolute",
        "label_delay_hours": 0,
        "minimum_label_coverage": 0.8,
        "consecutive_windows": 1,
    }
    values = {
        "label_table": "a.b.labels",
        "result_available_at_column": "available_at",
        "performance_policy": policy,
        "model_set_name": "a.b.group",
        "model_set_version": "1",
        "model_set_branch": "risk",
    }
    first = config(**values)
    second = config(**(values | {"serving_endpoint": "another"}))
    third = config(**(values | {"model_set_version": "2"}))
    spec = SimpleNamespace(target_column="target", record_key_columns=("company_id",))
    digests = {
        monitoring_performance.performance_contract(artifact(), spec, c)
        for c in (first, second, third)
    }
    assert len(digests) == 3


def test_performance_measurement_uses_event_keys_without_changing_training_spec(monkeypatch):
    """Repeated customer requests need separate outcome joins, with training keys retained."""
    from skyulf.integrations.databricks.observability.monitoring.spark import (
        spark_monitoring_metrics,
    )

    measure = Mock(return_value={"labeled_rows": 2})
    monkeypatch.setattr(spark_monitoring_metrics, "build_spark_performance_report", measure)
    spec = SimpleNamespace(target_column="target", record_key_columns=("company_id",))
    monitoring_performance._measure(
        object(), object(), artifact(), spec, config(), datetime.now(UTC)
    )
    assert measure.call_args.kwargs["record_key_columns"] == (
        "databricks_request_id",
        "request_row_index",
    )
    assert spec.record_key_columns == ("company_id",)


def test_serving_rejects_replaced_table_before_publishing_evidence(monkeypatch):
    """A table recreated during the read cannot retain the original source identity."""
    from skyulf.integrations.databricks.observability.monitoring.serving import serving_observation

    now = datetime(2026, 10, 6, tzinfo=UTC)
    monkeypatch.setattr(
        serving_observation, "read_serving_payload_snapshot", Mock(return_value=(None, {}))
    )
    monkeypatch.setattr(
        serving_observation,
        "verify_serving_payload_snapshot",
        Mock(side_effect=ValueError("Serving inference table was replaced during observation.")),
    )
    monkeypatch.setattr(
        serving_observation, "parse_serving_payloads", Mock(return_value=(None, None, {}, now))
    )
    with pytest.raises(ValueError, match="replaced during observation"):
        serving_observation.read_serving_observation(
            Mock(),
            config(),
            "1",
            ("databricks_request_id", "request_row_index"),
            ("amount", "count"),
            probabilities=0,
            as_of=now,
            start=now - timedelta(days=1),
            end=now,
            artifact=artifact(),
        )


def test_online_observation_routes_event_keys_and_operational_evidence(monkeypatch):
    """Scheduled online reports must use actual events for both metrics and delayed labels."""
    from skyulf.integrations.databricks.observability.monitoring import monitoring as monitoring
    from skyulf.integrations.databricks.observability.monitoring.spark import (
        spark_monitoring_metrics,
        spark_monitoring_reference,
        spark_monitoring_sources,
    )

    saved = artifact()
    spec = SimpleNamespace(record_key_columns=("company_id",), target_column="target")
    reference, current, predictions, labels = object(), object(), object(), object()
    now = datetime(2026, 10, 6, tzinfo=UTC)
    serving = {"request_count": 4, "success_count": 3, "error_count": 1}
    reader = Mock(return_value=(current, predictions, {"serving": serving}, now))
    label_reader = Mock(return_value=(labels, {}))
    builder = Mock(return_value={"status": "healthy", "metrics": []})
    row_builder = Mock(return_value={"report_id": "report"})
    monkeypatch.setattr(
        spark_monitoring_reference,
        "load_spark_monitoring_reference",
        Mock(return_value=(saved, spec, reference, {"model_version": "1"})),
    )
    monkeypatch.setattr(monitoring, "read_serving_observation", reader)
    monkeypatch.setattr(spark_monitoring_sources, "read_spark_labels", label_reader)
    monkeypatch.setattr(spark_monitoring_metrics, "build_spark_monitoring_report", builder)
    monkeypatch.setattr(monitoring, "result_row", row_builder)
    monkeypatch.setattr(monitoring, "_log_observation", Mock(return_value=None))
    monitoring.observe_model(
        Mock(),
        config(),
        as_of=now,
        window_start=now - timedelta(days=1),
        window_end=now,
        tracking_uri=None,
        registry_uri=None,
        experiment_name=None,
    )
    keys = ("databricks_request_id", "request_row_index")
    assert reader.call_args.args[3] == keys
    assert reader.call_args.kwargs["artifact"] is saved
    assert label_reader.call_args.args[2] == keys
    assert builder.call_args.kwargs["record_key_columns"] == keys
    assert row_builder.call_args.args[5]["serving"] == serving
    assert spec.record_key_columns == ("company_id",)
