"""Distributed monitoring selection preserves legacy identities and local safeguards."""

import pytest

from skyulf.integrations.databricks.monitoring_config import MonitorConfig


def fields():
    """Supply the stable legacy identity used by existing producer receipts."""
    return {
        "environment": "test",
        "project": "spark",
        "model_name": "a.b.model",
        "model_version": "1",
        "source_table": "a.b.source",
        "prediction_table": "a.b.predictions",
    }


def test_default_payload_does_not_change_legacy_identity():
    """An engine upgrade must not invalidate old local observation and request digests."""
    config = MonitorConfig(**fields())
    assert config.execution_engine == "local"
    assert "execution_engine" not in config.payload()
    assert "reference_namespace" not in config.payload()


def test_spark_requires_explicit_reference_store_and_valid_engine():
    """Missing durable reference configuration must fail before distributed observation."""
    with pytest.raises(ValueError, match="reference_namespace"):
        MonitorConfig(**fields(), execution_engine="spark")
    with pytest.raises(ValueError, match="execution_engine"):
        MonitorConfig(**fields(), execution_engine="unknown")
    config = MonitorConfig(**fields(), execution_engine="spark", reference_namespace="a.monitoring")
    assert config.payload()["execution_engine"] == "spark"
    assert config.payload()["reference_namespace"] == "a.monitoring"


def test_spark_enrollment_budget_is_independent_of_training_and_scoring():
    """Large scoring allowances cannot become monitoring driver-transfer allowances."""
    from skyulf.integrations.databricks.monitoring_registration import (
        build_monitor_enrollment_config,
    )

    config = build_monitor_enrollment_config(
        {
            "model_name": "a.b.model",
            "score_source_table": "a.b.source",
            "prediction_table": "a.b.predictions",
            "max_rows": 9_000_000,
            "max_input_mb": 4096,
        },
        {
            "monitoring_execution_engine": "spark",
            "monitoring_catalog": "a",
            "monitoring_schema": "monitoring",
            "monitoring_environment": "test",
            "monitoring_project": "spark",
        },
        "1",
    )
    assert config.execution_engine == "spark"
    assert config.reference_namespace == "a.monitoring"
    assert (config.max_rows, config.max_bytes) == (10000, 64 * 1024**2)
