"""Endpoint-only projects have no batch prediction table; monitoring preflight must not need one."""

import json

import pytest
from test_monitoring_registration import settings, workflow


def _endpoint_workflow(**changes):
    """An endpoint-only single model: no prediction table, no batch scoring."""
    config = workflow(
        **{"prediction_delivery": "endpoint", "training_layout": "single_model", **changes}
    )
    config.pop("prediction_table")
    return config


def test_endpoint_only_single_model_passes_monitoring_preflight():
    """Without this, training an endpoint-only single model with monitoring fails at initialize_run."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        validate_monitoring_settings,
    )

    assert validate_monitoring_settings(settings(), _endpoint_workflow()) is None


def test_endpoint_only_still_checks_destination_and_policy_names():
    """Skipping the batch config must keep the checks that do apply to endpoint monitoring."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        validate_monitoring_settings,
    )

    with pytest.raises(ValueError, match="monitoring_catalog"):
        validate_monitoring_settings(settings(monitoring_schema=""), _endpoint_workflow())
    policies = json.dumps({"models.risk.other": {"mode": "off"}})
    with pytest.raises(ValueError, match="model"):
        validate_monitoring_settings(
            settings(monitoring_performance_policies=policies), _endpoint_workflow()
        )


def test_batch_delivery_still_requires_prediction_table():
    """Batch and both deliveries keep the batch monitor check, so a missing table still fails."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        validate_monitoring_settings,
    )

    for delivery in ("batch", "both"):
        config = _endpoint_workflow(prediction_delivery=delivery)
        with pytest.raises(KeyError, match="prediction_table"):
            validate_monitoring_settings(settings(), config)
