"""Online observation identities must not collide with batch or another release."""

from dataclasses import replace
from typing import Any

import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
    MonitorConfig,
    json_digest,
)
from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_reference import (
    reference_id,
)


def options() -> dict[str, Any]:
    """Use one model with a native inference table and a durable training reference."""
    return {
        "environment": "test",
        "project": "online",
        "model_name": "a.b.model",
        "model_version": "1",
        "source_table": "a.b.payload",
        "prediction_table": "a.b.payload",
        "execution_engine": "spark",
        "reference_namespace": "a.monitoring",
    }


def test_legacy_monitoring_payload_and_identity_remain_unchanged():
    """A new optional source must preserve every existing batch receipt digest."""
    config = MonitorConfig(**options())
    assert "serving_endpoint" not in config.payload()
    assert config.monitor_id == json_digest(["test", "online", "a.b.model"])


def test_endpoint_and_version_each_get_independent_monitoring_history():
    """Batch, canary releases and distinct endpoints cannot overwrite enrollments."""
    batch = MonitorConfig(**options())
    online = MonitorConfig(**options(), serving_endpoint="risk-endpoint")
    other = replace(online, serving_endpoint="other-endpoint")
    newer = replace(online, model_version="2")
    assert len({c.monitor_id for c in (batch, online, other, newer)}) == 4
    assert MonitorConfig.from_dict(online.payload()) == online


def test_parent_releases_keep_distinct_component_monitoring():
    """A reused component cannot merge evidence from two served model-set versions."""
    config = MonitorConfig(
        **options(),
        serving_endpoint="risk-endpoint",
        model_set_name="a.b.group",
        model_set_version="1",
        model_set_branch="risk",
    )
    assert config.monitor_id != replace(config, model_set_version="2").monitor_id


@pytest.mark.parametrize(
    "changed, message",
    [
        ({"execution_engine": "local"}, "Spark"),
        ({"model_version": None, "model_alias": "champion"}, "concrete"),
        ({"prediction_table": "a.b.other"}, "same inference table"),
        ({"serving_endpoint": "bad/name"}, "endpoint"),
        ({"serving_endpoint": ""}, "endpoint"),
    ],
)
def test_invalid_online_sources_fail_before_remote_access(changed, message):
    """Ambiguous attribution must be rejected before observation or enrollment."""
    values: dict[str, Any] = {**options(), "serving_endpoint": "risk-endpoint", **changed}
    with pytest.raises(ValueError, match=message):
        MonitorConfig(**values)


def test_online_enrollment_reuses_the_same_verified_training_reference():
    """Endpoint identity must not force a second copy of immutable training data."""
    batch = MonitorConfig(**options())
    online = MonitorConfig(**options(), serving_endpoint="risk-endpoint")
    evidence = {"model_version": "1", "model_digest": "fixed-training-digest"}
    assert reference_id(batch, evidence) == reference_id(online, evidence)
