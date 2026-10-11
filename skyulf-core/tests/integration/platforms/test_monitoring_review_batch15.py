"""Monitoring keeps label availability and producer budgets independent."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.monitoring import monitoring_sources as sources
from skyulf.integrations.databricks.observability.monitoring.monitoring_config import MonitorConfig
from skyulf.integrations.databricks.observability.monitoring.monitoring_metrics import (
    build_monitoring_report,
)
from skyulf.integrations.databricks.scoring.incremental.incremental_batch import bounded_frame

AS_OF = datetime(2026, 10, 4, 12, tzinfo=UTC)


def _selected(records):
    """Keep the real bounded reader while replacing only Spark transport."""
    selected = Mock()
    selected.select.return_value = selected
    selected.orderBy.return_value = selected
    selected.join.return_value = selected
    selected.schema = {"available_at": Mock(dataType=Mock(typeName=lambda: "string"))}

    def limited(count):
        """Honor max_rows plus one instead of silently bypassing the transfer bound."""
        result = Mock()
        result.toLocalIterator.return_value = [
            Mock(asDict=Mock(return_value=row)) for row in records[:count]
        ]
        return result

    selected.limit.side_effect = limited
    return selected


def _read(monkeypatch, records, **budgets):
    """Run the public source loader with pinned-version metadata and real key checks."""
    config = MonitorConfig(
        environment="test",
        project="revision",
        model_name="a.b.model",
        model_version="1",
        source_table="a.b.features",
        prediction_table="a.b.predictions",
        label_table="a.b.labels",
        result_available_at_column="available_at",
        **budgets,
    )
    monkeypatch.setattr(sources, "snapshot_at", lambda *args: 7)
    monkeypatch.setattr(sources, "table_identity", lambda *args: "label-table-id")
    monkeypatch.setattr(sources, "read_snapshot", lambda *args: _selected(records))
    return sources.read_labels(
        Mock(), config, ("id",), "target", pd.DataFrame({"id": [1, 2]}), AS_OF
    )


def _report(labels):
    """Exercise availability filtering and uniqueness through the public metrics API."""
    features = pd.DataFrame({"id": [1, 2], "x": [2.0, 3.0]})
    return build_monitoring_report(
        features,
        features,
        pd.DataFrame({"id": [1, 2], "prediction": [5.0, 9.0]}),
        labels,
        feature_columns=("x",),
        record_key_columns=("id",),
        target_column="target",
        result_available_at_column="available_at",
        as_of=AS_OF,
        task="regression",
    )


@pytest.mark.parametrize("future_count", [1, 2])
def test_future_label_revisions_do_not_hide_available_truth(monkeypatch, future_count):
    """Future revisions must not reject an otherwise unique, already available label."""
    records = [{"id": 1, "target": 4.0, "available_at": "2026-10-04T11:00:00Z"}]
    records += [
        {"id": 1, "target": 999.0, "available_at": "2026-10-05T12:00:00+03:00"}
    ] * future_count
    records.append({"id": 2, "target": 8.0, "available_at": "2026-10-04T11:00:00Z"})
    labels, evidence = _read(monkeypatch, records)
    result = _report(labels)
    assert result["labeled_rows"] == 2
    assert result["label_coverage"] == 1.0
    assert evidence == {
        "label_table": "a.b.labels",
        "label_version": 7,
        "label_table_id": "label-table-id",
    }
    mae = next(item for item in result["metrics"] if item["metric_name"] == "mae")
    assert mae["value"] == 1.0


def test_two_available_label_revisions_remain_ambiguous(monkeypatch):
    """Deferring the key check must not invent a winner among eligible duplicates."""
    records = [
        {"id": 1, "target": target, "available_at": "2026-10-04T11:00:00Z"} for target in (4.0, 9.0)
    ]
    labels, _ = _read(monkeypatch, records)
    with pytest.raises(ValueError, match="labels has duplicate record keys"):
        _report(labels)


@pytest.mark.parametrize("available", [None, "2026-10-04T11:00:00", "invalid"])
def test_invalid_label_availability_remains_an_error(monkeypatch, available):
    """Malformed availability cannot be treated as a future label and ignored."""
    labels, _ = _read(monkeypatch, [{"id": 1, "target": 4.0, "available_at": available}])
    with pytest.raises(ValueError, match="availability"):
        _report(labels)


@pytest.mark.parametrize("budget", [{"max_rows": 1}, {"max_bytes": 1}])
def test_label_revision_transfer_remains_bounded(monkeypatch, budget):
    """Ineligible revisions still consume bounded driver transfer capacity."""
    records = [{"id": 1, "target": 4.0, "available_at": "2026-10-05T11:00:00Z"}] * 2
    with pytest.raises(ValueError, match="exceeds max_"):
        _read(monkeypatch, records, **budget)


def test_ordinary_bounded_reader_still_rejects_duplicate_keys():
    """Feature and prediction readers retain their original strict uniqueness guard."""
    with pytest.raises(ValueError, match="globally unique"):
        bounded_frame(_selected([{"id": 1}, {"id": 1}]), ("id",), ("id",), 10, 10000)


@pytest.mark.parametrize("enabled", ["true", "false"])
@pytest.mark.parametrize(
    "changes,rows,bytes_",
    [
        ({"max_rows": 2_000_000}, 1_000_000, 4 * 1024**2),
        ({"max_input_mb": 2048}, 500, 1024**3),
        ({"max_bytes": 2 * 1024**3}, 500, 1024**3),
    ],
)
def test_enrollment_caps_monitor_budgets_without_reducing_producer(changes, rows, bytes_, enabled):
    """A larger producer allowance must not prevent a smaller monitoring observation."""
    from copy import deepcopy

    from test_monitoring_registration import settings, workflow

    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        build_monitor_enrollment_config,
        validate_monitoring_settings,
    )

    producer = workflow(**changes)
    before = deepcopy(producer)
    values = settings(monitoring_enabled=enabled)
    validate_monitoring_settings(values, producer)
    config = build_monitor_enrollment_config(producer, values, "2")
    assert (config.max_rows, config.max_bytes) == (rows, bytes_)
    assert config.enabled is (enabled == "true")
    assert producer == before


@pytest.mark.parametrize(
    "changes",
    [
        {"max_rows": True},
        {"max_rows": 2e6},
        {"max_rows": -1},
        {"max_bytes": True},
        {"max_bytes": 2e10},
        {"max_bytes": 0},
    ],
)
def test_enrollment_does_not_hide_invalid_producer_budget_types(changes):
    """Clamping must never convert booleans, floats or invalid bounds into valid integers."""
    from test_monitoring_registration import settings, workflow

    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        build_monitor_enrollment_config,
    )

    with pytest.raises(ValueError, match="positive integer"):
        build_monitor_enrollment_config(workflow(**changes), settings(), "1")


@pytest.mark.parametrize("limits", [{"max_rows": 1_000_001}, {"max_bytes": 1024**3 + 1}])
def test_explicit_monitor_config_keeps_driver_budget_caps(limits):
    """Separating producer budgets must not remove the monitoring SDK's memory protection."""
    with pytest.raises(ValueError, match="at most"):
        MonitorConfig(
            environment="test",
            project="bounded",
            model_name="a.b.model",
            model_version="1",
            source_table="a.b.features",
            prediction_table="a.b.predictions",
            **limits,
        )


@pytest.mark.parametrize("enabled", ["true", "false"])
@pytest.mark.parametrize("activation", [True, False])
def test_large_producer_can_register_or_pause_monitor(monkeypatch, enabled, activation):
    """Both activation and scoring retain explicit enable/disable semantics with large producers."""
    from test_monitoring_registration import settings, workflow

    from skyulf.integrations.databricks.observability.monitoring import (
        monitoring_registration as registration,
    )

    enroll = Mock()
    monkeypatch.setattr(registration, "ensure_monitoring_store", Mock())
    monkeypatch.setattr(registration, "enroll_monitor", enroll)
    producer = workflow(max_rows=2_000_000, max_input_mb=2048)
    values = settings(monitoring_enabled=enabled)
    if activation:
        result = registration.register_deployed_monitor(
            Mock(),
            producer,
            values,
            {
                "action": "train",
                "result": {
                    "alias_change": {
                        "kind": "promotion",
                        "model_name": producer["model_name"],
                        "new_version": "2",
                    }
                },
            },
            activation_started_ms=123,
        )
    else:
        result = registration.register_scoring_monitor(
            Mock(),
            producer,
            values,
            {
                "result": {
                    "selected_model_name": producer["model_name"],
                    "selected_model_version": "2",
                }
            },
        )
    assert result is not None
    config = enroll.call_args.args[2]
    assert (config.max_rows, config.max_bytes) == (1_000_000, 1024**3)
    assert config.enabled is (enabled == "true")
    assert producer["max_rows"] == 2_000_000
