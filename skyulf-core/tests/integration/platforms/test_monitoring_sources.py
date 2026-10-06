"""Observation windows trust only pinned scoring receipts and aware cutoffs."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest


def test_cutoff_requires_aware_utc_and_half_open_window():
    """Host timezone and future windows must not silently change observed populations."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_sources import (
        observation_window,
    )

    end = datetime(2026, 10, 1, tzinfo=UTC)
    assert observation_window(end, None, end)[0].day == 30
    with pytest.raises(ValueError):
        observation_window(end.replace(tzinfo=None), None, end)
    with pytest.raises(ValueError):
        observation_window(end, end, end)


def test_receipts_require_exact_source_target_and_model():
    """A matching run id from another source or model cannot provide feature provenance."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_sources import (
        validate_receipt,
    )

    receipt = {
        "run_id": "abc",
        "source_table_id": "source",
        "target_table_id": "target",
        "source_end_version": 4,
        "model_name": "a.b.model",
        "model_version": "2",
    }
    assert validate_receipt(receipt, "source", "target", "a.b.model", "2") == 4
    with pytest.raises(ValueError):
        validate_receipt(receipt, "recreated-source", "target", "a.b.model", "2")
    with pytest.raises(ValueError):
        validate_receipt(receipt, "source", "target", "a.b.model", "3")
    with pytest.raises(ValueError):
        validate_receipt(
            receipt | {"source_end_version": True}, "source", "target", "a.b.model", "2"
        )


def test_duplicate_receipts_are_rejected():
    """One prediction run must resolve to exactly one source snapshot."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_sources import (
        receipt_index,
    )

    rows = [
        {"version": 1, "timestamp": datetime(2026, 10, 1), "userMetadata": '{"run_id":"x"}'},
        {"version": 2, "timestamp": datetime(2026, 10, 1), "userMetadata": '{"run_id":"x"}'},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        receipt_index(rows)


def test_maintenance_metadata_is_ignored_but_unreceipted_writes_fail():
    """Opaque maintenance text cannot hide data changes or prevent valid receipt reads."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_sources import (
        receipt_index,
    )

    maintenance = {"version": 2, "operation": "OPTIMIZE", "userMetadata": "compaction"}
    receipt = {
        "version": 1,
        "operation": "WRITE",
        "userMetadata": '{"run_id":"x"}',
        "committed_us": 123,
    }
    assert list(receipt_index([receipt, maintenance])) == ["x"]
    with pytest.raises(ValueError, match="unreceipted"):
        receipt_index([receipt, maintenance | {"operation": "UPDATE"}])
    with pytest.raises(ValueError, match="unreceipted"):
        receipt_index([receipt, maintenance | {"operation": "DELETE", "userMetadata": None}])


def test_late_labels_do_not_move_prediction_snapshot_past_window(monkeypatch):
    """Later prediction replacement cannot erase the population selected for an earlier window."""
    import pandas as pd

    from skyulf.integrations.databricks.observability.monitoring import (
        monitoring_sources as sources,
    )
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
    )

    end = datetime(2026, 10, 1, tzinfo=UTC)
    snapshot = Mock(return_value=4)
    monkeypatch.setattr(sources, "snapshot_at", snapshot)
    monkeypatch.setattr(sources, "table_identity", lambda *args: "table-id")
    monkeypatch.setattr(sources, "_window_receipts", lambda *args: {})
    monkeypatch.setattr(sources, "read_snapshot", lambda *args: Mock())
    monkeypatch.setattr(sources, "_model_predictions", lambda *args: Mock(columns=[]))
    monkeypatch.setattr(sources.importlib, "import_module", lambda name: Mock())
    monkeypatch.setattr(
        sources, "bounded_frame", lambda *args: pd.DataFrame(columns=["id", "run_id", "prediction"])
    )
    monkeypatch.setattr(
        sources, "_read_matching_features", lambda *args: (pd.DataFrame(), [], None)
    )
    config = MonitorConfig(
        environment="test",
        project="history",
        model_name="a.b.model",
        model_version="1",
        source_table="a.b.source",
        prediction_table="a.b.predictions",
    )
    sources.read_current_observation(
        Mock(),
        config,
        "1",
        ("id",),
        ("x",),
        probabilities=0,
        as_of=end + timedelta(days=2),
        start=end - timedelta(days=1),
        end=end,
    )
    assert snapshot.call_args.args[2] == end - timedelta(microseconds=1)
