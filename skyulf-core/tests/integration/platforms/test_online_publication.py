"""Native online publication validates source identity before requesting a sync."""

from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def api():
    """Require the optional publication adapter rather than substituting native writes."""
    return import_module("skyulf.integrations.databricks.feature_store.online_publication")


@pytest.fixture
def setup(api, monkeypatch):
    """Stub remote metadata while preserving adapter checks and exact native call arguments."""
    spec = api.OnlinePublicationSpec(
        "main.features.values", "main.online.values", "test-store", "table-id"
    )
    spark, fe, workspace = Mock(), Mock(), Mock()
    spark.sql.return_value.first.return_value = {
        "format": "delta",
        "id": "table-id",
        "properties": {"delta.enableChangeDataFeed": "true"},
    }
    frame = spark.table.return_value
    frame.columns = ["entity", "value", "feature_time"]
    frame.schema.__getitem__ = Mock(return_value=SimpleNamespace(nullable=False))
    fe.get_table.return_value = SimpleNamespace(primary_keys=["entity"], timestamp_keys=[])
    fe.get_online_store.return_value = SimpleNamespace(name="test-store", state="AVAILABLE")
    fe.publish_table.return_value = SimpleNamespace(
        online_table_name="main.online.values", pipeline_id="pipeline-id"
    )
    snapshot = Mock(
        return_value={"table_name": spec.source_table, "table_id": "table-id", "version": 7}
    )
    monkeypatch.setattr(api, "snapshot_table", snapshot)
    checked = Mock()
    monkeypatch.setattr(api, "validate_feature_tables", checked)
    admission = SimpleNamespace(local_only=False, hold=lambda key: nullcontext())
    return spec, spark, fe, workspace, admission, checked


def test_publication_is_submitted_without_claiming_features_are_ready(api, setup):
    """Native acceptance is not a completed sync or proof of entity feature freshness."""
    spec, spark, fe, workspace, admission, checked = setup
    receipt = api.publish_online_features(spark, spec, feature_client=fe, admission=admission)
    checked.assert_called_once()
    fe.publish_table.assert_called_once_with(
        online_store=fe.get_online_store.return_value,
        source_table_name=spec.source_table,
        online_table_name=spec.online_table,
        publish_mode="TRIGGERED",
    )
    assert receipt["pipeline_id"] == "pipeline-id"
    assert receipt["source_snapshot"]["version"] == 7
    assert receipt["status"] == "SUBMITTED"


@pytest.mark.parametrize("case", ["cdf", "identity", "nullable", "temporal", "store", "local_lock"])
def test_invalid_source_rejected_before_publication(api, setup, case):
    """Wrong ownership or unsupported source contracts cannot start native writes."""
    spec, spark, fe, workspace, admission, checked = setup
    if case == "cdf":
        spark.sql.return_value.first.return_value["properties"] = {}
    elif case == "identity":
        spark.sql.return_value.first.return_value["id"] = "replacement"
    elif case == "nullable":
        spark.table.return_value.schema.__getitem__.return_value.nullable = True
    elif case == "temporal":
        fe.get_table.return_value.timestamp_keys = ["feature_time"]
    elif case == "store":
        fe.get_online_store.return_value.state = "STARTING"
    else:
        admission.local_only = True
    with pytest.raises(ValueError):
        api.publish_online_features(spark, spec, feature_client=fe, admission=admission)
    fe.publish_table.assert_not_called()


@pytest.mark.parametrize(
    "source,online,store,identity",
    [
        ("values", "main.online.values", "test", "id"),
        ("main.f.values", "main.f.values", "test", "id"),
        ("main.f.values", "main.o." + "a" * 64, "test", "id"),
        ("main.f.values", "main.o.values", "../test", "id"),
        ("main.f.values", "main.o.values", "test", ""),
    ],
)
def test_invalid_selectors_rejected(api, source, online, store, identity):
    """Unsafe identifiers fail without creating or resolving any cloud resource."""
    with pytest.raises(ValueError):
        api.OnlinePublicationSpec(source, online, store, identity)


@pytest.mark.parametrize(
    "state,offset,expected",
    [
        ("COMPLETED", 1, "SYNCED"),
        ("RUNNING", 1, "PENDING"),
        ("COMPLETED", -1, "PENDING"),
        ("FAILED", 1, "FAILED"),
    ],
)
def test_status_requires_a_new_completed_pipeline_update(api, setup, state, offset, expected):
    """A previous successful sync cannot be presented as the current publication's success."""
    spec, spark, fe, workspace, admission, checked = setup
    receipt = api.publish_online_features(spark, spec, feature_client=fe, admission=admission)
    workspace.pipelines.list_updates.return_value = SimpleNamespace(
        updates=[
            SimpleNamespace(
                pipeline_id="pipeline-id",
                update_id="update-id",
                creation_time=receipt["submitted_at_ms"] + offset,
                state=state,
            )
        ]
    )
    result = api.online_publication_status(workspace, receipt)
    assert result["status"] == expected


def test_source_replacement_during_preflight_rejected(api, setup, monkeypatch):
    """The validated pinned rows must belong to the same source at the native publish call."""
    spec, spark, fe, workspace, admission, checked = setup
    monkeypatch.setattr(
        api,
        "snapshot_table",
        Mock(
            side_effect=[
                {"table_name": spec.source_table, "table_id": "table-id", "version": 7},
                {"table_name": spec.source_table, "table_id": "replacement", "version": 0},
            ]
        ),
    )
    with pytest.raises(ValueError, match="changed"):
        api.publish_online_features(spark, spec, feature_client=fe, admission=admission)
    fe.publish_table.assert_not_called()


def test_completed_native_update_requires_its_receipt_id(api, setup):
    """A completed status without an inspectable native update cannot certify the sync."""
    spec, spark, fe, workspace, admission, checked = setup
    receipt = api.publish_online_features(spark, spec, feature_client=fe, admission=admission)
    workspace.pipelines.list_updates.return_value = SimpleNamespace(
        updates=[
            SimpleNamespace(
                pipeline_id="pipeline-id",
                update_id=None,
                creation_time=receipt["submitted_at_ms"] + 1,
                state="COMPLETED",
            )
        ]
    )
    with pytest.raises(ValueError, match="update"):
        api.online_publication_status(workspace, receipt)
