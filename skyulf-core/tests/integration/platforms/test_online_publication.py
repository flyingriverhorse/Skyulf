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


@pytest.fixture
def endpoint_publication(setup):
    """Mirror native UC publication metadata without confusing UC and Delta identities."""
    from skyulf.integrations.databricks.feature_store.config import (
        FeatureLookupSpec,
        FeatureTrainingSpec,
    )
    from skyulf.integrations.databricks.feature_store.lifecycle_config import serialize_feature_spec

    spec, spark, fe, workspace, admission, checked = setup
    binding = {
        "version": 1,
        "lookup_spec": serialize_feature_spec(
            FeatureTrainingSpec(
                lookups=(
                    FeatureLookupSpec(
                        table_name=spec.source_table,
                        lookup_key=("entity",),
                        feature_names=("value", "feature_time"),
                    ),
                ),
                label=None,
            )
        ),
        "lookup_evidence": {
            "policy": "training_snapshot",
            "feature_tables": [
                {"table_name": spec.source_table, "table_id": spec.source_table_id, "version": 1}
            ],
        },
    }
    table = SimpleNamespace(
        full_name=spec.online_table,
        browse_only=False,
        properties={"source_table": spec.source_table, "source_table_id": "uc-id"},
        pipeline_id="pipeline-id",
    )
    workspace.catalogs.list.return_value = [SimpleNamespace(name="main")]
    workspace.schemas.list.return_value = [SimpleNamespace(name="online")]
    workspace.tables.list.return_value = [table]
    workspace.tables.get.side_effect = lambda name: (
        SimpleNamespace(table_id="uc-id") if name == spec.source_table else table
    )
    synced = SimpleNamespace(
        name=spec.online_table,
        effective_database_instance_name=spec.online_store,
        spec=SimpleNamespace(
            source_table_full_name=spec.source_table,
            primary_key_columns=["entity"],
            timeseries_key=None,
        ),
        unity_catalog_provisioning_state="ACTIVE",
        data_synchronization_status=SimpleNamespace(
            pipeline_id="pipeline-id",
            detailed_state="SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE",
            last_sync=SimpleNamespace(sync_end_timestamp="2026-10-11T09:00:00Z"),
        ),
    )
    workspace.database.get_synced_database_table.return_value = synced
    return spec, spark, fe, workspace, binding, table, synced


def test_setup_accepts_external_publication_without_republishing(api, endpoint_publication):
    """A readable existing copy with the same source identity needs no publisher job."""
    spec, spark, fe, workspace, binding, table, synced = endpoint_publication
    api.validate_online_publications(spark, binding, (spec,), client=workspace, feature_client=fe)
    fe.publish_table.assert_not_called()
    workspace.database.get_synced_database_table.assert_called_once_with(name=spec.online_table)


@pytest.mark.parametrize("feature_keys", [("entity",), ("entity", "tenant")])
def test_setup_accepts_mapped_request_keys(api, endpoint_publication, feature_keys):
    """Native lookups map request column names positionally onto feature-table primary keys."""
    spec, spark, fe, workspace, binding, table, synced = endpoint_publication
    request_keys = [f"request_{key}" for key in feature_keys]
    binding["lookup_spec"]["lookups"][0]["lookup_key"] = request_keys
    fe.get_table.return_value.primary_keys = list(feature_keys)
    spark.table.return_value.columns = [*feature_keys, "value", "feature_time"]
    synced.spec.primary_key_columns = list(feature_keys)

    api.validate_online_publications(spark, binding, (spec,), client=workspace, feature_client=fe)

    fe.publish_table.assert_not_called()
    assert binding["lookup_spec"]["lookups"][0]["lookup_key"] == request_keys


@pytest.mark.parametrize("failure", ["key_count", "key_order"])
def test_setup_rejects_incompatible_mapped_keys(api, endpoint_publication, failure):
    """Different native primary-key counts or ordering must not change saved request routing."""
    spec, spark, fe, workspace, binding, table, synced = endpoint_publication
    binding["lookup_spec"]["lookups"][0]["lookup_key"] = ["request_entity", "request_tenant"]
    if failure == "key_order":
        fe.get_table.return_value.primary_keys = ["entity", "tenant"]
        spark.table.return_value.columns = ["entity", "tenant", "value", "feature_time"]
        synced.spec.primary_key_columns = ["tenant", "entity"]

    with pytest.raises(ValueError, match="Publication primary keys"):
        api.validate_online_publications(
            spark, binding, (spec,), client=workspace, feature_client=fe
        )


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "multiple",
        "wrong_store",
        "wrong_delta",
        "wrong_uc",
        "keys",
        "initial_sync",
        "failed",
    ],
)
def test_setup_rejects_invalid_or_ambiguous_publication(api, endpoint_publication, failure):
    """Endpoint creation must not route through an unverified native online copy."""
    spec, spark, fe, workspace, binding, table, synced = endpoint_publication
    if failure == "missing":
        workspace.tables.list.return_value = []
    elif failure == "multiple":
        workspace.tables.list.return_value = [
            table,
            SimpleNamespace(full_name="main.other.copy", properties=table.properties),
        ]
    elif failure == "wrong_store":
        synced.effective_database_instance_name = "other-store"
    elif failure == "wrong_delta":
        binding["lookup_evidence"]["feature_tables"][0]["table_id"] = "old-delta"
    elif failure == "wrong_uc":
        table.properties["source_table_id"] = "old-uc"
    elif failure == "keys":
        synced.spec.primary_key_columns = ["different_key"]
    elif failure == "initial_sync":
        synced.data_synchronization_status.last_sync = None
    else:
        synced.data_synchronization_status.detailed_state = "SYNCED_TABLE_ONLINE_PIPELINE_FAILED"
    with pytest.raises(ValueError, match="publication|Publication"):
        api.validate_online_publications(
            spark, binding, (spec,), client=workspace, feature_client=fe
        )
    fe.publish_table.assert_not_called()


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
