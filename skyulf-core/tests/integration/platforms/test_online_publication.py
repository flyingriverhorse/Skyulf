"""Native online publication validates source identity before requesting a sync."""

from contextlib import nullcontext
from dataclasses import asdict, replace
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def api():
    """Require the optional publication adapter rather than substituting native writes."""
    pytest.importorskip("databricks.sdk")
    return import_module("skyulf.integrations.databricks.feature_store.online_publication")


@pytest.fixture
def setup(api, monkeypatch):
    """Stub remote metadata while preserving adapter checks and exact native call arguments."""
    spec = api.OnlinePublicationSpec(
        "main.features.values", "main.online.values", "test-store", "table-id"
    )
    spark, fe, workspace = Mock(), Mock(), Mock()
    from databricks.sdk.errors import NotFound

    workspace.tables.get.side_effect = NotFound("missing")
    monkeypatch.setattr(api, "_workspace_client", lambda client: client or workspace)
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
            scheduling_policy="TRIGGERED",
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


def test_publication_mode_defaults_and_rejects_unknown_values(api, setup):
    """Only native supported modes may reach publication admission."""
    spec = setup[0]
    assert spec.publish_mode == "TRIGGERED"
    assert replace(spec, publish_mode="CONTINUOUS").publish_mode == "CONTINUOUS"
    with pytest.raises(ValueError, match="publish_mode"):
        replace(spec, publish_mode="continuous")


@pytest.fixture
def continuous(api, endpoint_publication):
    """Expose real public SDK field shapes for one continuous native publication."""
    spec, spark, fe, workspace, binding, table, synced = endpoint_publication
    spec = replace(spec, publish_mode="CONTINUOUS")
    synced.spec.scheduling_policy = "CONTINUOUS"
    pipeline = SimpleNamespace(
        pipeline_id="pipeline-id", state="RUNNING", spec=SimpleNamespace(continuous=True)
    )
    workspace.pipelines.get.return_value = pipeline
    workspace.pipelines.list_updates.return_value = SimpleNamespace(
        updates=[
            SimpleNamespace(
                pipeline_id="pipeline-id", update_id="update-id", creation_time=1, state="RUNNING"
            )
        ]
    )
    admission = SimpleNamespace(local_only=False, hold=lambda key: nullcontext())
    receipt = {
        "spec": asdict(spec),
        "pipeline_id": "pipeline-id",
        "submitted_at_ms": 0,
        "status": "SUBMITTED",
    }
    return spec, spark, fe, workspace, synced, pipeline, admission, receipt


def test_continuous_start_reuses_active_publication(api, continuous):
    """A repeated start must inspect the same pipeline without submitting another update."""
    spec, spark, fe, workspace, synced, pipeline, admission, receipt = continuous
    result = api.publish_online_features(
        spark, spec, feature_client=fe, client=workspace, admission=admission
    )
    fe.publish_table.assert_not_called()
    workspace.pipelines.start_update.assert_not_called()
    assert result["pipeline_id"] == receipt["pipeline_id"]
    assert result["status"] == "RUNNING"


def test_new_continuous_publication_uses_native_mode_and_releases_admission(api, setup):
    """Starting a continuous pipeline must release writer ownership after submission."""
    from contextlib import contextmanager

    from databricks.sdk.errors import NotFound

    spec, spark, fe, workspace, admission, checked = setup
    workspace.tables.get.side_effect = NotFound("missing")
    held = []

    @contextmanager
    def hold(key):
        """Record the short preflight mutation admission lifetime."""
        held.append(key)
        yield
        held.remove(key)

    admission.hold = hold
    fe.publish_table.side_effect = lambda **kwargs: (
        SimpleNamespace(online_table_name=spec.online_table, pipeline_id="pipeline-id")
        if held
        else pytest.fail("Native publication escaped writer admission")
    )
    result = api.publish_online_features(
        spark,
        replace(spec, publish_mode="CONTINUOUS"),
        client=workspace,
        feature_client=fe,
        admission=admission,
    )
    assert fe.publish_table.call_args.kwargs["publish_mode"] == "CONTINUOUS"
    assert result["status"] == "SUBMITTED"
    assert not held


@pytest.mark.parametrize(
    "state,expected",
    [("RUNNING", "RUNNING"), ("IDLE", "STOPPED"), ("FAILED", "FAILED"), ("STOPPING", "STOPPING")],
)
def test_continuous_state_is_independent_of_first_sync(api, continuous, state, expected):
    """Continuous activity is not a promise of first-sync completion or freshness."""
    spec, spark, fe, workspace, synced, pipeline, admission, receipt = continuous
    pipeline.state = state
    synced.data_synchronization_status.last_sync = None
    result = api.online_publication_status(workspace, receipt)
    assert result["status"] == expected
    assert result["first_sync_completed"] is False
    assert result["update_id"] == "update-id"


def test_continuous_first_sync_evidence_is_reported_separately(api, continuous):
    """Native first-sync evidence must remain distinct from a running pipeline."""
    workspace, receipt = continuous[3], continuous[-1]
    result = api.online_publication_status(workspace, receipt)
    assert result["status"] == "RUNNING"
    assert result["first_sync_completed"] is True
    assert result["last_sync_end_timestamp"] == "2026-10-11T09:00:00Z"


@pytest.mark.parametrize("mode", ["CONTINUOUS", "TRIGGERED"])
def test_mode_changes_never_recreate_existing_publication(api, continuous, mode):
    """Mode migration requires operator action instead of native implicit replacement."""
    spec, spark, fe, workspace, synced, pipeline, admission, receipt = continuous
    synced.spec.scheduling_policy = "TRIGGERED" if mode == "CONTINUOUS" else "CONTINUOUS"
    with pytest.raises(ValueError, match="mode.*operator|mode.*migration"):
        api.publish_online_features(
            spark,
            replace(spec, publish_mode=mode),
            client=workspace,
            feature_client=fe,
            admission=admission,
        )
    fe.publish_table.assert_not_called()
    workspace.pipelines.start_update.assert_not_called()


def test_stop_and_restart_keep_the_native_pipeline(api, continuous):
    """Lifecycle controls must stop and restart the existing stream without replacing it."""
    spec, spark, fe, workspace, synced, pipeline, admission, receipt = continuous
    result = api.stop_online_publication(workspace, receipt, admission=admission)
    workspace.pipelines.stop.assert_called_once_with(pipeline_id="pipeline-id")
    assert result["status"] == "STOP_REQUESTED"
    pipeline.state = "IDLE"
    workspace.pipelines.start_update.return_value = SimpleNamespace(update_id="restart-id")
    result = api.restart_online_publication(
        spark, receipt, client=workspace, feature_client=fe, admission=admission
    )
    workspace.pipelines.start_update.assert_called_once_with(pipeline_id="pipeline-id")
    fe.publish_table.assert_not_called()
    assert result["pipeline_id"] == "pipeline-id"
    assert result["update_id"] == "restart-id"


def test_stop_rejects_receipt_for_an_unrelated_pipeline(api, continuous):
    """A caller supplied pipeline ID cannot authorize mutations outside its publication."""
    spec, spark, fe, workspace, synced, pipeline, admission, receipt = continuous
    receipt["pipeline_id"] = "unrelated"
    with pytest.raises(ValueError, match="pipeline"):
        api.stop_online_publication(workspace, receipt, admission=admission)
    workspace.pipelines.stop.assert_not_called()


def test_ambiguous_continuous_start_is_not_retried(api, continuous):
    """A timeout must reach the operator without an automatic duplicate stream start."""
    spec, spark, fe, workspace, synced, pipeline, admission, receipt = continuous
    pipeline.state = "IDLE"
    workspace.pipelines.start_update.side_effect = TimeoutError("ambiguous")
    with pytest.raises(TimeoutError, match="ambiguous"):
        api.publish_online_features(
            spark, spec, client=workspace, feature_client=fe, admission=admission
        )
    workspace.pipelines.start_update.assert_called_once()
    fe.publish_table.assert_not_called()


def test_triggered_mode_allows_omitted_native_continuous_false(api, endpoint_publication):
    """SDK omission of a false continuous flag must preserve existing triggered publications."""
    spec, spark, fe, workspace, binding, table, synced = endpoint_publication
    workspace.pipelines.get.return_value = SimpleNamespace(
        pipeline_id="pipeline-id", state="IDLE", spec=SimpleNamespace(continuous=None)
    )
    receipt = api.publish_online_features(
        spark,
        spec,
        client=workspace,
        feature_client=fe,
        admission=SimpleNamespace(local_only=False, hold=lambda key: nullcontext()),
    )
    assert receipt["status"] == "SUBMITTED"
    fe.publish_table.assert_called_once()


@pytest.mark.parametrize("failure", ["update", "synchronization"])
def test_continuous_failure_is_visible_while_pipeline_reports_running(api, continuous, failure):
    """A recovering pipeline must not hide native update or data synchronization failure."""
    spec, spark, fe, workspace, synced, pipeline, admission, receipt = continuous
    if failure == "update":
        workspace.pipelines.list_updates.return_value.updates[0].state = "FAILED"
    else:
        synced.data_synchronization_status.detailed_state = "SYNCED_TABLE_ONLINE_PIPELINE_FAILED"
    result = api.online_publication_status(workspace, receipt)
    assert result["status"] == "FAILED"
    assert result["pipeline_state"] == "RUNNING"


def test_legacy_receipt_without_mode_remains_triggered(api, setup):
    """Saved receipts from the earlier adapter retain their native triggered status semantics."""
    spec, spark, fe, workspace, admission, checked = setup
    receipt = api.publish_online_features(spark, spec, feature_client=fe, admission=admission)
    del receipt["spec"]["publish_mode"]
    workspace.pipelines.list_updates.return_value = SimpleNamespace(
        updates=[
            SimpleNamespace(
                pipeline_id="pipeline-id",
                update_id="old-compatible",
                creation_time=receipt["submitted_at_ms"] + 1,
                state="COMPLETED",
            )
        ]
    )
    assert api.online_publication_status(workspace, receipt)["status"] == "SYNCED"


def test_running_pipeline_can_be_stopped_when_sync_has_failed(api, continuous):
    """Failed data sync must not prevent stopping native compute that remains running."""
    spec, spark, fe, workspace, synced, pipeline, admission, receipt = continuous
    synced.data_synchronization_status.detailed_state = "SYNCED_TABLE_ONLINE_PIPELINE_FAILED"
    result = api.stop_online_publication(workspace, receipt, admission=admission)
    workspace.pipelines.stop.assert_called_once_with(pipeline_id="pipeline-id")
    assert result["status"] == "STOP_REQUESTED"


@pytest.mark.parametrize("visible", ["old_failed", "none", "unrelated_new"])
def test_continuous_restart_waits_for_its_requested_update(api, continuous, monkeypatch, visible):
    """Eventual native listing must not replace a restart receipt with an earlier failure."""
    spec, spark, fe, workspace, synced, pipeline, admission, receipt = continuous
    monkeypatch.setattr(api.time, "time_ns", lambda: 200_000_000)
    pipeline.state = "IDLE"
    workspace.pipelines.start_update.return_value = SimpleNamespace(update_id="new-update")
    restarted = api.restart_online_publication(
        spark, receipt, client=workspace, admission=admission, feature_client=fe
    )
    pipeline.state = "RUNNING"
    old = workspace.pipelines.list_updates.return_value.updates[0]
    old.state = "FAILED"
    synced.data_synchronization_status.detailed_state = "SYNCED_TABLE_ONLINE_PIPELINE_FAILED"
    if visible == "none":
        workspace.pipelines.list_updates.return_value.updates = []
    elif visible == "unrelated_new":
        old.creation_time = 201
    result = api.online_publication_status(workspace, restarted)
    assert result["status"] == "PENDING"
    assert result["update_id"] == "new-update"
    assert result["submitted_at_ms"] == 200
    assert result["pipeline_state"] == "RUNNING"
    assert "update_state" not in result


def test_continuous_restart_reports_failure_when_requested_update_becomes_visible(
    api, continuous, monkeypatch
):
    """Correlation must suppress stale failures while preserving failure of the actual restart."""
    spec, spark, fe, workspace, synced, pipeline, admission, receipt = continuous
    monkeypatch.setattr(api.time, "time_ns", lambda: 200_000_000)
    pipeline.state = "IDLE"
    workspace.pipelines.start_update.return_value = SimpleNamespace(update_id="new-update")
    restarted = api.restart_online_publication(
        spark, receipt, client=workspace, admission=admission, feature_client=fe
    )
    pipeline.state = "RUNNING"
    update = workspace.pipelines.list_updates.return_value.updates[0]
    update.update_id, update.creation_time, update.state = "new-update", 201, "FAILED"
    result = api.online_publication_status(workspace, restarted)
    assert result["status"] == "FAILED"
    assert result["update_id"] == "new-update"
    assert result["update_state"] == "FAILED"


def test_continuous_restart_keeps_previous_sync_failure_separate_from_new_update(
    api, continuous, monkeypatch
):
    """A lagging synced-table failure label must not mark the acknowledged running restart failed."""
    spec, spark, fe, workspace, synced, pipeline, admission, receipt = continuous
    monkeypatch.setattr(api.time, "time_ns", lambda: 200_000_000)
    pipeline.state = "IDLE"
    workspace.pipelines.start_update.return_value = SimpleNamespace(update_id="new-update")
    restarted = api.restart_online_publication(
        spark, receipt, client=workspace, admission=admission, feature_client=fe
    )
    pipeline.state = "RUNNING"
    update = workspace.pipelines.list_updates.return_value.updates[0]
    update.update_id, update.creation_time, update.state = "new-update", 201, "RUNNING"
    synced.data_synchronization_status.detailed_state = "SYNCED_TABLE_ONLINE_PIPELINE_FAILED"
    result = api.online_publication_status(workspace, restarted)
    assert result["status"] == "RUNNING"
    assert result["update_id"] == "new-update"
    assert result["synchronization_state"] == "SYNCED_TABLE_ONLINE_PIPELINE_FAILED"
