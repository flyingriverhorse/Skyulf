"""Native lookup admission preserves the incremental source and package contracts."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.integration.platforms.test_local_cdf_recovery import recovery_case as recovery_case


def _binding():
    """Freeze a named lookup and one exact training feature-table snapshot."""
    return {
        "version": 1,
        "lookup_spec": {
            "lookups": [
                {
                    "table_name": "catalog.schema.features",
                    "lookup_key": ["entity"],
                    "feature_names": ["amount"],
                    "timestamp_lookup_key": "event_time",
                    "timestamp_type": "timestamp",
                    "lookback_seconds": None,
                }
            ],
            "label": "target",
            "exclude_columns": ["id", "entity", "event_time"],
        },
        "lookup_evidence": {
            "feature_tables": [
                {
                    "table_name": "catalog.schema.features",
                    "table_id": "feature-id",
                    "version": 7,
                }
            ],
            "policy": "training_snapshot",
        },
    }


def _artifact(binding=None, dtype="float64"):
    """Expose the immutable metadata carried by a verified registry artifact."""
    return SimpleNamespace(
        feature_lookup_json=json.dumps(binding or _binding()),
        manifest=SimpleNamespace(
            input_columns=("amount", "direct"), input_dtypes=(dtype, "float64")
        ),
    )


def test_feature_source_columns_require_lookup_controls_and_direct_features():
    """Automatic lookup cannot select absent model features before the SDK join."""
    from skyulf.integrations.databricks.feature_store import scoring

    assert scoring.feature_source_columns(_artifact(), ("id",)) == (
        "id",
        "direct",
        "entity",
        "event_time",
    )


def test_ordinary_source_columns_remain_exact():
    """Legacy artifacts retain their original source projection and ordering."""
    from skyulf.integrations.databricks.feature_store import scoring

    artifact = _artifact()
    artifact.feature_lookup_json = None
    assert scoring.feature_source_columns(artifact, ("id",)) == ("id", "amount", "direct")


@pytest.mark.parametrize("dtype", ["Int32", "Int64", "boolean", "Boolean"])
def test_native_lookup_rejects_nullable_codec_before_sdk(dtype):
    """SDK lookup has no pre-Arrow encoder to preserve nullable integer precision."""
    from skyulf.integrations.databricks.feature_store import scoring

    with pytest.raises(ValueError, match="nullable"):
        scoring.feature_source_columns(_artifact(dtype=dtype), ("id",))


@pytest.mark.parametrize("change", [{"version": 8}, {"table_id": "replacement"}])
def test_feature_snapshot_drift_rejected(change, monkeypatch):
    """Feature-only updates cannot bypass an unchanged base-source watermark."""
    from skyulf.integrations.databricks.feature_store import scoring

    snapshot = _binding()["lookup_evidence"]["feature_tables"][0] | change
    monkeypatch.setattr(scoring.snapshots, "snapshot_table", lambda *args: snapshot)
    with pytest.raises(ValueError, match="changed.*retrain"):
        scoring.validate_feature_snapshot(None, _artifact())


def test_feature_snapshot_receipt_is_detached_and_rechecked(monkeypatch):
    """The receipt carries the exact training lookup contract through commit guards."""
    from skyulf.integrations.databricks.feature_store import scoring

    current = Mock(return_value=_binding()["lookup_evidence"]["feature_tables"][0])
    monkeypatch.setattr(scoring.snapshots, "snapshot_table", current)
    receipt = scoring.validate_feature_snapshot(None, _artifact())
    assert receipt == {"feature_lookup": _binding()}
    scoring.validate_feature_receipt(None, receipt)
    assert current.call_count == 2


def test_empty_feature_binding_never_reads_tables():
    """An ordinary model must not acquire an optional SDK or table dependency."""
    from skyulf.integrations.databricks.feature_store import scoring

    artifact = _artifact()
    artifact.feature_lookup_json = None
    spark = Mock()
    assert scoring.validate_feature_snapshot(spark, artifact) == {}
    spark.sql.assert_not_called()


def test_single_feature_change_rejected_before_noop(recovery_case, monkeypatch):
    """An unchanged base table cannot hide feature updates from the publication gate."""
    from skyulf.integrations.databricks.scoring.incremental import local_incremental as batch

    store = recovery_case
    store.source_version = store.previous["source_end_version"]
    monkeypatch.setattr(
        batch,
        "validate_feature_snapshot",
        Mock(side_effect=ValueError("features changed")),
        raising=False,
    )
    with pytest.raises(ValueError, match="features changed"):
        batch.run_incremental_local_batch(
            store, store.prepared, record_key_columns=("id",), admission=store
        )
    assert store.writes == []


def test_model_set_feature_change_rejected_before_noop(tmp_path, monkeypatch):
    """The complete-set lifecycle checks feature freshness before its base-watermark shortcut."""
    from tests.integration.platforms.test_model_set_batch import _saved_set, _transport

    artifact, model, query = _saved_set(tmp_path)
    previous = {
        "skyulf_mode": "incremental_append",
        "artifact_kind": "model_set",
        "source_table_id": "source",
        "target_table_id": "target",
        "source_end_version": 0,
        "model_set_name": model.name,
        "model_set_version": model.version,
        "model_set_digest": model.digest,
        "set_history": {},
    }
    batch, commit = _transport(monkeypatch, query, previous)
    monkeypatch.setattr(
        batch,
        "validate_feature_snapshot",
        Mock(side_effect=ValueError("features changed")),
        raising=False,
    )
    with pytest.raises(ValueError, match="features changed"):
        batch._run_admitted_set(
            None,
            model,
            artifact,
            "source",
            "target",
            "source",
            "target",
            "incremental_append",
            20,
            100000,
        )
    commit.assert_not_called()


def test_native_call_passes_pinned_uri_named_schema_and_worker_params(monkeypatch):
    """The SDK must execute the inspected concrete release with the same worker contract."""
    from skyulf.integrations.databricks.feature_store import scoring

    client = Mock()
    sdk = SimpleNamespace(FeatureEngineeringClient=Mock(return_value=client))
    monkeypatch.setattr(scoring, "_sdk", lambda: sdk, raising=False)
    monkeypatch.setattr(scoring, "_validate_feature_stores", lambda *args: None, raising=False)
    schema, frame = object(), object()
    result = scoring.native_feature_score(
        frame,
        model_uri="models:/catalog.schema.model/3",
        result_type=schema,
        env_manager="virtualenv",
        prediction_batch_rows=7,
        tracking_uri="databricks",
        registry_uri="databricks-uc",
    )
    client.score_batch.assert_called_once_with(
        model_uri="models:/catalog.schema.model/3",
        df=frame,
        result_type=schema,
        env_manager="virtualenv",
        params={"skyulf_spark_output": True, "skyulf_spark_batch_rows": 7},
    )
    assert result is client.score_batch.return_value


def test_native_call_rejects_mismatching_active_registry(monkeypatch):
    """A second SDK download cannot silently select another workspace or registry."""
    mlflow = pytest.importorskip("mlflow")
    from skyulf.integrations.databricks.feature_store import scoring

    monkeypatch.setattr(mlflow, "get_tracking_uri", lambda: "databricks")
    monkeypatch.setattr(mlflow, "get_registry_uri", lambda: "databricks-uc:other")
    with pytest.raises(ValueError, match="active MLflow"):
        scoring._validate_feature_stores("databricks", "databricks-uc")


def test_feature_binding_transition_requires_explicit_rebuild():
    """A refreshed model cannot bless stale predictions at an unchanged source watermark."""
    from skyulf.integrations.databricks.feature_store import scoring

    with pytest.raises(ValueError, match="rebuild"):
        scoring.validate_feature_continuation({}, {"feature_lookup": _binding()})
    previous = {"feature_lookup": _binding()}
    scoring.validate_feature_continuation(previous, previous)
    assert previous["feature_lookup"] == _binding()


@pytest.mark.parametrize(
    "source_keys,output_keys,error",
    [
        ([1, 2], [2, 1], None),
        ([1, 2], [1], "cardinality"),
        ([1, 2], [1, 3], "record keys"),
    ],
)
def test_feature_join_preserves_exact_key_membership(monkeypatch, source_keys, output_keys, error):
    """Equal output counts alone cannot prove that SDK lookup preserved source membership."""
    from skyulf.integrations.databricks.scoring.batch import spark_scoring
    from skyulf.integrations.mlflow.spark import spark_model

    class Keys:
        """Evaluate anti-join membership with an in-memory transport stand-in."""

        def __init__(self, values):
            """Retain only identities; predictions never affect this check."""
            self.values = values

        def join(self, other, on, how):
            """Expose the distributed anti-join result without relying on Spark availability."""
            assert on == ["id"] and how == "left_anti"
            return Keys([value for value in self.values if value not in other.values])

        def limit(self, count):
            """Bound failure evidence to one mismatched key."""
            return Keys(self.values[:count])

        def count(self):
            """Count actual unmatched values."""
            return len(self.values)

    monkeypatch.setattr(
        spark_scoring,
        "read_distributed_rows",
        lambda frame, *args: spark_scoring.DistributedRows(frame, len(frame.values)),
    )
    if error:
        with pytest.raises(ValueError, match=error):
            spark_model._validate_feature_keys(Keys(source_keys), Keys(output_keys), ("id",))
    else:
        spark_model._validate_feature_keys(Keys(source_keys), Keys(output_keys), ("id",))
        assert sorted(source_keys) == sorted(output_keys)


@pytest.mark.parametrize(
    "columns,dtypes,error",
    [
        (
            ["id", "entity", "event_time", "direct", "amount"],
            [("event_time", "timestamp")],
            "override",
        ),
        (
            ["id", "entity", "event_time", "direct", "prediction"],
            [("event_time", "timestamp")],
            "reserved",
        ),
        (["id", "entity", "event_time", "direct"], [("event_time", "date")], "Timestamp"),
        (["id", "entity", "event_time"], [("event_time", "timestamp")], "direct"),
    ],
)
def test_feature_source_rejects_override_and_lookup_schema_drift(columns, dtypes, error):
    """Input projection must not hide a feature override or repair a changed time contract."""
    from skyulf.integrations.databricks.feature_store import scoring

    frame = SimpleNamespace(columns=columns, dtypes=dtypes, isStreaming=False)
    with pytest.raises(ValueError, match=error):
        scoring.validate_feature_source(frame, _artifact())


def test_feature_table_identity_stays_stable_while_reading_version(monkeypatch):
    """A replacement between Delta identity and history reads cannot inherit old evidence."""
    from skyulf.integrations.databricks.feature_store import scoring

    monkeypatch.setattr(
        scoring.snapshots, "table_identity", Mock(side_effect=["feature-id", "replaced"])
    )
    history = Mock()
    history.orderBy.return_value.select.return_value.first.return_value = {"version": 7}
    monkeypatch.setattr(scoring.snapshots, "history", lambda *args: history)
    with pytest.raises(ValueError, match="identity changed"):
        scoring._current_feature_table(None, "catalog.schema.features")


def test_feature_snapshot_rechecked_after_waiting_for_admission(recovery_case, monkeypatch):
    """A feature update while the target is locked must not escape a no-op decision."""
    from skyulf.integrations.databricks.scoring.incremental import local_incremental as batch

    store = recovery_case
    store.source_version = store.previous["source_end_version"]
    guard = Mock(side_effect=[{}, ValueError("features changed during admission")])
    monkeypatch.setattr(batch, "validate_feature_snapshot", guard)
    with pytest.raises(ValueError, match="during admission"):
        batch.run_incremental_local_batch(
            store, store.prepared, record_key_columns=("id",), admission=store
        )
    assert store.writes == [] and guard.call_count == 2


def test_single_feature_receipt_checked_before_recovery_write(recovery_case, monkeypatch):
    """A changed feature snapshot must abort the actual guarded Delta writer."""
    from tests.integration.platforms.test_local_cdf_recovery import recover, recovery_request

    from skyulf.integrations.databricks.scoring.incremental import local_incremental as batch

    store = recovery_case
    request = recovery_request(store)
    monkeypatch.setattr(
        batch, "validate_feature_receipt", Mock(side_effect=ValueError("feature drift"))
    )
    with pytest.raises(ValueError, match="feature drift"):
        recover(store, request)
    assert store.writes == [] and store.previous["source_end_version"] == 7


def test_set_feature_receipt_checked_before_write(monkeypatch):
    """Model-set publication cannot evaluate its writer after feature evidence changed."""
    from skyulf.integrations.databricks.model_sets import model_set_batch as batch

    monkeypatch.setattr(batch, "table_identity", lambda spark, table: table)
    monkeypatch.setattr(batch, "latest_source_version", lambda *args: {"version": 0})
    monkeypatch.setattr(
        batch, "validate_feature_receipt", Mock(side_effect=ValueError("feature drift"))
    )
    output = Mock()
    with pytest.raises(ValueError, match="feature drift"):
        batch._commit_set(
            None,
            output,
            "source",
            "target",
            {
                "source_table_id": "source",
                "target_table_id": "target",
                "expected_target_version": 0,
            },
            "append",
        )
    output.write.format.assert_not_called()


def test_shared_snapshot_collection_reads_each_table_once(monkeypatch):
    """Training and scoring must bind the same unique table set in deterministic order."""
    from skyulf.integrations.databricks.feature_store import snapshots
    from skyulf.integrations.databricks.feature_store.lifecycle_config import (
        deserialize_feature_spec,
    )

    record = _binding()["lookup_evidence"]["feature_tables"][0]
    lookup = deserialize_feature_spec(_binding()["lookup_spec"])
    read = Mock(return_value=record)
    monkeypatch.setattr(snapshots, "snapshot_table", read)
    actual = snapshots.snapshot_tables(None, lookup)
    read.assert_called_once_with(None, record["table_name"])
    snapshots.validate_snapshots(None, actual)
    assert actual == [record] and read.call_count == 2
