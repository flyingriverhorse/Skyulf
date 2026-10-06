"""Serving payload views retain canonical semantics and immutable backing snapshots."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from skyulf.integrations.databricks.observability.monitoring.serving import serving_sources

SOURCE = "workspace.observability.model_payload"
BACKING = "workspace.observability.model_otel_logs"
NOW = datetime(2026, 10, 6, tzinfo=UTC)
DDL = """CREATE VIEW observability.model_payload (
  databricks_request_id, request_date, client_request_id, request_time,
  status_code, sampling_fraction, execution_duration_ms, request, response,
  served_entity_id, logging_error_codes, requester)
TBLPROPERTIES ('otel.sourceLogsTable' = 'workspace.observability.model_otel_logs')
WITH SCHEMA BINDING
AS SELECT
  attributes:databricks_request_id::STRING AS databricks_request_id,
  attributes:request_date::DATE AS request_date,
  attributes:client_request_id::STRING AS client_request_id,
  timestamp_millis(attributes:request_time::BIGINT) AS request_time,
  attributes:status_code::INT AS status_code,
  attributes:sampling_fraction::DOUBLE AS sampling_fraction,
  attributes:execution_duration_ms::BIGINT AS execution_duration_ms,
  body:request::STRING AS request,
  body:response::STRING AS response,
  attributes:served_entity_id::STRING AS served_entity_id,
  body:logging_error_codes::ARRAY<STRING> AS logging_error_codes,
  attributes:requester::STRING AS requester
FROM `workspace`.`observability`.`model_otel_logs`
WHERE resource.attributes:["databricks.product"]::STRING = 'custom-model-serving'
  AND resource.attributes:["telemetry.sdk.name"]::STRING = 'payload-logger'
"""


@pytest.fixture
def source(monkeypatch):
    """Expose read-only metadata and pinned DataFrame operations independently."""
    metadata = {"kind": "VIEW", "backing": BACKING, "ddl": DDL}

    def query(sql):
        """Admit only the two expected metadata commands, never remote view SQL."""
        if (
            sql
            == "SHOW TBLPROPERTIES `workspace`.`observability`.`model_payload` ('otel.sourceLogsTable')"
        ):
            return SimpleNamespace(first=lambda: {"value": metadata["backing"]})
        if sql == "SHOW CREATE TABLE `workspace`.`observability`.`model_payload`":
            return SimpleNamespace(first=lambda: {"createtab_stmt": metadata["ddl"]})
        raise AssertionError(f"Unexpected SQL: {sql}")

    spark = SimpleNamespace(
        catalog=SimpleNamespace(
            getTable=Mock(side_effect=lambda name: SimpleNamespace(tableType=metadata["kind"]))
        ),
        sql=Mock(side_effect=query),
    )
    projected = object()
    frame = Mock()
    frame.filter.return_value = frame
    frame.selectExpr.return_value = projected
    identity, snapshot = Mock(return_value="physical-id"), Mock(return_value=7)
    read = Mock(return_value=frame)
    monkeypatch.setattr(serving_sources, "table_identity", identity)
    monkeypatch.setattr(serving_sources, "snapshot_at", snapshot)
    monkeypatch.setattr(serving_sources, "read_snapshot", read)
    return SimpleNamespace(
        metadata=metadata,
        spark=spark,
        frame=frame,
        projected=projected,
        identity=identity,
        snapshot=snapshot,
        read=read,
    )


def test_managed_delta_keeps_legacy_snapshot_route(source):
    """Direct Delta payloads retain their physical identity and cutoff semantics."""
    source.metadata["kind"] = "MANAGED"
    frame, evidence = serving_sources.read_serving_payload_snapshot(source.spark, SOURCE, NOW)
    assert frame is source.frame
    source.snapshot.assert_called_once_with(source.spark, SOURCE, NOW)
    source.read.assert_called_once_with(source.spark, SOURCE, 7)
    source.spark.sql.assert_not_called()
    assert evidence == {
        "source_table": SOURCE,
        "source_table_id": "physical-id",
        "prediction_table": SOURCE,
        "prediction_table_id": "physical-id",
        "prediction_version": 7,
    }
    serving_sources.verify_serving_payload_snapshot(source.spark, SOURCE, evidence)
    assert source.identity.call_args_list == [
        call(source.spark, SOURCE),
        call(source.spark, SOURCE),
    ]


def test_canonical_view_projects_pinned_backing_without_executing_view(source):
    """View population must come from its attested projection over the selected Delta version."""
    frame, evidence = serving_sources.read_serving_payload_snapshot(source.spark, SOURCE, NOW)
    assert frame is source.projected
    source.snapshot.assert_called_once_with(source.spark, BACKING, NOW)
    source.read.assert_called_once_with(source.spark, BACKING, 7)
    assert source.frame.filter.call_args.args == (
        "resource.attributes:[\"databricks.product\"]::STRING = 'custom-model-serving' "
        "AND resource.attributes:[\"telemetry.sdk.name\"]::STRING = 'payload-logger'",
    )
    assert (
        "timestamp_millis(attributes:request_time::BIGINT) AS request_time"
        in source.frame.selectExpr.call_args.args
    )
    assert len(source.frame.selectExpr.call_args.args) == 12
    assert evidence["payload_backing_table"] == BACKING
    assert evidence["source_table_id"] == evidence["prediction_table_id"] == "physical-id"
    assert evidence["prediction_version"] == 7
    assert len(evidence["payload_view_definition_sha256"]) == 64
    serving_sources.verify_serving_payload_snapshot(source.spark, SOURCE, evidence)
    assert source.identity.call_args_list == [
        call(source.spark, BACKING),
        call(source.spark, BACKING),
    ]


def test_safe_whitespace_and_identifier_backticks_are_equivalent(source):
    """Formatting differences must not alter canonical projection admission or verification."""
    _, evidence = serving_sources.read_serving_payload_snapshot(source.spark, SOURCE, NOW)
    source.metadata["ddl"] = DDL.replace("\n", "  ").replace("`workspace`", "workspace")
    serving_sources.verify_serving_payload_snapshot(source.spark, SOURCE, evidence)
    assert source.identity.call_count == 2


@pytest.mark.parametrize(
    "before,after",
    [
        ("body:response::STRING", "body:request::STRING"),
        ("'custom-model-serving'", "'other-model-serving'"),
        ("'payload-logger'", "'payload-  logger'"),
        ("`model_otel_logs`", "`foreign_logs`"),
        ("AS SELECT", "AS SELECT DISTINCT"),
        ("request_date, client_request_id", "client_request_id, request_date"),
    ],
)
def test_changed_view_semantics_fail_before_delta_read(source, before, after):
    """Foreign projections, filters and output aliases cannot silently change attribution."""
    source.metadata["ddl"] = DDL.replace(before, after)
    with pytest.raises(ValueError, match="canonical"):
        serving_sources.read_serving_payload_snapshot(source.spark, SOURCE, NOW)
    source.read.assert_not_called()


@pytest.mark.parametrize("backing", [None, "", "unqualified", "cat.schema.logs; DROP TABLE x"])
def test_missing_or_unsafe_backing_property_fails(source, backing):
    """Telemetry properties must resolve to one concrete safe UC Delta table."""
    source.metadata["backing"] = backing
    with pytest.raises(ValueError):
        serving_sources.read_serving_payload_snapshot(source.spark, SOURCE, NOW)
    source.read.assert_not_called()


def test_replaced_backing_table_fails_post_read_verification(source):
    """Same-name backing replacement cannot mix physical populations in one report."""
    _, evidence = serving_sources.read_serving_payload_snapshot(source.spark, SOURCE, NOW)
    source.identity.return_value = "replacement-id"
    with pytest.raises(ValueError, match="replaced"):
        serving_sources.verify_serving_payload_snapshot(source.spark, SOURCE, evidence)
    assert source.identity.call_count == 2


def test_changed_view_definition_fails_post_read_verification(source):
    """Metadata changes after planning cannot retain an earlier definition receipt."""
    _, evidence = serving_sources.read_serving_payload_snapshot(source.spark, SOURCE, NOW)
    source.metadata["ddl"] = DDL.replace("WITH SCHEMA BINDING", "WITH SCHEMA EVOLUTION")
    with pytest.raises(ValueError, match="definition"):
        serving_sources.verify_serving_payload_snapshot(source.spark, SOURCE, evidence)
    assert source.read.call_count == 1


def test_changed_object_kind_fails_post_read_verification(source):
    """A payload view replaced by a direct table cannot reuse a view observation receipt."""
    _, evidence = serving_sources.read_serving_payload_snapshot(source.spark, SOURCE, NOW)
    source.metadata["kind"] = "MANAGED"
    with pytest.raises(ValueError, match="definition"):
        serving_sources.verify_serving_payload_snapshot(source.spark, SOURCE, evidence)
    assert source.read.call_count == 1
