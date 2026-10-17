"""Source preflight checks real table contracts without issuing mutations."""

import pytest


def _query(schema_changes=None, *, cdf=True, duplicate=False, availability_type="timestamp"):
    """Simulate bounded warehouse metadata, keeping the generated SQL observable."""
    statements = []
    columns = {"id": "bigint", "x": "double", "y": "double", "known_at": availability_type}

    def execute(sql):
        """Reject any unexpected query instead of silently granting source validity."""
        statements.append(sql)
        if sql.startswith("DESCRIBE DETAIL"):
            return [
                {"format": "delta", "properties": {"delta.enableChangeDataFeed": str(cdf).lower()}}
            ]
        if sql.startswith("DESCRIBE TABLE"):
            values = columns | (schema_changes or {}) if "score" in sql else columns
            return [{"col_name": name, "data_type": kind} for name, kind in values.items()]
        if sql.startswith("SELECT"):
            return [{"problem": 1}] if duplicate and "HAVING" in sql else []
        raise AssertionError(f"Unexpected SQL: {sql}")

    return execute, statements


def _config():
    """Use separate labeled training and score sources with one numeric feature."""
    return {
        "training_table": "cat.data.train",
        "score_source_table": "cat.data.score",
        "record_key_columns": ["id"],
        "input_columns": ["x"],
        "target_column": "y",
    }


def test_source_validation_is_read_only_and_checks_shared_types():
    """A successful report must cover both input sources and explicitly bound outcomes."""
    from skyulf.integrations.databricks.projects.source_validation import validate_project_sources

    query, statements = _query()
    report = validate_project_sources(
        _config(), query, label_table="cat.data.labels", label_available_at_column="known_at"
    )
    assert report["status"] == "passed"
    assert set(report["tables"]) == {"cat.data.train", "cat.data.score", "cat.data.labels"}
    assert all(sql.startswith(("SELECT", "DESCRIBE")) for sql in statements)


def test_from_now_source_check_does_not_claim_training_rows_are_scored():
    """A source check must describe the chosen first-run policy accurately."""
    from skyulf.integrations.databricks.projects.source_validation import validate_project_sources

    query, _ = _query()
    config = _config() | {"score_source_table": "cat.data.train", "initial_scoring": "from_now"}
    report = validate_project_sources(config, query)
    assert not any("training history" in warning for warning in report["warnings"])


def test_monitoring_text_availability_has_runtime_warning():
    """A supported ISO text column must pass schema checks without claiming its values valid."""
    from skyulf.integrations.databricks.projects.source_validation import validate_project_sources

    query, _ = _query(availability_type="string")
    report = validate_project_sources(
        _config(), query, label_table="cat.data.labels", label_available_at_column="known_at"
    )
    assert report["status"] == "passed"
    assert any(
        "offset" in warning and "cat.data.labels.known_at" in warning
        for warning in report["warnings"]
    )


@pytest.mark.parametrize("kind", ["timestamp_ntz", "date", "bigint"])
def test_monitoring_rejects_unsupported_availability_types(kind):
    """Training parsing rules must not silently admit unsupported monitoring timestamps."""
    from skyulf.integrations.databricks.projects.source_validation import validate_project_sources

    query, _ = _query(availability_type=kind)
    with pytest.raises(ValueError, match="availability"):
        validate_project_sources(
            _config(), query, label_table="cat.data.labels", label_available_at_column="known_at"
        )


@pytest.mark.parametrize(
    "kind,parsing,accepted",
    [
        ("timestamp", {}, True),
        ("timestamp", {"timezone": "UTC"}, False),
        ("string", {"format": "%Y-%m-%dT%H:%M:%S%z"}, True),
        ("string", {"format": "%Y-%m-%d %H:%M:%S", "timezone": "UTC"}, True),
        ("string", {}, False),
        ("string", {"format": "%Y-%m-%d %H:%M:%S"}, False),
        ("timestamp_ntz", {"timezone": "UTC"}, True),
        ("timestamp_ntz", {}, False),
        ("date", {"timezone": "UTC", "date_only": "midnight"}, True),
        ("date", {}, False),
        ("bigint", {}, False),
    ],
)
def test_training_availability_matches_declared_parsing(kind, parsing, accepted):
    """Preflight must agree with native training date type and parsing admission."""
    from skyulf.integrations.databricks.projects.source_validation import validate_project_sources

    config = _config() | {
        "training_label_table": "cat.data.labels",
        "result_available_at_column": "known_at",
        "result_time_parsing": parsing,
    }
    query, _ = _query(availability_type=kind)
    if not accepted:
        with pytest.raises(ValueError):
            validate_project_sources(config, query)
        return
    report = validate_project_sources(config, query)
    assert report["status"] == "passed"
    if kind != "timestamp":
        assert any("result_time_parsing" in warning for warning in report["warnings"])


def test_shared_outcomes_keep_monitoring_type_restrictions():
    """A training timezone does not make a shared NTZ outcome usable by monitoring."""
    from skyulf.integrations.databricks.projects.source_validation import validate_project_sources

    config = _config() | {
        "training_label_table": "cat.data.labels",
        "result_available_at_column": "known_at",
        "result_time_parsing": {"timezone": "UTC"},
    }
    query, _ = _query(availability_type="timestamp_ntz")
    with pytest.raises(ValueError, match="Monitoring.*availability"):
        validate_project_sources(
            config, query, label_table="cat.data.labels", label_available_at_column="known_at"
        )


@pytest.mark.parametrize(
    "options,error",
    [
        ({"cdf": False}, "change data feed"),
        ({"duplicate": True}, "duplicate"),
        ({"schema_changes": {"x": "string"}}, "type"),
    ],
)
def test_source_validation_rejects_bad_contracts(options, error):
    """Readable tables are insufficient when scoring would duplicate or reinterpret records."""
    from skyulf.integrations.databricks.projects.source_validation import validate_project_sources

    query, _ = _query(**options)
    with pytest.raises(ValueError, match=error):
        validate_project_sources(_config(), query)


def test_source_validation_rejects_unsafe_identifiers_before_query():
    """A configuration name must never become an additional SQL statement."""
    from skyulf.integrations.databricks.projects.source_validation import validate_project_sources

    query, statements = _query()
    with pytest.raises(ValueError):
        validate_project_sources(_config() | {"training_table": "x; DROP TABLE example"}, query)
    assert statements == []


@pytest.mark.parametrize(
    "extra",
    [
        {"event_column": "event_at"},
        {"weight_column": "weight"},
        {
            "pre_split_steps": [
                {
                    "name": "valid",
                    "transformer": "DropMissingRows",
                    "params": {"subset": ["sensor"]},
                }
            ]
        },
    ],
)
def test_external_outcomes_do_not_hide_required_training_columns(extra):
    """Joining targets must not skip checks on dates, weights or pre-split filter inputs."""
    from skyulf.integrations.databricks.projects.source_validation import validate_project_sources

    query, _ = _query()
    config = (
        _config()
        | {"training_label_table": "cat.data.labels", "result_available_at_column": "known_at"}
        | extra
    )
    with pytest.raises(ValueError, match="missing columns"):
        validate_project_sources(config, query)


def test_lookup_outputs_are_not_required_in_raw_sources():
    """Source preflight must distinguish fetched features from columns supplied by the caller."""
    from skyulf.integrations.databricks.projects.source_validation import validate_project_sources

    query, _ = _query()
    config = _config() | {
        "input_columns": ["fetched"],
        "feature_lookup": {
            "lookups": [
                {
                    "table_name": "cat.features.values",
                    "lookup_key": ["id"],
                    "feature_names": ["fetched"],
                }
            ]
        },
    }
    report = validate_project_sources(config, query)
    assert report["status"] == "passed"
    assert any("Feature lookup" in warning for warning in report["warnings"])


@pytest.mark.parametrize("change", [{"truncated": True}, {"next_chunk_index": 1}])
def test_warehouse_metadata_rejects_partial_results(change):
    """A truncated metadata response cannot prove a table satisfies the source contract."""
    from types import SimpleNamespace as NS
    from unittest.mock import Mock

    from databricks.sdk.service.sql import StatementState

    from skyulf.integrations.databricks.projects.source_validation import query_source_sql

    result = NS(
        status=NS(state=StatementState.SUCCEEDED),
        manifest=NS(
            truncated=change.get("truncated", False), schema=NS(columns=[NS(name="format")])
        ),
        result=NS(next_chunk_index=change.get("next_chunk_index"), data_array=[["delta"]]),
    )
    client = Mock()
    client.statement_execution.execute_statement.return_value = result
    with pytest.raises(ValueError, match="bounded"):
        query_source_sql(client, "warehouse", "DESCRIBE DETAIL cat.data.source")


def test_delta_text_availability_preflight_and_runtime(delta_spark):
    """Real Delta text outcomes must pass preflight and preserve runtime UTC validation."""
    import os
    from datetime import UTC, datetime
    from uuid import uuid4

    from skyulf.integrations.databricks.data.training.training_dates import (
        TrainingDateSpec,
        instant_microseconds,
        normalize_training_dates,
        training_date_spec,
    )
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_metrics import (
        _eligible,
    )
    from skyulf.integrations.databricks.projects.source_validation import (
        query_source_sql,
        validate_project_sources,
    )
    from skyulf.integrations.databricks.shared._contracts import table_name

    namespace = os.environ.get("SKYULF_TEST_SCHEMA", "spark_catalog.default")
    source, labels = [f"{namespace}.availability_test_{uuid4().hex}" for _ in range(2)]
    source_sql, labels_sql = table_name(source), table_name(labels)
    parsing = {"format": "%Y-%m-%dT%H:%M:%S%z"}

    def query(sql):
        """Execute actual read-only SQL, including the warehouse transport when provided."""
        assert sql.startswith(("SELECT", "DESCRIBE"))
        if warehouse := os.environ.get("SKYULF_TEST_WAREHOUSE"):
            from databricks.sdk import WorkspaceClient

            return query_source_sql(WorkspaceClient(), warehouse, sql)
        return [row.asDict(recursive=True) for row in delta_spark.sql(sql).collect()]

    try:
        delta_spark.sql(
            f"CREATE TABLE {source_sql} USING DELTA "
            "TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true') AS "
            "SELECT CAST(1 AS BIGINT) AS id, CAST(2 AS DOUBLE) AS x"
        )
        delta_spark.sql(
            f"CREATE TABLE {labels_sql} USING DELTA AS SELECT CAST(1 AS BIGINT) AS id, "
            "CAST(3 AS DOUBLE) AS y, '2026-01-01T02:00:00+02:00' AS known_at"
        )
        config = _config() | {
            "training_table": source,
            "score_source_table": source,
            "training_label_table": labels,
            "result_available_at_column": "known_at",
            "result_time_parsing": parsing,
        }
        report = validate_project_sources(
            config, query, label_table=labels, label_available_at_column="known_at"
        )
        assert report["status"] == "passed"
        assert any("offset" in warning for warning in report["warnings"])
        outcomes = delta_spark.table(labels)
        normalized = normalize_training_dates(
            outcomes,
            event_column=None,
            result_column="known_at",
            event_spec=TrainingDateSpec(),
            result_spec=training_date_spec(parsing),
        )
        assert normalized.first()["known_at"] == instant_microseconds(
            datetime(2026, 1, 1, tzinfo=UTC)
        )
        assert (
            _eligible(outcomes, "known_at", datetime(2026, 1, 2, tzinfo=UTC), ("id",), "y").count()
            == 1
        )
        invalid = delta_spark.sql(
            "SELECT CAST(1 AS BIGINT) AS id, CAST(3 AS DOUBLE) AS y, "
            "'2026-01-01T00:00:00' AS known_at"
        )
        with pytest.raises(ValueError, match="naive label availability"):
            _eligible(invalid, "known_at", datetime(2026, 1, 2, tzinfo=UTC), ("id",), "y")
        with pytest.raises(ValueError, match="Invalid training dates"):
            normalize_training_dates(
                invalid,
                event_column=None,
                result_column="known_at",
                event_spec=TrainingDateSpec(),
                result_spec=training_date_spec(parsing),
            )
    finally:
        for name in (labels_sql, source_sql):
            delta_spark.sql(f"DROP TABLE IF EXISTS {name}")
    assert not delta_spark.catalog.tableExists(source)
    assert not delta_spark.catalog.tableExists(labels)
