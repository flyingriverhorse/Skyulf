"""Publish bounded local predictions to a real period-scoped Delta target."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib import import_module
from importlib.metadata import version
from uuid import uuid4

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference._manifest import ColumnSpec
from skyulf.integrations.databricks._contracts import BatchSpec
from skyulf.integrations.databricks.admission import BatchConflictError, LocalTableLock
from skyulf.integrations.databricks.local_batch import LocalSourceSpec, fit_local_workflow
from skyulf.integrations.databricks.local_sdk import (
    InputSource,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    PreflightResult,
    PreparedLocalWorkflow,
)


@pytest.fixture
def local_delta_case(delta_spark, tmp_path):
    """Provision isolated source, target and a fitted local model for two periods."""
    spark = delta_spark
    suffix = uuid4().hex
    source = f"spark_catalog.default.local_source_{suffix}"
    target = f"spark_catalog.default.local_predictions_{suffix}"
    spark.createDataFrame(
        [(1, datetime(2026, 1, 5, tzinfo=UTC), 2.0), (2, datetime(2026, 1, 7, tzinfo=UTC), 4.0)],
        "id long, event_time timestamp, x double",
    ).write.format("delta").saveAsTable(source)
    spark.createDataFrame(
        [(9, datetime(2025, 12, 1, tzinfo=UTC), 99.0, "prior", "risk", "1")],
        "id long, event_time timestamp, prediction double, run_id string, "
        "model_name string, model_version string",
    ).write.format("delta").saveAsTable(target)
    values = np.arange(20, dtype="float64")
    data = pd.DataFrame({"x": values, "target": 2.0 * values})
    artifact = fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        SplitDataset(train=data.iloc[:16], test=data.iloc[16:]),
        target_column="target",
        artifact_path=tmp_path / "fitted",
        max_rows=20,
        max_bytes=10_000,
    )
    config = LocalWorkflowConfig(
        runtime="databricks",
        engine="pandas",
        source=InputSource(kind="uc_table", table=source, version=0, max_rows=10, max_bytes=10_000),
        model=ModelSelection(kind="local_pipeline", name="workspace.test.local_model", version="1"),
        sink=OutputSink(kind="uc_delta", table=target),
    )
    preflight = PreflightResult(
        issues=(),
        remote_checked=True,
        model_version="1",
        model_digest=artifact.manifest.pipeline_sha256,
        output_schema=(ColumnSpec(name="prediction", dtype="float64"),),
    )
    prepared = PreparedLocalWorkflow(config, artifact, preflight)
    january = LocalSourceSpec(
        table=source,
        version=0,
        period_start=datetime(2026, 1, 1, tzinfo=UTC),
        period_end=datetime(2026, 2, 1, tzinfo=UTC),
        row_keys=("id",),
        input_columns=("x",),
        max_rows=10,
        max_bytes=10_000,
    )
    request = BatchSpec(
        period_start=january.period_start,
        period_end=january.period_end,
        as_of=datetime.now(UTC) + timedelta(minutes=10),
        row_keys=("id",),
        output_table=target,
        model_name="workspace.test.local_model",
        model_version="1",
        source_version=0,
        code_version=version("skyulf-core"),
        run_id="january-local",
        model_digest=artifact.manifest.pipeline_sha256,
        expected_target_version=0,
        mode="local_pipeline",
    )
    try:
        yield spark, source, target, prepared, january, request, LocalTableLock(tmp_path / "locks")
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {source}")
        spark.sql(f"DROP TABLE IF EXISTS {target}")


def test_local_predictions_publish_two_months_and_replay(local_delta_case):
    """A second month adds only its rows and replay cannot rewrite either month."""
    from skyulf.integrations.databricks import run_local_batch

    spark, source, target, prepared, january, request, admission = local_delta_case
    first = run_local_batch(spark, january, prepared, request, admission=admission)
    spark.createDataFrame(
        [(3, datetime(2026, 2, 5, tzinfo=UTC), 6.0), (4, datetime(2026, 2, 7, tzinfo=UTC), 8.0)],
        "id long, event_time timestamp, x double",
    ).write.format("delta").mode("append").saveAsTable(source)
    february = replace(
        january,
        version=1,
        period_start=datetime(2026, 2, 1, tzinfo=UTC),
        period_end=datetime(2026, 3, 1, tzinfo=UTC),
    )
    config = prepared.config.model_copy(
        update={
            "source": InputSource(
                kind="uc_table", table=source, version=1, max_rows=10, max_bytes=10_000
            )
        }
    )
    prepared_february = replace(prepared, config=config)
    next_request = replace(
        request,
        period_start=february.period_start,
        period_end=february.period_end,
        source_version=1,
        run_id="february-local",
        expected_target_version=first.commit_version,
    )
    second = run_local_batch(spark, february, prepared_february, next_request, admission=admission)
    replay = run_local_batch(spark, february, prepared_february, next_request, admission=admission)
    rows = spark.table(target).orderBy("id").collect()
    assert [(row.id, round(row.prediction), row["run_id"]) for row in rows] == [
        (1, 4, "january-local"),
        (2, 8, "january-local"),
        (3, 12, "february-local"),
        (4, 16, "february-local"),
        (9, 99, "prior"),
    ]
    assert first.input_count == second.input_count == replay.input_count == 2
    assert replay.replayed and replay.commit_version == second.commit_version
    with pytest.raises(BatchConflictError):
        run_local_batch(
            spark,
            february,
            prepared_february,
            replace(
                next_request, run_id="stale-local", expected_target_version=first.commit_version
            ),
            admission=admission,
        )
    assert spark.table(target).count() == 5


def test_local_publisher_is_available() -> None:
    """Local publication needs its own entry point separate from Spark scoring."""
    from skyulf.integrations.databricks import run_local_batch

    assert callable(run_local_batch)


def test_empty_local_month_rejects_without_changing_target(local_delta_case):
    """An accidental empty monthly source must not erase an existing Delta period."""
    from skyulf.integrations.databricks import run_local_batch

    spark, _, target, prepared, january, request, admission = local_delta_case
    empty = replace(
        january,
        period_start=datetime(2026, 3, 1, tzinfo=UTC),
        period_end=datetime(2026, 4, 1, tzinfo=UTC),
    )
    empty_request = replace(
        request,
        period_start=empty.period_start,
        period_end=empty.period_end,
        run_id="empty-local",
    )
    with pytest.raises(ValueError, match="Empty period replacement requires allow_empty=True"):
        run_local_batch(spark, empty, prepared, empty_request, admission=admission)
    assert spark.table(target).count() == 1
    allowed = run_local_batch(
        spark, empty, prepared, replace(empty_request, allow_empty=True), admission=admission
    )
    assert allowed.output_count == 0
    assert spark.table(target).count() == 1


def test_local_publisher_denied_admission_preserves_target(local_delta_case):
    """An admission denial must propagate without changing the prediction table."""
    from skyulf.integrations.databricks import run_local_batch

    class DeniedAdmission:
        """Simulate a distributed authority that denies this writer."""

        local_only = False

        @contextmanager
        def hold(self, table_id):
            """Raise before granting publication authority."""
            raise PermissionError("publish access denied")
            yield  # pragma: no cover - context manager contract

    spark, _, target, prepared, january, request, _ = local_delta_case
    before_version = spark.sql(f"DESCRIBE HISTORY {target}").first().version
    with pytest.raises(PermissionError, match="access denied"):
        run_local_batch(spark, january, prepared, request, admission=DeniedAdmission())
    assert spark.sql(f"DESCRIBE HISTORY {target}").first().version == before_version
    assert [row.id for row in spark.table(target).collect()] == [9]


def test_local_publisher_rejects_target_schema_before_commit(local_delta_case):
    """A target/output dtype mismatch must not stage a Delta period replacement."""
    from skyulf.integrations.databricks import run_local_batch

    spark, _, target, prepared, january, request, admission = local_delta_case
    invalid = replace(
        prepared,
        preflight=replace(
            prepared.preflight,
            output_schema=(ColumnSpec(name="prediction", dtype="string"),),
        ),
    )
    before_version = spark.sql(f"DESCRIBE HISTORY {target}").first().version
    with pytest.raises(ValueError, match="Target output type"):
        run_local_batch(spark, january, invalid, request, admission=admission)
    assert spark.sql(f"DESCRIBE HISTORY {target}").first().version == before_version
    assert [row.id for row in spark.table(target).collect()] == [9]


def test_local_publisher_rejects_stale_model_and_source_identity(local_delta_case):
    """Changed model or source pins must fail before touching the Delta target."""
    from skyulf.integrations.databricks import run_local_batch

    spark, _, target, prepared, january, request, admission = local_delta_case
    before_version = spark.sql(f"DESCRIBE HISTORY {target}").first().version
    for field, value, message in (
        ("model_version", "2", "model version"),
        ("model_digest", "0" * 64, "model digest"),
        ("source_version", 1, "source contract"),
    ):
        with pytest.raises(ValueError, match=message):
            run_local_batch(
                spark,
                january,
                prepared,
                replace(request, **{field: value}),
                admission=admission,
            )
    assert spark.sql(f"DESCRIBE HISTORY {target}").first().version == before_version
    assert [row.id for row in spark.table(target).collect()] == [9]


def test_incremental_local_batch_discovers_appends_without_period_inputs(local_delta_case):
    """A second run scores only a late-arriving insert and preserves old predictions."""
    from skyulf.integrations.databricks import run_incremental_local_batch

    spark, source, target, prepared, _, _, admission = local_delta_case
    spark.sql(f"DELETE FROM {target} WHERE id = 9")
    spark.sql(f"ALTER TABLE {source} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    config = prepared.config.model_copy(
        update={
            "source": InputSource(
                kind="uc_table",
                table=source,
                read_mode="incremental",
                max_rows=10,
                max_bytes=10_000,
            )
        }
    )
    prepared = replace(prepared, config=config)
    first = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    spark.createDataFrame(
        [(3, datetime(2026, 1, 6, tzinfo=UTC), 6.0)],
        "id long, event_time timestamp, x double",
    ).write.format("delta").mode("append").saveAsTable(source)
    second = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    replay = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    rows = spark.table(target).orderBy("id").collect()
    assert [(row.id, round(row.prediction)) for row in rows] == [(1, 4), (2, 8), (3, 12)]
    assert first.input_count == 2 and second.input_count == 1
    assert replay.input_count == 0 and replay.noop
    assert rows[0]["run_id"] == rows[1]["run_id"]
    assert rows[2]["run_id"] != rows[1]["run_id"]


def test_incremental_single_writer_uses_receipts_without_control_table(local_delta_case):
    """A sole writer can replay Delta receipts without provisioning lock state."""
    from skyulf.integrations.databricks import run_incremental_local_batch
    from skyulf.integrations.databricks.admission import SingleWriterAdmission

    spark, source, target, prepared, _ = _prepared_incremental_case(local_delta_case)
    admission = SingleWriterAdmission()
    first = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    spark.createDataFrame(
        [(3, datetime(2026, 1, 6, tzinfo=UTC), 6.0)],
        "id long, event_time timestamp, x double",
    ).write.format("delta").mode("append").saveAsTable(source)
    second = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    replay = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    assert (first.input_count, second.input_count, replay.input_count) == (2, 1, 0)
    assert replay.noop and replay.commit_version == second.commit_version
    assert [row.id for row in spark.table(target).orderBy("id").collect()] == [1, 2, 3]


def test_incremental_local_batch_does_not_require_a_date_column(local_delta_case):
    """New rows are found from Delta commits even when no event date exists."""
    from skyulf.integrations.databricks import run_incremental_local_batch

    spark, _, _, prepared, _, _, admission = local_delta_case
    suffix = uuid4().hex
    source = f"spark_catalog.default.undated_source_{suffix}"
    target = f"spark_catalog.default.undated_target_{suffix}"
    spark.createDataFrame([(1, 2.0), (2, 4.0)], "id long, x double").write.format("delta").option(
        "delta.enableChangeDataFeed", "true"
    ).saveAsTable(source)
    spark.createDataFrame(
        [],
        "id long, prediction double, run_id string, model_name string, model_version string",
    ).write.format("delta").saveAsTable(target)
    config = prepared.config.model_copy(
        update={
            "source": InputSource(
                kind="uc_table",
                table=source,
                read_mode="incremental",
                max_rows=10,
                max_bytes=10_000,
            ),
            "sink": OutputSink(kind="uc_delta", table=target),
        }
    )
    prepared = replace(prepared, config=config)
    try:
        first = run_incremental_local_batch(spark, prepared, row_keys=("id",), admission=admission)
        spark.createDataFrame([(3, 6.0)], "id long, x double").write.format("delta").mode(
            "append"
        ).saveAsTable(source)
        second = run_incremental_local_batch(spark, prepared, row_keys=("id",), admission=admission)
        assert (first.input_count, second.input_count) == (2, 1)
        assert [
            (row.id, round(row.prediction)) for row in spark.table(target).orderBy("id").collect()
        ] == [(1, 4), (2, 8), (3, 12)]
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {source}")
        spark.sql(f"DROP TABLE IF EXISTS {target}")


def _prepared_incremental_case(local_delta_case):
    """Create an empty target and enable source change tracking for safety checks."""
    spark, source, target, prepared, _, _, admission = local_delta_case
    spark.sql(f"DELETE FROM {target} WHERE id = 9")
    spark.sql(f"ALTER TABLE {source} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    config = prepared.config.model_copy(
        update={
            "source": InputSource(
                kind="uc_table",
                table=source,
                read_mode="incremental",
                max_rows=10,
                max_bytes=10_000,
            )
        }
    )
    return spark, source, target, replace(prepared, config=config), admission


def test_incremental_local_batch_rejects_source_updates(local_delta_case):
    """Changed source rows cannot silently keep stale predictions or advance the watermark."""
    from skyulf.integrations.databricks import run_incremental_local_batch

    spark, source, target, prepared, admission = _prepared_incremental_case(local_delta_case)
    first = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    original = spark.table(target).orderBy("id").collect()
    spark.sql(f"UPDATE {source} SET x = 100.0 WHERE id = 1")
    with pytest.raises(ValueError, match="updates and deletes"):
        run_incremental_local_batch(
            spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
        )
    assert spark.table(target).orderBy("id").collect() == original
    assert spark.sql(f"DESCRIBE HISTORY {target}").first().version == first.commit_version


def test_incremental_local_batch_rejects_existing_prediction_key(local_delta_case):
    """A repeated source key must fail without appending a second prediction."""
    from skyulf.integrations.databricks import run_incremental_local_batch

    spark, source, target, prepared, admission = _prepared_incremental_case(local_delta_case)
    first = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    spark.createDataFrame(
        [(1, datetime(2026, 2, 8, tzinfo=UTC), 40.0)],
        "id long, event_time timestamp, x double",
    ).write.format("delta").mode("append").saveAsTable(source)
    with pytest.raises(BatchConflictError, match="already has"):
        run_incremental_local_batch(
            spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
        )
    assert spark.table(target).count() == 2
    assert spark.sql(f"DESCRIBE HISTORY {target}").first().version == first.commit_version


def test_incremental_local_batch_requires_change_feed_before_bootstrap(local_delta_case):
    """A source without CDF must fail before writing a target that cannot resume safely."""
    from skyulf.integrations.databricks import run_incremental_local_batch

    spark, source, target, prepared, _, _, admission = local_delta_case
    spark.sql(f"DELETE FROM {target} WHERE id = 9")
    config = prepared.config.model_copy(
        update={
            "source": InputSource(
                kind="uc_table",
                table=source,
                read_mode="incremental",
                max_rows=10,
                max_bytes=10_000,
            )
        }
    )
    prepared = replace(prepared, config=config)
    before = spark.sql(f"DESCRIBE HISTORY {target}").first().version
    with pytest.raises(ValueError, match="Change Data Feed"):
        run_incremental_local_batch(
            spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
        )
    assert spark.sql(f"DESCRIBE HISTORY {target}").first().version == before
    assert spark.table(target).count() == 0


def test_incremental_local_batch_rejects_target_reset_without_watermark(local_delta_case):
    """An empty external rewrite must not erase the committed source watermark."""
    from skyulf.integrations.databricks import run_incremental_local_batch

    spark, _, target, prepared, admission = _prepared_incremental_case(local_delta_case)
    run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    spark.sql(f"DELETE FROM {target} WHERE id IN (1, 2)")
    before = spark.sql(f"DESCRIBE HISTORY {target}").first().version
    with pytest.raises(BatchConflictError, match="outside|receipt"):
        run_incremental_local_batch(
            spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
        )
    assert spark.table(target).count() == 0
    assert spark.sql(f"DESCRIBE HISTORY {target}").first().version == before


def test_incremental_local_batch_scores_polars_artifact(local_delta_case, tmp_path):
    """Automatic source discovery must preserve the fitted Polars engine."""
    from skyulf.integrations.databricks import run_incremental_local_batch

    spark, source, target, prepared, admission = _prepared_incremental_case(local_delta_case)
    values = np.arange(20, dtype="float64")
    training = pl.DataFrame({"x": values, "target": 2.0 * values})
    artifact = fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        SplitDataset(train=training[:16], test=training[16:]),
        target_column="target",
        artifact_path=tmp_path / "polars_fitted",
        max_rows=20,
        max_bytes=10_000,
    )
    prepared = replace(
        prepared,
        config=prepared.config.model_copy(update={"engine": "polars"}),
        artifact=artifact,
        preflight=replace(
            prepared.preflight,
            model_digest=artifact.manifest.pipeline_sha256,
        ),
    )
    first = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    spark.createDataFrame(
        [(3, datetime(2026, 1, 6, tzinfo=UTC), 6.0)],
        "id long, event_time timestamp, x double",
    ).write.format("delta").mode("append").saveAsTable(source)
    second = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    assert (first.input_count, second.input_count) == (2, 1)
    assert [
        (row.id, round(row.prediction)) for row in spark.table(target).orderBy("id").collect()
    ] == [(1, 4), (2, 8), (3, 12)]


def test_incremental_local_batch_failed_write_keeps_watermark(local_delta_case, monkeypatch):
    """A failed target append must leave new source rows available for a clean retry."""
    DataFrameWriter = import_module("pyspark.sql.readwriter").DataFrameWriter

    from skyulf.integrations.databricks import run_incremental_local_batch
    from skyulf.integrations.databricks.delta import DeltaPublishError

    spark, source, target, prepared, admission = _prepared_incremental_case(local_delta_case)
    first = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    spark.createDataFrame(
        [(3, datetime(2026, 1, 6, tzinfo=UTC), 6.0)],
        "id long, event_time timestamp, x double",
    ).write.format("delta").mode("append").saveAsTable(source)
    original = DataFrameWriter.saveAsTable

    def fail_target(writer, table):
        """Inject one target failure after source selection but before Delta commit."""
        if table == target:
            raise RuntimeError("injected target failure")
        return original(writer, table)

    with monkeypatch.context() as patch:
        patch.setattr(DataFrameWriter, "saveAsTable", fail_target)
        with pytest.raises(DeltaPublishError, match="Incremental Delta write failed"):
            run_incremental_local_batch(
                spark,
                prepared,
                row_keys=("id",),
                period_column="event_time",
                admission=admission,
            )
    assert spark.sql(f"DESCRIBE HISTORY {target}").first().version == first.commit_version
    retried = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    assert retried.input_count == 1 and spark.table(target).count() == 3


def test_incremental_local_batch_lost_acknowledgement_is_noop(local_delta_case, monkeypatch):
    """A committed append remains visible when the caller loses its response."""
    from skyulf.integrations.databricks import local_incremental, run_incremental_local_batch

    spark, source, target, prepared, admission = _prepared_incremental_case(local_delta_case)
    run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    spark.createDataFrame(
        [(3, datetime(2026, 1, 6, tzinfo=UTC), 6.0)],
        "id long, event_time timestamp, x double",
    ).write.format("delta").mode("append").saveAsTable(source)
    original = local_incremental._last_receipt
    calls = 0

    def lose_reply(*args):
        """Hide the receipt only after the target append has succeeded."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("lost acknowledgement")
        return original(*args)

    with monkeypatch.context() as patch:
        patch.setattr(local_incremental, "_last_receipt", lose_reply)
        with pytest.raises(RuntimeError, match="lost acknowledgement"):
            run_incremental_local_batch(
                spark,
                prepared,
                row_keys=("id",),
                period_column="event_time",
                admission=admission,
            )
    committed_version = spark.sql(f"DESCRIBE HISTORY {target}").first().version
    retry = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    assert retry.noop and retry.commit_version == committed_version
    assert spark.table(target).count() == 3


def test_incremental_local_batch_respects_shared_admission(local_delta_case):
    """A competing publisher must not score or write while the target is owned."""
    from skyulf.integrations.databricks import run_incremental_local_batch
    from skyulf.integrations.databricks.delta import table_identity

    spark, _, target, prepared, admission = _prepared_incremental_case(local_delta_case)
    before = spark.sql(f"DESCRIBE HISTORY {target}").first().version
    with (
        admission.hold(table_identity(spark, target)),
        pytest.raises(BatchConflictError, match="holds"),
    ):
        run_incremental_local_batch(
            spark,
            prepared,
            row_keys=("id",),
            period_column="event_time",
            admission=admission,
        )
    assert spark.table(target).count() == 0
    assert spark.sql(f"DESCRIBE HISTORY {target}").first().version == before


def test_incremental_local_batch_expired_change_read_does_not_fallback(
    local_delta_case, monkeypatch
):
    """A missing CDF range must fail closed instead of rescoring the full table."""
    DataFrameReader = import_module("pyspark.sql.readwriter").DataFrameReader

    from skyulf.integrations.databricks import run_incremental_local_batch

    spark, source, target, prepared, admission = _prepared_incremental_case(local_delta_case)
    first = run_incremental_local_batch(
        spark, prepared, row_keys=("id",), period_column="event_time", admission=admission
    )
    spark.createDataFrame(
        [(3, datetime(2026, 1, 6, tzinfo=UTC), 6.0)],
        "id long, event_time timestamp, x double",
    ).write.format("delta").mode("append").saveAsTable(source)
    original = DataFrameReader.table

    def expired(reader, table):
        """Model an expired CDF range at the Delta read boundary."""
        if table == source:
            raise RuntimeError("CDF range expired")
        return original(reader, table)

    with monkeypatch.context() as patch:
        patch.setattr(DataFrameReader, "table", expired)
        with pytest.raises(RuntimeError, match="CDF range expired"):
            run_incremental_local_batch(
                spark,
                prepared,
                row_keys=("id",),
                period_column="event_time",
                admission=admission,
            )
    assert spark.table(target).count() == 2
    assert spark.sql(f"DESCRIBE HISTORY {target}").first().version == first.commit_version
