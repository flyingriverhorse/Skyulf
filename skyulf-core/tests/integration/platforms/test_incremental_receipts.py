"""Incremental watermarks survive maintenance without trusting unreceipted writes."""

import json
import os
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from skyulf.integrations.databricks.data.admission import BatchConflictError, SingleWriterAdmission
from skyulf.integrations.databricks.scoring.incremental import incremental_batch as batch


def test_latest_version_limits_delta_history_at_the_source(monkeypatch):
    """Looking up the current version must not ask Delta for its complete history."""
    spark = Mock()
    monkeypatch.setattr(batch, "importlib", SimpleNamespace(import_module=lambda _: Mock()))
    batch.latest_source_version(spark, "workspace.default.predictions")
    spark.sql.assert_called_once_with(
        "DESCRIBE HISTORY `workspace`.`default`.`predictions` LIMIT 1"
    )


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1; DROP TABLE predictions"])
def test_history_limit_rejects_invalid_values_before_sql(limit):
    """A bounded metadata read must never interpolate arbitrary input into SQL."""
    from skyulf.integrations.databricks.data.delta_io.delta import history

    spark = Mock()
    with pytest.raises(ValueError, match="positive integer"):
        history(spark, "workspace.default.predictions", limit=limit)
    spark.sql.assert_not_called()


def test_historical_receipt_queries_keep_unlimited_history():
    """A receipt retry or observation may need a commit older than the latest one."""
    from skyulf.integrations.databricks.data.delta_io.delta import history

    spark = Mock()
    history(spark, "workspace.default.predictions")
    spark.sql.assert_called_once_with("DESCRIBE HISTORY `workspace`.`default`.`predictions`")


@pytest.fixture(params=["single", "model_set"])
def receipt_case(request, delta_spark, tmp_path):
    """Use real fitted bytes and isolated Delta tables for both incremental consumers."""
    from tests.integration.platforms.test_model_set_batch import _saved_set

    from skyulf.inference._manifest import ColumnSpec
    from skyulf.inference.fitted_pipeline import load_pipeline
    from skyulf.integrations.databricks.model_sets.model_set_batch import run_model_set_batch
    from skyulf.integrations.databricks.scoring.workflow import (
        InputSource,
        ModelSelection,
        OutputSink,
        PreflightResult,
        PreparedWorkflow,
        WorkflowConfig,
    )

    spark = delta_spark
    schema = os.environ.get("SKYULF_MODEL_SET_TEST_SCHEMA", "default")
    source = f"{schema}.au01_{uuid4().hex[:12]}_source"
    target = source.removesuffix("_source") + "_predictions"
    artifact, model, _ = _saved_set(tmp_path)
    try:
        spark.createDataFrame(
            [(1, "Riga", 8.0)], "id long, city string, amount double"
        ).write.format("delta").option("delta.enableChangeDataFeed", "true").saveAsTable(source)
        if request.param == "single":
            spark.sql(
                f"CREATE TABLE {target} (id BIGINT, prediction DOUBLE, run_id STRING, "
                "model_name STRING, model_version STRING) USING DELTA"
            ).collect()
            component = load_pipeline(tmp_path / "component")
            config = WorkflowConfig(
                runtime="databricks",
                engine="pandas",
                source=InputSource(
                    kind="uc_table",
                    table=source,
                    read_mode="incremental",
                    max_rows=100,
                    max_bytes=4 * 1024 * 1024,
                ),
                model=ModelSelection(kind="local_pipeline", name=model.name, version=model.version),
                sink=OutputSink(kind="uc_delta", table=target),
            )
            prepared = PreparedWorkflow(
                config,
                component,
                PreflightResult(
                    issues=(),
                    remote_checked=True,
                    model_version=model.version,
                    model_digest=component.manifest.pipeline_sha256,
                    output_schema=(ColumnSpec(name="prediction", dtype="float64"),),
                ),
            )

            def score():
                """Exercise the single-model public entrypoint with explicit single-writer admission."""
                return batch.run_incremental_batch(
                    spark, prepared, record_key_columns=("id",), admission=SingleWriterAdmission()
                )
        else:

            def score():
                """Exercise the model-set public entrypoint with the same receipt contract."""
                return run_model_set_batch(
                    spark,
                    model,
                    artifact,
                    source_table=source,
                    prediction_table=target,
                    admission=SingleWriterAdmission(),
                    max_rows=100,
                    max_bytes=4 * 1024 * 1024,
                )

        yield spark, source, target, score
    finally:
        try:
            spark.sql(f"DROP TABLE IF EXISTS {target}").collect()
        finally:
            spark.sql(f"DROP TABLE IF EXISTS {source}").collect()


def test_incremental_continues_after_target_maintenance(receipt_case):
    """A metadata-only commit must preserve the watermark and the original receipt version."""
    spark, source, target, score = receipt_case
    first = score()
    spark.sql(
        f"ALTER TABLE {target} SET TBLPROPERTIES ('skyulf.test.maintenance' = 'yes')"
    ).collect()
    assert spark.sql(f"DESCRIBE HISTORY {target} LIMIT 1").first().operation == "SET TBLPROPERTIES"
    spark.sql(f"ALTER TABLE {target} UNSET TBLPROPERTIES ('skyulf.test.maintenance')").collect()
    replay = score()
    assert replay.noop and replay.commit_version == first.commit_version
    spark.createDataFrame(
        [(2, "Tallinn", 9.0)], "id long, city string, amount double"
    ).write.format("delta").mode("append").saveAsTable(source)
    spark.sql(f"ALTER TABLE {source} SET TBLPROPERTIES ('skyulf.test.source' = 'yes')").collect()
    second = score()
    assert second.input_count == 1 and not second.noop
    assert (
        second.source_end_version == spark.sql(f"DESCRIBE HISTORY {source} LIMIT 1").first().version
    )
    assert [row.id for row in spark.table(target).orderBy("id").collect()] == [1, 2]


def test_maintenance_does_not_hide_an_unreceipted_data_write(receipt_case):
    """Skipping maintenance must stop at an intervening data change, never an older good receipt."""
    spark, _, target, score = receipt_case
    score()
    spark.sql(f"DELETE FROM {target} WHERE id = 1").collect()
    spark.sql(
        f"ALTER TABLE {target} SET TBLPROPERTIES ('skyulf.test.maintenance' = 'yes')"
    ).collect()
    with pytest.raises(BatchConflictError, match="outside the incremental receipt protocol"):
        score()
    assert spark.table(target).count() == 0


def test_post_write_maintenance_keeps_the_committed_receipt(receipt_case, monkeypatch):
    """Compaction metadata appearing before verification must not turn a saved write into failure."""
    spark, _, target, score = receipt_case
    writer_type = type(spark.range(0).write)
    original = writer_type.saveAsTable

    def save_then_maintain(writer, name, *args, **kwargs):
        """Perform real Delta maintenance immediately after a real target append."""
        result = original(writer, name, *args, **kwargs)
        if name == target:
            spark.sql(
                f"ALTER TABLE {target} SET TBLPROPERTIES ('skyulf.test.maintenance' = 'yes')"
            ).collect()
        return result

    monkeypatch.setattr(writer_type, "saveAsTable", save_then_maintain)
    result = score()
    rows = spark.sql(f"DESCRIBE HISTORY {target}").collect()
    saved = next(row for row in rows if row.version == result.commit_version)
    assert result.output_count == 1
    assert json.loads(saved.userMetadata) == result.manifest


@pytest.mark.parametrize("operation", ["DELETE", None, "FUTURE OPERATION", "CREATE TABLE"])
def test_prediction_commit_keeps_barriers_and_snapshot_bound(delta_spark, monkeypatch, operation):
    """A receipt search cannot jump a data/unknown barrier or see a later concurrent commit."""
    from skyulf.integrations.databricks.data.delta_io import delta

    rows = delta_spark.createDataFrame(
        [
            (0, "WRITE", "old"),
            (1, operation, None),
            (2, "OPTIMIZE", None),
            (3, "VACUUM END", None),
            (4, "WRITE", "future"),
        ],
        "version long, operation string, userMetadata string",
    )
    monkeypatch.setattr(delta, "history", lambda *args: rows)
    selected = delta.prediction_commit(
        delta_spark, "workspace.default.target", {"version": 3, "operation": "VACUUM END"}
    )
    assert selected.version == 1 and selected.operation == operation


def test_maintenance_during_scoring_still_rejects_stale_publication(receipt_case, monkeypatch):
    """Finding the older receipt must not weaken the physical-version write conflict check."""
    from skyulf.integrations.databricks.model_sets import model_set_batch

    spark, source, target, score = receipt_case
    score()
    spark.createDataFrame(
        [(2, "Tallinn", 9.0)], "id long, city string, amount double"
    ).write.format("delta").mode("append").saveAsTable(source)

    def maintain_before_commit(original):
        """Inject a real concurrent maintenance commit before the actual guarded publisher."""

        def commit(*args, **kwargs):
            """Delegate publication only after the physical target version has changed."""
            spark.sql(
                f"ALTER TABLE {target} SET TBLPROPERTIES ('skyulf.test.concurrent' = 'yes')"
            ).collect()
            return original(*args, **kwargs)

        return commit

    monkeypatch.setattr(batch, "_commit_increment", maintain_before_commit(batch._commit_increment))
    monkeypatch.setattr(
        model_set_batch, "_commit_set", maintain_before_commit(model_set_batch._commit_set)
    )
    with pytest.raises(BatchConflictError, match="Target changed while scoring"):
        score()
    assert [row.id for row in spark.table(target).collect()] == [1]
