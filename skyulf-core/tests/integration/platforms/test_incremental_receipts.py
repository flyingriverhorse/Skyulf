"""Incremental watermarks survive maintenance without trusting unreceipted writes."""

import json
import os
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
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


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_empty_increment_keeps_temporal_continuation(tmp_path, engine):
    """Advancing an empty checkpoint must retain the last rows needed for rolling features."""
    import numpy as np
    import pandas as pd
    from tests.integration.platforms.test_temporal_history import fitted_temporal_pipeline

    from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
    from skyulf.integrations.databricks.scoring.incremental.history import (
        history_receipt,
        incremental_history,
    )
    from skyulf.integrations.databricks.scoring.workflow import PreparedWorkflow

    save_pipeline(fitted_temporal_pipeline(engine), tmp_path / "model")
    artifact = load_pipeline(tmp_path / "model")
    prepared = cast(PreparedWorkflow, SimpleNamespace(artifact=artifact))
    rows = pd.DataFrame({"t": [20, 21, 22], "v": [20.0, 21.0, 22.0]})
    with incremental_history(prepared, None) as first:
        predict_pipeline(rows[:2], artifact)
    previous = history_receipt(first)
    with incremental_history(prepared, previous) as empty:
        predictions = batch._empty_incremental_predictions(prepared, ("prediction",))
    assert predictions.empty and list(predictions.columns) == ["prediction"]
    assert history_receipt(empty) == previous
    with incremental_history(prepared, history_receipt(empty)):
        actual = predict_pipeline(rows[2:], artifact)
    with incremental_history(prepared, None):
        expected = predict_pipeline(rows, artifact)
    np.testing.assert_allclose(actual["prediction"], expected["prediction"].iloc[2:])


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


@pytest.mark.parametrize("bootstrap", [False, True])
def test_empty_increment_commits_watermark_and_replays_once(receipt_case, bootstrap):
    """Empty source windows must advance durable CDF progress without inventing predictions."""
    spark, source, target, score = receipt_case
    if bootstrap:
        spark.sql(f"DELETE FROM {source}").collect()
        first = None
    else:
        first = score()
        spark.sql(f"ALTER TABLE {source} SET TBLPROPERTIES ('skyulf.test.empty' = 'yes')").collect()
    upper = batch.latest_source_version(spark, source)["version"]
    result = score()
    assert not result.noop
    assert result.input_count == result.output_count == 0
    assert result.manifest["source_end_version"] == upper
    assert result.manifest["source_start_version"] == (
        None if first is None else first.source_end_version + 1
    )
    saved = batch.latest_source_version(spark, target)
    assert saved["version"] == result.commit_version
    assert json.loads(saved["userMetadata"]) == result.manifest
    assert spark.table(target).count() == (0 if bootstrap else 1)
    replay = score()
    assert replay.noop and replay.manifest == result.manifest
    assert replay.commit_version == result.commit_version
    assert batch.latest_source_version(spark, target)["version"] == result.commit_version
    _assert_empty_monitoring_window(spark, source, target, result)
    spark.createDataFrame(
        [(2, "Tallinn", 9.0)], "id long, city string, amount double"
    ).write.format("delta").mode("append").saveAsTable(source)
    next_result = score()
    assert next_result.input_count == next_result.output_count == 1
    assert next_result.manifest["source_start_version"] == upper + 1
    assert [row.id for row in spark.table(target).orderBy("id").collect()] == (
        [2] if bootstrap else [1, 2]
    )


def _assert_empty_monitoring_window(spark, source, target, result):
    """A checkpoint without output must not refresh batch freshness in either reader."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
    )
    from skyulf.integrations.databricks.observability.monitoring.monitoring_sources import (
        read_current_observation,
    )
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_sources import (
        read_spark_observation,
    )

    catalog = spark.catalog.currentCatalog()
    source = source if source.count(".") == 2 else f"{catalog}.{source}"
    target = target if target.count(".") == 2 else f"{catalog}.{target}"
    model_set = result.manifest.get("artifact_kind") == "model_set"
    config = MonitorConfig(
        environment="test",
        project="empty_checkpoint",
        model_name="workspace.test.amount" if model_set else "workspace.test.set",
        model_version="1",
        source_table=source,
        prediction_table=target,
        **(
            {
                "model_set_name": "workspace.test.set",
                "model_set_version": "1",
                "model_set_branch": "amount",
            }
            if model_set
            else {}
        ),
    )
    start = spark.sql(f"DESCRIBE HISTORY {target} LIMIT 1").first().timestamp.replace(tzinfo=UTC)
    end = datetime.now(UTC)
    for reader in (read_current_observation, read_spark_observation):
        current, predictions, evidence, observed = reader(
            spark,
            config,
            "1",
            ("id",),
            ("city", "amount"),
            probabilities=0,
            as_of=end,
            start=start,
            end=end,
        )
        assert evidence["batches"] == [] and observed is None
        if reader is read_current_observation:
            assert current.empty and predictions.empty
        else:
            assert current.count() == predictions.count() == 0


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


@pytest.mark.parametrize("empty", [False, True])
def test_maintenance_during_scoring_still_rejects_stale_publication(
    receipt_case, monkeypatch, empty
):
    """Finding the older receipt must not weaken the physical-version write conflict check."""
    from skyulf.integrations.databricks.model_sets import model_set_batch

    spark, source, target, score = receipt_case
    score()
    if empty:
        spark.sql(f"ALTER TABLE {source} SET TBLPROPERTIES ('skyulf.test.empty' = 'yes')").collect()
    else:
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
