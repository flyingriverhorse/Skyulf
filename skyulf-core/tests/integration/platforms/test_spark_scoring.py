"""Distributed scoring keeps population data out of the local batch bridge."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from tests.integration.platforms.test_local_cdf_recovery import recovery_case as recovery_case

from skyulf.integrations.databricks.local_sdk import LocalWorkflowConfig


def _config(**overrides):
    """Retain local training budgets independently of distributed inference."""
    return LocalWorkflowConfig.model_validate(
        {
            "runtime": "databricks",
            "engine": "pandas",
            "source": {
                "kind": "uc_table",
                "table": "db.source",
                "read_mode": "incremental",
                "max_rows": 1,
                "max_bytes": 1024,
            },
            "model": {"kind": "local_pipeline", "name": "db.model", "version": "1"},
            "sink": {"kind": "uc_delta", "table": "db.target"},
            **overrides,
        }
    )


def test_legacy_configs_keep_local_inference():
    """Existing saved projects must not change execution mode on upgrade."""
    assert _config().inference_mode == "local"


@pytest.mark.parametrize("model_set", [False, True])
def test_prediction_batch_limit_does_not_mutate_restricted_spark_settings(monkeypatch, model_set):
    """Serverless must bound model calls without reading or setting forbidden Arrow config."""
    from skyulf.integrations.databricks import spark_scoring
    from skyulf.integrations.mlflow import spark_model

    spark = Mock()
    spark.conf.set.side_effect = AssertionError("serverless Spark setting unavailable")
    output = Mock()
    output.count.return_value = 3
    predict = Mock(return_value=output)
    monkeypatch.setattr(spark_model, "predict_spark_pyfunc", predict)
    artifact = SimpleNamespace(manifest=SimpleNamespace(record_key_columns=("id",)))
    if model_set:
        execution = spark_scoring.SparkSetExecution(
            spark, artifact, "models:/db.model/1", "virtualenv", 7
        )
        spark_scoring.score_distributed_set(spark_scoring.DistributedRows(Mock(), 3, execution))
    else:
        config = _config(inference_mode="spark")
        prepared = SimpleNamespace(config=config, artifact=artifact)
        spark_scoring.score_distributed_single(
            spark,
            prepared,
            spark_scoring.DistributedRows(Mock(), 3),
            ("id",),
            Mock(),
            replacing=True,
        )
    assert predict.call_args.kwargs["prediction_batch_rows"] == (7 if model_set else 10000)
    spark.conf.set.assert_not_called()
    spark.conf.get.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"engine": "polars"},
        {"runtime": "local"},
        {"spark_udf_env_manager": "conda"},
        {"spark_udf_prediction_batch_rows": 0},
        {"spark_udf_prediction_batch_rows": True},
    ],
)
def test_invalid_distributed_runtime_rejected(overrides):
    """Contradictory settings fail before source reads or registry access."""
    with pytest.raises(ValueError):
        _config(inference_mode="spark", **overrides)


@pytest.mark.parametrize(
    "override",
    [
        {"source": {"kind": "caller_frame"}},
        {"sink": {"kind": "return_frame"}},
        {"model": {"kind": "local_pipeline", "path": "/trusted/model"}},
        {"source": {"kind": "uc_table", "table": "db.source", "version": 1}},
    ],
)
def test_spark_mode_rejects_local_only_sdk_routes(override):
    """An explicit distributed request must never reach a driver materialization path."""
    with pytest.raises(ValueError, match="Spark inference"):
        _config(inference_mode="spark", **override)


def test_prepared_spark_predict_rejects_driver_execution():
    """Even a direct call to the prepared local API cannot silently ignore Spark mode."""
    import pandas as pd

    from skyulf.integrations.databricks.local_sdk import PreparedLocalWorkflow

    prepared = PreparedLocalWorkflow(_config(inference_mode="spark"), Mock(), Mock())
    with pytest.raises(ValueError, match="distributed"):
        prepared.predict(pd.DataFrame({"x": [1.0]}))


def test_distributed_source_retains_large_population_on_workers(monkeypatch):
    """A local row budget must not truncate a distributed scoring population."""
    from skyulf.integrations.databricks import spark_scoring

    selected = MagicMock()
    frame = selected.select.return_value
    frame.agg.return_value.first.return_value = {"rows": 2_500_000, "keys": 2_500_000, "missing": 0}
    functions = MagicMock()
    monkeypatch.setattr(spark_scoring.importlib, "import_module", lambda name: functions)
    batch = spark_scoring.read_distributed_rows(selected, ("id", "x"), ("id",))
    assert len(batch) == 2_500_000 and not batch.empty
    assert batch.frame is frame
    frame.toPandas.assert_not_called()
    frame.toLocalIterator.assert_not_called()
    frame.collect.assert_not_called()


@pytest.mark.parametrize("null_keys", [True, False])
def test_distributed_keys_reject_null_or_duplicate_population(monkeypatch, null_keys):
    """Key validity must be checked globally before prediction or publication."""
    from skyulf.integrations.databricks import spark_scoring

    selected = MagicMock()
    frame = selected.select.return_value
    frame.agg.return_value.first.return_value = {
        "rows": 2,
        "keys": 2 if null_keys else 1,
        "missing": int(null_keys),
    }
    monkeypatch.setattr(spark_scoring.importlib, "import_module", lambda name: MagicMock())
    with pytest.raises(ValueError, match="null|unique"):
        spark_scoring.read_distributed_rows(selected, ("id", "x"), ("id",))


def test_unsupported_spark_model_fails_before_source_or_target(monkeypatch):
    """Even target provisioning must wait until partition safety is proven."""
    from skyulf.integrations.databricks import local_workflow, spark_scoring

    prepared = SimpleNamespace(config=_config(inference_mode="spark"))
    monkeypatch.setattr(local_workflow, "prepare_local_workflow", lambda config: prepared)
    monkeypatch.setattr(local_workflow, "_scoring_config", lambda config: prepared.config)
    monkeypatch.setattr(local_workflow, "scoring_target", lambda config: "db.target")
    monkeypatch.setattr(
        spark_scoring, "validate_prepared_spark", Mock(side_effect=ValueError("unsafe step"))
    )
    provision = Mock()
    monkeypatch.setattr(local_workflow, "provision_prediction_table", provision)
    spark = Mock()
    with pytest.raises(ValueError, match="unsafe step"):
        local_workflow._run_scoring_action(
            spark,
            {
                "model_name": "db.model",
                "model_version": "1",
                "record_key_columns": ["id"],
                "inference_mode": "spark",
            },
            selection="pinned_version",
            tracking_uri="databricks",
            registry_uri="databricks-uc",
        )
    provision.assert_not_called()
    spark.assert_not_called()


@pytest.fixture
def distributed_case(recovery_case, monkeypatch):
    """Exercise existing Delta/CDF lifecycle with driver materialization forbidden."""
    from tests.integration.platforms.test_local_cdf_recovery import Frame

    from skyulf.integrations.databricks import local_incremental as batch
    from skyulf.integrations.databricks.spark_scoring import DistributedRows

    store = recovery_case
    config = store.prepared.config
    store.prepared = replace(
        store.prepared,
        config=config.model_copy(
            update={
                "inference_mode": "spark",
                "source": config.source.model_copy(update={"max_rows": 1, "max_bytes": 1}),
            }
        ),
    )
    monkeypatch.setattr(batch, "validate_prepared_spark", lambda prepared: None)
    monkeypatch.setattr(
        batch, "bounded_frame", Mock(side_effect=AssertionError("driver collection"))
    )
    monkeypatch.setattr(
        batch,
        "read_distributed_rows",
        lambda selected, columns, keys: DistributedRows(
            selected.select(*columns), selected.count()
        ),
    )

    def predict(spark, prepared, rows, keys, target, *, replacing):
        """Supply a worker result while preserving the actual commit and membership checks."""
        data = rows.frame.data.loc[:, list(keys)].copy()
        data["prediction"] = rows.frame.data.x * 2
        return Frame(data, {"id": "long", "prediction": "double"}, store), {}

    monkeypatch.setattr(batch, "score_distributed_single", predict)
    return store


def test_distributed_recovery_commits_pinned_snapshot_and_noop_replay(distributed_case):
    """Spark recovery must retain overwrite receipts, model pins and replay idempotence."""
    from tests.integration.platforms.test_local_cdf_recovery import recover, recovery_request

    store = distributed_case
    request = recovery_request(store)
    store.source_version = 11
    result = recover(store, request)
    assert result.input_count == result.output_count == 2
    assert result.source_end_version == 10
    assert result.selected_model_version == "2"
    assert store.target.data.prediction.tolist() == [6.0, 10.0]
    assert store.writes[0][0] == "overwrite"
    assert store.previous["cdf_recovered"] is True
    replay = recover(store, request)
    assert replay.noop and len(store.writes) == 1


def test_distributed_worker_failure_keeps_previous_receipt(distributed_case, monkeypatch):
    """A failing UDF cannot publish a partial population or advance its watermark."""
    from tests.integration.platforms.test_local_cdf_recovery import recover, recovery_request

    from skyulf.integrations.databricks import local_incremental as batch

    store = distributed_case
    request = recovery_request(store)
    previous = dict(store.previous)
    monkeypatch.setattr(
        batch, "score_distributed_single", Mock(side_effect=RuntimeError("worker failed"))
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        recover(store, request)
    assert store.previous == previous and store.writes == []


def test_distributed_concurrent_target_change_prevents_write(distributed_case, monkeypatch):
    """The same commit guard must cover target changes during distributed execution."""
    from tests.integration.platforms.test_local_cdf_recovery import recover, recovery_request

    from skyulf.integrations.databricks import local_incremental as batch
    from skyulf.integrations.databricks.admission import BatchConflictError

    store = distributed_case
    request = recovery_request(store)
    original = batch.score_distributed_single

    def predict(*args, **kwargs):
        """Simulate a foreign commit after source selection but before publication."""
        store.target_version += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(batch, "score_distributed_single", predict)
    with pytest.raises(BatchConflictError, match="Target changed"):
        recover(store, request)
    assert store.writes == [] and store.target.data.prediction.tolist() == [-1.0]


def test_distributed_model_set_failure_never_commits(tmp_path, monkeypatch):
    """One failed component must abort the complete set transaction."""
    from tests.integration.platforms.test_model_set_batch import _saved_set, _transport

    from skyulf.integrations.databricks.spark_scoring import DistributedRows, SparkSetExecution

    artifact, model, query = _saved_set(tmp_path)
    batch, commit = _transport(monkeypatch, query)
    execution = SparkSetExecution(None, artifact, model.model_uri, "local", 3)
    monkeypatch.setattr(
        batch,
        "read_distributed_rows",
        lambda *args, **kwargs: DistributedRows(Mock(), len(query), execution),
    )
    monkeypatch.setattr(batch, "bounded_frame", Mock(side_effect=AssertionError("driver")))
    monkeypatch.setattr(
        batch, "score_distributed_set", Mock(side_effect=RuntimeError("component failed"))
    )
    with pytest.raises(RuntimeError, match="component failed"):
        batch._run_admitted_set(
            None,
            model,
            artifact,
            "source",
            "target",
            "source",
            "target",
            "incremental_append",
            1,
            1,
            execution=execution,
        )
    commit.assert_not_called()


def test_distributed_model_set_new_release_rebuilds_all_rows(tmp_path, monkeypatch):
    """Full rebuild retains registry release provenance even for identical model bytes."""
    from tests.integration.platforms.test_model_set_batch import _saved_set, _transport

    from skyulf.integrations.databricks.spark_scoring import (
        DistributedRows,
        DistributedSetResult,
        SparkSetExecution,
    )

    artifact, model, query = _saved_set(tmp_path)
    previous = {
        "skyulf_mode": "incremental_append",
        "artifact_kind": "model_set",
        "source_table_id": "source",
        "target_table_id": "target",
        "source_end_version": 0,
        "model_set_name": model.name,
        "model_set_version": "2",
        "model_set_digest": model.digest,
        "set_history": {},
    }
    batch, commit = _transport(monkeypatch, query, previous)
    execution = SparkSetExecution(None, artifact, model.model_uri, "local", 3)
    rows = DistributedRows(Mock(), len(query), execution)
    monkeypatch.setattr(batch, "read_distributed_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(batch, "bounded_frame", Mock(side_effect=AssertionError("driver")))
    monkeypatch.setattr(
        batch, "score_distributed_set", lambda frame: DistributedSetResult(rows, {})
    )
    result = batch._run_admitted_set(
        None,
        model,
        artifact,
        "source",
        "target",
        "source",
        "target",
        "full_rebuild",
        1,
        1,
        execution=execution,
    )
    assert result.input_count == result.output_count == 3 and not result.noop
    assert result.manifest["model_set_version"] == "1"
    assert commit.call_args.args[-1] == "overwrite"
