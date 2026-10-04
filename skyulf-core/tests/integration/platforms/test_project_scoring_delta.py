"""Deliver saved eligibility and output rules through actual Delta transactions.

The file is standalone apart from the ``delta_spark`` fixture. A cloud runner
can supply its existing Spark session and set ``SKYULF_TEST_TABLE_PREFIX`` to
an isolated catalog.schema.table_prefix before importing this module.
"""

import json
import os
import re
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
from skyulf.inference.local_scoring import scoring_output_schema
from skyulf.integrations.databricks import run_incremental_local_batch, run_local_batch
from skyulf.integrations.databricks._contracts import BatchSpec
from skyulf.integrations.databricks.admission import SingleWriterAdmission
from skyulf.integrations.databricks.local_batch import LocalSourceSpec
from skyulf.integrations.databricks.local_sdk import (
    InputSource,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    PreflightResult,
    PreparedLocalWorkflow,
)
from skyulf.integrations.databricks.prediction_output import provision_prediction_table
from skyulf.integrations.databricks.project import load_project_workflow
from skyulf.pipeline import SkyulfPipeline

TABLE_PREFIX = os.environ.get("SKYULF_TEST_TABLE_PREFIX", "spark_catalog.default.sm36a_scoring_")
SOURCE = """
import pandas as pd

def build_preprocessing():
    return []

def eligible(frame, params):
    return pd.Series(["negative_input" if value < params["minimum"] else None
                      for value in frame.x], index=frame.index, dtype="string")

def output_band(frame, predictions, params):
    return pd.DataFrame({"band": ["high" if value >= params["threshold"] else "low"
                                  for value in predictions.prediction]}, index=frame.index)

def build_scoring():
    return {
        "eligibility": [{"name": "nonnegative", "version": "1", "function": "eligible",
                         "params": {"minimum": 0}}],
        "outputs": [{"name": "band", "version": "1", "function": "output_band",
                     "params": {"threshold": 10}, "columns": [{"name": "band", "dtype": "string"}]}],
    }
"""


def _policy_source(policy):
    """Exercise combined filters with distinct pre-split and custom exclusion causes."""
    if policy != "combined":
        return SOURCE
    source = SOURCE.replace('value < params["minimum"]', 'value > params["minimum"]')
    source = source.replace("negative_input", "above_limit").replace('"minimum": 0', '"minimum": 5')
    return (
        source
        + """
_custom_scoring = build_scoring

def build_pre_split_steps():
    return [
        {"name": "known_target", "transformer": "DropMissingRows",
         "params": {"subset": ["target"], "how": "any"}},
        {"name": "nonnegative", "transformer": "ManualBounds",
         "params": {"bounds": {"x": {"lower": 0.0}}}},
    ]

def build_scoring():
    return {"reuse_pre_split": True, "skip_target_steps": True, **_custom_scoring()}
"""
    )


def _saved_artifact(directory, engine, policy):
    """Fit and reload exact saved source after replacing the editable project file."""
    directory.mkdir()
    source = directory / "preprocessing.py"
    source.write_text(_policy_source(policy), encoding="utf-8")
    workflow: dict[str, Any] = {
        "engine": engine,
        "target_column": "target",
        "input_columns": ["x"],
        "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
    }
    if policy:
        workflow = load_project_workflow(workflow, source)
    data = pd.DataFrame({"x": np.arange(20, dtype=float), "target": 2 * np.arange(20, dtype=float)})
    native = pl.from_pandas(data) if engine == "polars" else data
    pipeline = SkyulfPipeline(workflow["pipeline"])
    pipeline.fit(SplitDataset(train=native[:16], test=native[16:]), target_column="target")
    save_local_pipeline(pipeline, directory / "artifact")
    source.write_text("raise RuntimeError('editable source must not execute')", encoding="utf-8")
    return load_local_pipeline(directory / "artifact")


def _latest(spark, table):
    """Read exact Delta commit and receipt evidence for unchanged-state assertions."""
    row = spark.sql(f"DESCRIBE HISTORY {table}").orderBy("version", ascending=False).first()
    return int(row.version), row.userMetadata


def _append(case, rows):
    """Append real new observations to the change-feed-enabled source."""
    case.spark.createDataFrame(
        [(key, datetime(2026, 1, 5, tzinfo=UTC), value) for key, value in rows],
        f"{case.key} long, event_time timestamp, x double",
    ).write.format("delta").mode("append").saveAsTable(case.source)


@pytest.fixture
def make_scoring_delta_case(delta_spark, tmp_path):
    """Own isolated tables and saved models, usable on local Delta or serverless."""
    assert re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", TABLE_PREFIX
    )
    tables = []

    def make(engine, *, period=False, policy=True, key="id"):
        """Provision real source and a target derived from the saved scoring schema."""
        suffix = uuid4().hex
        source, target = f"{TABLE_PREFIX}{suffix}_source", f"{TABLE_PREFIX}{suffix}_target"
        tables.extend([source, target])
        artifact = _saved_artifact(tmp_path / suffix, engine, policy)
        delta_spark.createDataFrame(
            [
                (1, datetime(2026, 1, 5, tzinfo=UTC), 2.0),
                (2, datetime(2026, 1, 6, tzinfo=UTC), -1.0),
                (3, datetime(2026, 1, 7, tzinfo=UTC), 8.0),
            ],
            f"{key} long, event_time timestamp, x double",
        ).write.format("delta").option("delta.enableChangeDataFeed", "true").saveAsTable(source)
        source_version = _latest(delta_spark, source)[0]
        config = LocalWorkflowConfig(
            runtime="databricks",
            engine=engine,
            source=InputSource(
                kind="uc_table",
                table=source,
                version=source_version if period else None,
                read_mode="snapshot" if period else "incremental",
                max_rows=20,
                max_bytes=100_000,
            ),
            model=ModelSelection(
                kind="local_pipeline", name="workspace.test.scoring_model", version="1"
            ),
            sink=OutputSink(kind="uc_delta", table=target),
        )
        preflight = PreflightResult(
            issues=(),
            remote_checked=True,
            model_version="1",
            model_digest=artifact.manifest.pipeline_sha256,
            output_schema=scoring_output_schema(artifact),
        )
        prepared = PreparedLocalWorkflow(config, artifact, preflight)
        if period:
            _create_period_target(delta_spark, target, key, preflight)
        else:
            request = {
                "score_source_table": source,
                "prediction_table": target,
                "model_name": config.model.name,
                "record_key_columns": [key],
                "input_columns": ["x"],
                "max_rows": 20,
            }
            assert provision_prediction_table(delta_spark, request, prepared)
            assert not provision_prediction_table(delta_spark, request, prepared)
        return SimpleNamespace(
            spark=delta_spark,
            policy=policy,
            source=source,
            target=target,
            prepared=prepared,
            key=key,
            source_version=source_version,
            admission=SingleWriterAdmission(),
        )

    try:
        yield make
    finally:
        for table in reversed(tables):
            delta_spark.sql(f"DROP TABLE IF EXISTS {table}").collect()


def _create_period_target(spark, target, key, preflight):
    """Add the period control column to the saved output schema for period publication."""
    types = {"float64": "DOUBLE", "int64": "BIGINT", "string": "STRING", "bool": "BOOLEAN"}
    outputs = ", ".join(
        f"{column.name} {types[column.dtype]}" for column in preflight.output_schema
    )
    spark.sql(
        f"CREATE TABLE {target} ({key} BIGINT, event_time TIMESTAMP, {outputs}, "
        "run_id STRING, model_name STRING, model_version STRING) USING DELTA"
    ).collect()


def _increment(case):
    """Use the public incremental writer with explicit sole-writer admission."""
    return run_incremental_local_batch(
        case.spark, case.prepared, record_key_columns=(case.key,), admission=case.admission
    )


def _assert_initial_rows(case):
    """Check business values, exclusion nulls and exact source key membership."""
    rows = case.spark.table(case.target).orderBy(case.key).collect()
    assert [row[case.key] for row in rows] == [1, 2, 3]
    combined = case.policy == "combined"
    assert [row.scoring_status for row in rows] == [
        "predicted",
        "excluded",
        "excluded" if combined else "predicted",
    ]
    assert [row.band for row in rows] == ["low", None, None if combined else "high"]
    assert [row.exclusion_reason for row in rows] == (
        [None, "pre_split:nonnegative", "above_limit"]
        if combined
        else [None, "negative_input", None]
    )
    assert [row.prediction for row in rows if row.prediction is not None] == pytest.approx(
        [4] if combined else [4, 16]
    )
    assert rows[1].prediction is None
    assert all(row.model_version == "1" and row.run_id for row in rows)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("policy", ["custom", "combined"])
def test_incremental_scoring_rules_publish_exclusions_retry_and_noop(
    make_scoring_delta_case, monkeypatch, engine, policy
):
    """Every input key is committed once, and failed writes never advance a source receipt."""
    from skyulf.integrations.databricks import local_incremental

    case = make_scoring_delta_case(engine, policy=policy)
    first = _increment(case)
    _assert_initial_rows(case)
    assert (first.input_count, first.output_count) == (3, 3)
    assert (first.manifest["predicted_count"], first.manifest["excluded_count"]) == (
        (1, 2) if policy == "combined" else (2, 1)
    )
    assert {
        field.name: field.dataType.typeName()
        for field in case.spark.table(case.target).schema.fields
    } == {
        "id": "long",
        "prediction": "double",
        "band": "string",
        "scoring_status": "string",
        "exclusion_reason": "string",
        "run_id": "string",
        "model_name": "string",
        "model_version": "string",
    }
    before = _latest(case.spark, case.target)
    _append(case, [(4, 4.0)])

    def failed_commit(*args, **kwargs):
        """Inject failure at the commit boundary after real reading and scoring."""
        raise RuntimeError("injected failure before Delta commit")

    with monkeypatch.context() as patch:
        patch.setattr(local_incremental, "_commit_increment", failed_commit)
        with pytest.raises(RuntimeError, match="injected failure"):
            _increment(case)
    assert _latest(case.spark, case.target) == before
    _assert_initial_rows(case)
    retried = _increment(case)
    assert (retried.input_count, retried.output_count) == (1, 1)
    row = case.spark.table(case.target).where("id = 4").first()
    assert row.prediction == pytest.approx(8)
    assert row.band == "low" and row.scoring_status == "predicted"
    _append(case, [(5, -2.0), (6, 9.0 if policy == "combined" else -3.0)])

    def forbidden_model(*args, **kwargs):
        """Prove excluded-only source increments do not invoke the estimator."""
        raise AssertionError("model invoked for excluded-only batch")

    with monkeypatch.context() as patch:
        patch.setattr(case.prepared.artifact.pipeline, "predict", forbidden_model)
        excluded = _increment(case)
    assert (excluded.input_count, excluded.output_count) == (2, 2)
    assert (excluded.manifest["predicted_count"], excluded.manifest["excluded_count"]) == (0, 2)
    assert excluded.source_end_version == _latest(case.spark, case.source)[0]
    committed = _latest(case.spark, case.target)
    receipt = json.loads(committed[1])
    assert receipt["source_end_version"] == excluded.source_end_version
    assert receipt["output_count"] == 2 and receipt["excluded_count"] == 2
    noop = _increment(case)
    assert noop.noop and noop.input_count == noop.output_count == 0
    assert noop.commit_version == excluded.commit_version
    assert _latest(case.spark, case.target) == committed
    rows = case.spark.table(case.target).orderBy("id").collect()
    assert [row.id for row in rows] == [1, 2, 3, 4, 5, 6]
    assert all(
        row.scoring_status == "excluded" and row.prediction is None and row.band is None
        for row in rows[-2:]
    )


def _period_request(case):
    """Pin the exact source and current target version for January publication."""
    source = LocalSourceSpec(
        table=case.source,
        version=case.source_version,
        period_start=datetime(2026, 1, 1, tzinfo=UTC),
        period_end=datetime(2026, 2, 1, tzinfo=UTC),
        record_key_columns=(case.key,),
        input_columns=("x",),
        max_rows=20,
        max_bytes=100_000,
    )
    request = BatchSpec(
        period_start=source.period_start,
        period_end=source.period_end,
        as_of=datetime.now(UTC) + timedelta(minutes=10),
        record_key_columns=(case.key,),
        output_table=case.target,
        model_name=case.prepared.config.model.name,
        model_version="1",
        source_version=case.source_version,
        code_version=version("skyulf-core"),
        run_id="scoring-period-" + uuid4().hex,
        model_digest=case.prepared.artifact.manifest.pipeline_sha256,
        expected_target_version=_latest(case.spark, case.target)[0],
        mode="local_pipeline",
    )
    return source, request


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_period_scoring_rules_publish_all_input_outcomes_and_replay(
    make_scoring_delta_case, engine
):
    """Period receipts preserve excluded membership and explicit scoring coverage on replay."""
    case = make_scoring_delta_case(engine, period=True)
    source, request = _period_request(case)
    first = run_local_batch(case.spark, source, case.prepared, request, admission=case.admission)
    _assert_initial_rows(case)
    assert (first.input_count, first.output_count) == (3, 3)
    assert (first.manifest["predicted_count"], first.manifest["excluded_count"]) == (2, 1)
    committed = _latest(case.spark, case.target)
    replay = run_local_batch(case.spark, source, case.prepared, request, admission=case.admission)
    assert replay.replayed and replay.commit_version == first.commit_version
    assert _latest(case.spark, case.target) == committed
    assert replay.manifest == first.manifest


def test_legacy_period_record_key_named_scoring_status_is_not_policy_metadata(
    make_scoring_delta_case,
):
    """A pre-policy business key must not be interpreted as scoring outcome values."""
    case = make_scoring_delta_case("pandas", period=True, policy=False, key="scoring_status")
    source, request = _period_request(case)
    result = run_local_batch(case.spark, source, case.prepared, request, admission=case.admission)
    rows = case.spark.table(case.target).orderBy("scoring_status").collect()
    assert [row.scoring_status for row in rows] == [1, 2, 3]
    assert [row.prediction for row in rows] == pytest.approx([4, -2, 16])
    assert (result.input_count, result.output_count) == (3, 3)
    assert "predicted_count" not in result.manifest and "excluded_count" not in result.manifest
