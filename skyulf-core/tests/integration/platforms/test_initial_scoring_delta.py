"""Exercise start checkpoints and bounded rebuilds through real Delta transactions."""

import os
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference._manifest import ColumnSpec
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.model_set import ComponentReference, save_model_set
from skyulf.integrations.databricks.data.admission import BatchConflictError, SingleWriterAdmission
from skyulf.integrations.databricks.data.delta_io.cdf_recovery import (
    CdfHistoryExpired,
    CdfRecoveryRequired,
)
from skyulf.integrations.databricks.model_sets import model_set_batch as sets
from skyulf.integrations.databricks.scoring.incremental import incremental_batch as single
from skyulf.integrations.databricks.scoring.shared.prediction_output import (
    activate_prediction_view,
    active_prediction_generation,
)
from skyulf.integrations.databricks.scoring.workflow import (
    InputSource,
    ModelSelection,
    OutputSink,
    PreflightResult,
    PreparedWorkflow,
    WorkflowConfig,
)
from skyulf.integrations.mlflow.registration.registry import ResolvedModel
from skyulf.pipeline import SkyulfPipeline


@pytest.fixture(params=["single", "polars", "model_set"])
def start_case(request, delta_spark, tmp_path):
    """Provision only UUID-owned source/output tables and real fitted artifacts."""
    spark = delta_spark
    schema = os.environ.get("SKYULF_MODEL_SET_TEST_SCHEMA", "default")
    base = f"{schema}.fromnow_{uuid4().hex[:12]}"
    source, target = base + "_source", base + "_predictions"
    train = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]}
    )
    engine = "polars" if request.param == "polars" else "pandas"
    if engine == "polars":
        import polars as pl

        train = pl.from_pandas(train)
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=train[:4], test=train[4:]), target_column="y")
    component_path = tmp_path / "component"
    save_pipeline(pipeline, component_path)
    component = load_pipeline(component_path)
    artifact = component
    model = ResolvedModel(
        base + "_model",
        "1",
        f"models:/{base}_model/1",
        None,
        component.manifest.pipeline_sha256,
    )
    if request.param == "model_set":
        artifact = save_model_set(
            tmp_path / "set",
            {
                "value": (
                    ComponentReference(
                        name=model.name, version="1", digest=component.manifest.pipeline_sha256
                    ),
                    component_path,
                )
            },
            record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        )
        model = replace(model, digest=artifact.manifest.set_sha256)
    config = WorkflowConfig(
        runtime="databricks",
        engine=engine,
        initial_scoring="from_now",
        source=InputSource(
            kind="uc_table", table=source, read_mode="incremental", max_rows=100, max_bytes=1048576
        ),
        model=ModelSelection(kind="local_pipeline", name=model.name, version="1"),
        sink=OutputSink(kind="uc_delta", table=target),
    )
    prepared = PreparedWorkflow(
        config,
        component,
        PreflightResult(
            issues=(),
            remote_checked=True,
            model_version="1",
            model_digest=model.digest,
            output_schema=(ColumnSpec(name="prediction", dtype="float64"),),
        ),
    )
    case = SimpleNamespace(
        spark=spark,
        source=source,
        target=target,
        model=model,
        artifact=artifact,
        prepared=prepared,
        kind=request.param,
        base=base,
        component_path=component_path,
    )
    try:
        spark.createDataFrame([(1, 1.0), (2, 2.0)], "id long, x double").write.format(
            "delta"
        ).option("delta.enableChangeDataFeed", "true").saveAsTable(source)
        if request.param != "model_set":
            _create_single_target(spark, target)
        yield case
    finally:
        if spark.catalog.tableExists(target) and spark.catalog.getTable(target).tableType == "VIEW":
            spark.sql(f"DROP VIEW {target}").collect()
        for name in (target + "_v1", target + "_v2", target, source):
            spark.sql(f"DROP TABLE IF EXISTS {name}").collect()


def _create_single_target(spark, target):
    """Match the exact single-model publication schema, including provenance."""
    spark.sql(
        f"CREATE TABLE {target} (id BIGINT, prediction DOUBLE, run_id STRING, "
        "model_name STRING, model_version STRING) USING DELTA"
    ).collect()


def _run(case, *, request=None, mode="from_now", version="1", rebuild=False, previous=None):
    """Use each layout's public scoring API with the same fixed selection policy."""
    if case.kind == "model_set":
        return sets.run_model_set_batch(
            case.spark,
            replace(case.model, version=version, model_uri=f"models:/{case.model.name}/{version}"),
            case.artifact,
            source_table=case.source,
            prediction_table=case.target,
            admission=SingleWriterAdmission(),
            initial_scoring=mode,
            model_change_mode="full_rebuild" if rebuild else "incremental_append",
            recovery_request=request,
        )
    config = case.prepared.config.model_copy(
        update={
            "initial_scoring": mode,
            "model": case.prepared.config.model.model_copy(update={"version": version}),
            "sink": OutputSink(kind="uc_delta", table=case.target),
        }
    )
    prepared = replace(
        case.prepared,
        config=config,
        preflight=replace(case.prepared.preflight, model_version=version),
    )
    return single.run_incremental_batch(
        case.spark,
        prepared,
        record_key_columns=("id",),
        admission=SingleWriterAdmission(),
        recovery_request=request,
        previous_prediction_table=previous,
    )


def _append(case, key):
    """Append a genuinely new immutable key in its own source commit."""
    case.spark.createDataFrame([(key, float(key))], "id long, x double").write.format("delta").mode(
        "append"
    ).saveAsTable(case.source)


def test_start_checkpoint_increment_and_recovery(start_case, monkeypatch):
    """Recovery/retries must preserve zero history plus every admitted new record."""
    case = start_case
    first = _run(case)
    assert first.input_count == first.output_count == 0 and not first.noop
    assert first.manifest["scoring_start_version"] == first.source_end_version
    assert case.spark.table(case.target).count() == 0
    replay = _run(case)
    assert replay.noop and replay.commit_version == first.commit_version
    _append(case, 3)
    increment = _run(case)
    assert increment.input_count == increment.output_count == 1
    with pytest.raises(BatchConflictError, match="initial_scoring"):
        _run(case, mode="full_snapshot")
    _append(case, 4)
    module = sets if case.kind == "model_set" else single
    original = module.select_incremental_rows

    def expired(spark, source, prior, upper, period, functions):
        """Simulate only unavailable CDF; snapshots and Delta writes remain real."""
        if prior is not None:
            raise CdfHistoryExpired("test-owned expired CDF window")
        return original(spark, source, prior, upper, period, functions)

    with monkeypatch.context() as patch:
        patch.setattr(module, "select_incremental_rows", expired)
        with pytest.raises(CdfRecoveryRequired) as failure:
            _run(case)
    request = failure.value.request
    _append(case, 5)
    recovered = _run(case, request=request)
    assert recovered.input_count == recovered.output_count == 2
    assert recovered.manifest["scoring_start_version"] == first.source_end_version
    assert _run(case, request=request).noop
    assert _run(case).output_count == 1
    rows = case.spark.table(case.target).orderBy("id").collect()
    assert [row.id for row in rows] == [3, 4, 5]


def test_model_change_keeps_original_start(start_case, monkeypatch):
    """A replacement release rescoring new rows must not reset or backfill its boundary."""
    case = start_case
    if case.kind != "model_set":
        case.target += "_v1"
        _create_single_target(case.spark, case.target)
    first = _run(case)
    logical = case.base + "_predictions"
    if case.kind != "model_set":
        # The fixture's unused stable target is test-owned and intentionally replaced by a view.
        case.spark.sql(f"DROP TABLE {logical}").collect()
        activate_prediction_view(case.spark, logical, case.target)
        assert active_prediction_generation(case.spark, logical) == case.target
    try:
        _append(case, 3)
        assert _run(case).output_count == 1
        case.spark.sql(
            f"ALTER TABLE {case.target} SET TBLPROPERTIES ('skyulf.test.maintenance' = 'yes')"
        ).collect()
        retained_target_version = int(
            single.latest_source_version(case.spark, case.target)["version"]
        )
        _append(case, 4)
        module = sets if case.kind == "model_set" else single
        snapshot = module.scoring_snapshot

        def durable_snapshot(*args, **kwargs):
            """A normal model change must use retained predictions plus recent insert CDF."""
            assert kwargs["checkpoint"] is not None
            assert kwargs["checkpoint"].source_version == first.source_end_version + 1
            assert kwargs["checkpoint"].version == retained_target_version
            return snapshot(*args, **kwargs)

        monkeypatch.setattr(module, "scoring_snapshot", durable_snapshot)
        previous = None
        if case.kind != "model_set":
            previous = active_prediction_generation(case.spark, logical)
            case.target = logical + "_v2"
            _create_single_target(case.spark, case.target)
        rebuilt = _run(case, version="2", rebuild=True, previous=previous)
        assert rebuilt.output_count == 2
        assert rebuilt.manifest["scoring_start_version"] == first.source_end_version
        if case.kind != "model_set":
            activate_prediction_view(case.spark, logical, case.target)
        assert sorted(row.id for row in case.spark.table(case.target).collect()) == [3, 4]
    finally:
        if case.kind != "model_set":
            case.spark.sql(f"DROP VIEW IF EXISTS {logical}").collect()


@pytest.mark.parametrize("start_case", ["model_set"], indirect=True)
def test_model_set_correction_uses_only_admitted_and_new_keys(start_case, monkeypatch):
    """Corrections to excluded historical keys cannot enroll them in a rebuilt population."""
    case = start_case
    first = _run(case)
    _append(case, 3)
    assert _run(case).output_count == 1
    case.spark.sql(f"UPDATE {case.source} SET x = x + 10").collect()
    _append(case, 4)
    original = sets.scoring_snapshot

    def durable_snapshot(*args, **kwargs):
        """Source corrections share the same retained-key rebuild path as model changes."""
        assert kwargs["checkpoint"] is not None
        return original(*args, **kwargs)

    monkeypatch.setattr(sets, "scoring_snapshot", durable_snapshot)
    corrected = sets.run_model_set_batch(
        case.spark,
        case.model,
        case.artifact,
        source_table=case.source,
        prediction_table=case.target,
        admission=SingleWriterAdmission(),
        initial_scoring="from_now",
        source_change_policy="rebuild_on_change",
    )
    assert corrected.manifest is not None
    assert corrected.output_count == 2 and corrected.manifest["source_rebuilt"]
    assert corrected.manifest["scoring_start_version"] == first.source_end_version
    rows = case.spark.table(case.target).orderBy("id").collect()
    assert [row.id for row in rows] == [3, 4]
    assert rows[0].value__prediction == pytest.approx(26.0)


@pytest.mark.parametrize("start_case", ["single", "model_set"], indirect=True)
def test_registered_spark_udf_respects_start_boundary(start_case, tmp_path):
    """Real registered packages must skip history before Spark workers predict new inserts."""
    if os.environ.get("SKYULF_INITIAL_SCORING_UDF") != "1":
        pytest.skip("Explicit Databricks registered-model acceptance lane")
    mlflow = pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.models.model_set import log_model_set
    from skyulf.integrations.mlflow.models.pipeline_model import log_pipeline_model
    from skyulf.integrations.mlflow.registration.registry import register_model

    case = start_case
    client = mlflow.MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    folder = os.environ["SKYULF_INITIAL_SCORING_EXPERIMENT_PARENT"]
    experiment = client.create_experiment(f"{folder}/{case.base.rsplit('.', 1)[-1]}")
    run = client.create_run(experiment)
    try:
        logger = log_model_set if case.kind == "model_set" else log_pipeline_model
        path = tmp_path / "set" if case.kind == "model_set" else case.component_path
        uri = logger(path, run_id=run.info.run_id, artifact_path="model", tracking_uri="databricks")
        registered = register_model(
            uri, case.model.name, tracking_uri="databricks", registry_uri="databricks-uc"
        )
        assert registered.version == "1"
        case.prepared = replace(
            case.prepared,
            config=case.prepared.config.model_copy(
                update={
                    "inference_mode": "spark",
                    "spark_udf_env_manager": "local",
                    "engine": "pandas",
                }
            ),
        )

        def score():
            """Keep the complete production Spark execution and registry-resolution route."""
            if case.kind == "model_set":
                return sets.run_model_set_batch(
                    case.spark,
                    case.model,
                    case.artifact,
                    source_table=case.source,
                    prediction_table=case.target,
                    admission=SingleWriterAdmission(),
                    initial_scoring="from_now",
                    inference_mode="spark",
                    spark_udf_env_manager="local",
                    tracking_uri="databricks",
                    registry_uri="databricks-uc",
                )
            return _run(case)

        first = score()
        assert first.input_count == first.output_count == 0 and not first.noop
        _append(case, 3)
        second = score()
        assert second.input_count == second.output_count == 1
        rows = case.spark.table(case.target).collect()
        prediction = "value__prediction" if case.kind == "model_set" else "prediction"
        assert len(rows) == 1 and rows[0].id == 3
        assert rows[0][prediction] == pytest.approx(6.0)
        assert score().noop
    finally:
        # These exact UUID names are created above; no existing model/experiment is selected.
        try:
            client.delete_registered_model(case.model.name)
        finally:
            client.set_terminated(run.info.run_id)
            client.delete_experiment(experiment)
