"""Real Delta transactions for monthly Spark batch inference."""

import importlib
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest

from skyulf.core.execution import ExecutionOptions
from skyulf.data.dataset import SplitDataset
from skyulf.inference.bundle import build_bundle
from skyulf.integrations.databricks.batch import BatchSpec, run_batch
from skyulf.pipeline import SkyulfPipeline


@pytest.fixture
def harness(delta_spark, tmp_path):
    """Create test-owned source and output tables with a separate prior period."""
    from skyulf.integrations.databricks.admission import LocalTableLock

    spark = delta_spark
    name = uuid4().hex
    source, target = f"default.source_{name}", f"default.predictions_{name}"
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 2.0, 4.0]})
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}})
    pipeline.fit(SplitDataset(train=data, test=data.head(0)), target_column="y")
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
    spark.createDataFrame(
        [
            (1, datetime(2026, 1, 1, tzinfo=UTC), 1.0),
            (2, datetime(2026, 1, 31, tzinfo=UTC), 2.0),
            (3, datetime(2026, 2, 1, tzinfo=UTC), 3.0),
        ],
        "id long, event_time timestamp, x double",
    ).write.format("delta").saveAsTable(source)
    spark.createDataFrame(
        [
            (9, datetime(2025, 12, 1, tzinfo=UTC), 99.0, "prior", "risk", "6"),
        ],
        "id long, event_time timestamp, prediction double, run_id string, "
        "model_name string, model_version string",
    ).write.format("delta").saveAsTable(target)
    spec = BatchSpec(
        period_start=datetime(2026, 1, 1, tzinfo=UTC),
        period_end=datetime(2026, 2, 1, tzinfo=UTC),
        as_of=datetime.now(UTC) + timedelta(minutes=10),
        row_keys=("id",),
        output_table=target,
        model_name="risk",
        model_version="7",
        source_version=0,
        code_version=version("skyulf-core"),
        run_id="january",
        model_digest=bundle.semantic_digest,
        expected_target_version=0,
    )
    h = SimpleNamespace(
        spark=spark,
        source=source,
        target=target,
        bundle=bundle,
        spec=spec,
        lock=LocalTableLock(tmp_path / "locks"),
    )
    try:
        yield h
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {source}")
        spark.sql(f"DROP TABLE IF EXISTS {target}")


def _run(h, spec=None):
    """Exercise the public runner through a real Delta read and committed write."""
    return run_batch(
        h.spark,
        spec or h.spec,
        source=h.source,
        bundle=h.bundle,
        options=ExecutionOptions("spark"),
        admission=h.lock,
    )


@pytest.mark.parametrize("mode", ["native_features", "python_pipeline"])
def test_retry_preserves_other_periods_and_predictions(harness, mode):
    """Monthly retries neither duplicate rows nor overwrite December or February."""
    h = harness
    spec = replace(h.spec, mode=mode)
    first, second = _run(h, spec), _run(h, spec)
    rows = h.spark.table(h.target).orderBy("id").collect()
    assert [(r.id, round(r.prediction)) for r in rows] == [(1, 2), (2, 4), (9, 99)]
    assert first.input_count == first.output_count == 2
    assert first.commit_version == second.commit_version == 1
    assert second.replayed is True
    assert first.manifest["source_version"] == 0
    assert first.manifest["model_digest"] == h.bundle.semantic_digest


def test_delta_admission_publishes_and_replays_with_released_owner(harness):
    """Public batch publication and replay release shared admission and preserve prior rows."""
    from skyulf.integrations.databricks.delta_admission import DeltaTableAdmission

    h = harness
    control = "default.batch_admission_" + uuid4().hex
    target_id = h.spark.sql(f"DESCRIBE DETAIL {h.target}").first()["id"]
    prior = h.spark.table(h.target).first()
    h.spark.createDataFrame([(target_id, None)], "target_id string, owner string").write.format(
        "delta"
    ).saveAsTable(control)
    try:
        h.lock = DeltaTableAdmission(h.spark, control)
        first = _run(h)
        assert h.spark.table(control).first().owner is None
        rows = h.spark.table(h.target).orderBy("id").collect()
        assert [row.id for row in rows] == [1, 2, 9]
        assert [row.prediction for row in rows] == pytest.approx([2.0, 4.0, 99.0])
        assert rows[-1] == prior
        second = _run(h)
        assert h.spark.table(control).first().owner is None
        assert h.spark.table(h.target).orderBy("id").collect() == rows
        assert first.input_count == first.output_count == 2
        assert first.commit_version == second.commit_version == 1
        assert first.replayed is False
        assert second.replayed is True
        assert second.manifest == first.manifest
    finally:
        h.spark.sql(f"DROP TABLE IF EXISTS {control}").collect()


def test_stale_writer_and_reused_run_identity_fail(harness):
    """Stale expectations and changed requests cannot silently replace a committed month."""
    from skyulf.integrations.databricks.delta import BatchConflictError

    h = harness
    _run(h)
    with pytest.raises(BatchConflictError):
        _run(h, replace(h.spec, run_id="competing"))
    with pytest.raises(BatchConflictError):
        _run(h, replace(h.spec, model_version="8"))
    assert h.spark.table(h.target).count() == 3


def test_old_retry_cannot_undo_explicit_recomputation(harness):
    """A replay returns the original receipt even after a newer intentional run."""
    h = harness
    _run(h)
    newer = _run(h, replace(h.spec, run_id="recompute", expected_target_version=1))
    old = _run(h)
    assert newer.commit_version == 2
    assert old.commit_version == 1 and old.replayed
    assert h.spark.table(h.target).where("id = 1").first()["run_id"] == "recompute"


def test_empty_replacement_is_explicit_and_records_a_commit(harness):
    """Deleting an empty month requires opt-in and preserves a retry receipt in history."""
    h = harness
    _run(h)
    h.spark.table(h.source).where("id = 3").write.format("delta").mode("overwrite").saveAsTable(
        h.source
    )
    spec = replace(h.spec, run_id="delete-january", source_version=1, expected_target_version=1)
    with pytest.raises(ValueError, match="allow_empty"):
        _run(h, spec)
    result = _run(h, replace(spec, allow_empty=True))
    retry = _run(h, replace(spec, allow_empty=True))
    assert result.output_count == 0 and result.commit_version == retry.commit_version == 2
    assert [r.id for r in h.spark.table(h.target).collect()] == [9]


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"as_of": datetime(2020, 1, 1, tzinfo=UTC)}, "snapshot.*as_of"),
        ({"model_digest": "0" * 64}, "digest"),
        ({"code_version": "0.0.0"}, "code_version"),
    ],
)
def test_invalid_provenance_does_not_modify_target(harness, changes, match):
    """A recent snapshot, wrong model or wrong runtime cannot pretend to reproduce a run."""
    h = harness
    with pytest.raises(ValueError, match=match):
        _run(h, replace(h.spec, **changes))
    assert h.spark.table(h.target).count() == 1


def test_missing_admission_fails_before_any_write(harness):
    """A configured Delta engine alone cannot authorize competing publishers."""
    h = harness
    with pytest.raises(ValueError, match="admission"):
        run_batch(
            h.spark, h.spec, source=h.source, bundle=h.bundle, options=ExecutionOptions("spark")
        )
    assert h.spark.table(h.target).count() == 1


def test_source_version_remains_pinned_after_new_source_commit(harness):
    """A later source update must not change predictions for the selected snapshot."""
    h = harness
    h.spark.sql(f"UPDATE {h.source} SET x = 100.0")
    result = _run(h)
    rows = h.spark.table(h.target).where("id < 3").orderBy("id").collect()
    assert [round(row.prediction) for row in rows] == [2, 4]
    assert result.manifest["source_version"] == 0
    assert h.spark.sql(f"DESCRIBE HISTORY {h.source}").first().version == 1


def test_non_utc_session_preserves_period_instants(harness):
    """Neither selection nor Delta replacement can reinterpret UTC as local wall time."""
    h = harness
    before = h.spark.conf.get("spark.sql.session.timeZone")
    try:
        h.spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
        _run(h)
        result = _run(h, replace(h.spec, run_id="non-utc-recompute", expected_target_version=1))
        assert result.output_count == 2
        assert sorted(r.id for r in h.spark.table(h.target).collect()) == [1, 2, 9]
    finally:
        h.spark.conf.set("spark.sql.session.timeZone", before)


@pytest.mark.parametrize("violation", ["before", "end", "null", "metadata", "count"])
def test_sink_rejects_invalid_rows_without_modifying_target(harness, violation):
    """Invalid output must fail before the real Delta transaction changes any period."""
    from skyulf.integrations.databricks.delta import publish_replace_period

    functions = importlib.import_module("pyspark.sql.functions")
    h = harness
    instant = {
        "before": h.spec.period_start - timedelta(microseconds=1),
        "end": h.spec.period_end,
        "null": None,
    }.get(violation, h.spec.period_start)
    output = (
        h.spark.table(h.target)
        .withColumn("event_time", functions.lit(instant).cast("timestamp"))
        .withColumn("run_id", functions.lit(h.spec.run_id))
        .withColumn("model_version", functions.lit(h.spec.model_version))
    )
    if violation == "metadata":
        output = output.withColumn("model_name", functions.lit(None).cast("string"))
    manifest = {"output_count": 2 if violation == "count" else 1, "request_digest": "unused"}
    with pytest.raises(ValueError, match="Output"):
        publish_replace_period(h.spark, output, h.spec, manifest=manifest, admission=h.lock)
    assert h.spark.sql(f"DESCRIBE HISTORY {h.target}").first().version == 0
    assert [r.id for r in h.spark.table(h.target).collect()] == [9]


def test_held_admission_rejects_publish_without_modifying_target(harness):
    """A competing logical publisher cannot write while another owns the table lock."""
    from skyulf.integrations.databricks.admission import BatchConflictError
    from skyulf.integrations.databricks.delta import table_identity

    h = harness
    with (
        h.lock.hold(table_identity(h.spark, h.target)),
        pytest.raises(BatchConflictError, match="admission"),
    ):
        _run(h)
    assert h.spark.sql(f"DESCRIBE HISTORY {h.target}").first().version == 0


def test_admission_permission_failure_preserves_target(harness):
    """An authority denying admission must surface its error without publishing."""

    class DeniedAdmission:
        """Represent an external admission authority denying this caller."""

        local_only = False

        @contextmanager
        def hold(self, table_id):
            """Deny access before yielding publication authority."""
            raise PermissionError("publish access denied")
            yield  # pragma: no cover - context manager contract

    h = harness
    with pytest.raises(PermissionError, match="access denied"):
        run_batch(
            h.spark,
            h.spec,
            source=h.source,
            bundle=h.bundle,
            options=ExecutionOptions("spark"),
            admission=DeniedAdmission(),
        )
    assert h.spark.sql(f"DESCRIBE HISTORY {h.target}").first().version == 0


@pytest.mark.parametrize(
    "mode,condition_api",
    [("native_features", "getCondition"), ("python_pipeline", "getErrorClass")],
)
def test_serverless_cache_rejection_publishes_and_replays(
    harness, monkeypatch, mode, condition_api
):
    """Only structured serverless cache rejection may use distributed uncached publication."""
    h = harness
    error = RuntimeError("cache unsupported")
    monkeypatch.setattr(
        error, condition_api, lambda: "NOT_SUPPORTED_WITH_SERVERLESS", raising=False
    )

    def reject_persist(frame, *args, **kwargs):
        """Represent the rejected cache API while keeping every Spark/Delta action real."""
        raise error

    def reject_unpersist(frame, *args, **kwargs):
        """Unpersist is also unsupported and must not follow a failed persist."""
        raise AssertionError("Unpersist called after rejected persistence.")

    frame_type = type(h.spark.table(h.source))
    monkeypatch.setattr(frame_type, "persist", reject_persist)
    monkeypatch.setattr(frame_type, "unpersist", reject_unpersist)
    spec = replace(h.spec, mode=mode)
    first, replay = _run(h, spec), _run(h, spec)
    rows = h.spark.table(h.target).orderBy("id").collect()
    assert [row.id for row in rows] == [1, 2, 9]
    assert [row.prediction for row in rows] == pytest.approx([2.0, 4.0, 99.0])
    assert rows[-1]["run_id"] == "prior"
    assert first.input_count == first.output_count == 2
    assert first.commit_version == replay.commit_version == 1
    assert first.replayed is False
    assert replay.replayed is True
    assert first.manifest == replay.manifest


@pytest.mark.parametrize("structured", [False, True])
def test_other_cache_errors_propagate_before_target_mutation(harness, monkeypatch, structured):
    """Neither message text nor unrelated structured failures may bypass cache errors."""
    h = harness
    error = RuntimeError("NOT_SUPPORTED_WITH_SERVERLESS")
    if structured:
        monkeypatch.setattr(error, "getCondition", lambda: "PERMISSION_DENIED", raising=False)

    def reject_persist(frame, *args, **kwargs):
        """Fail only at the real prediction frame's optional cache boundary."""
        raise error

    monkeypatch.setattr(type(h.spark.table(h.source)), "persist", reject_persist)
    before = h.spark.table(h.target).collect()
    with pytest.raises(RuntimeError) as caught:
        _run(h)
    assert caught.value is error
    assert h.spark.table(h.target).collect() == before
    assert h.spark.sql(f"DESCRIBE HISTORY {h.target}").first().version == 0


def test_classic_cache_is_released_after_success_and_conflict(harness, monkeypatch):
    """Classic runtimes retain caching but release real persisted frames on every exit."""
    from skyulf.integrations.databricks.admission import BatchConflictError

    h = harness
    frame_type = type(h.spark.table(h.source))
    persist = frame_type.persist
    persisted = []

    def retain_persisted_frame(frame, *args, **kwargs):
        """Observe actual cache state without replacing Spark's persistence behavior."""
        result = persist(frame, *args, **kwargs)
        assert result.storageLevel.useMemory
        persisted.append(result)
        return result

    monkeypatch.setattr(frame_type, "persist", retain_persisted_frame)
    _run(h)
    with pytest.raises(BatchConflictError):
        _run(h, replace(h.spec, run_id="stale-cached-run"))
    assert len(persisted) == 2
    assert all(not frame.storageLevel.useMemory for frame in persisted)
