"""Fail early on ambiguous batch identity and temporal contracts."""

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

from skyulf.core.execution import ExecutionOptions
from skyulf.integrations.databricks.batch import BatchSpec, run_batch


@pytest.fixture
def spec():
    """Declare concrete identities; dates are independent of scheduler time."""
    return BatchSpec(
        period_start=datetime(2026, 1, 1, tzinfo=UTC),
        period_end=datetime(2026, 2, 1, tzinfo=UTC),
        as_of=datetime(2026, 2, 2, tzinfo=UTC),
        row_keys=("id",),
        output_table="default.predictions",
        model_name="risk",
        model_version="7",
        source_version=0,
        code_version="0.9.0",
        run_id="january-v7",
        model_digest="a" * 64,
        expected_target_version=0,
    )


@pytest.mark.parametrize("field", ["period_start", "period_end", "as_of"])
def test_naive_dates_fail(spec, field):
    """Missing timezone information must not inherit the driver's timezone."""
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(spec, **{field: datetime(2026, 1, 1)})


@pytest.mark.parametrize(
    "changes",
    [
        {"period_end": datetime(2025, 12, 1, tzinfo=UTC)},
        {"row_keys": ("id", "id")},
        {"row_keys": ("__skyulf_run_id",)},
        {"row_keys": ("run_id",)},
        {"period_column": "model_name"},
        {"output_table": "default.x; DROP TABLE x"},
        {"model_version": "champion"},
        {"model_version": "-1"},
        {"source_version": True},
        {"source_version": -1},
        {"expected_target_version": -1},
        {"model_digest": "bad"},
        {"publish_mode": "append"},
        {"mode": "streaming"},
        {"run_id": ""},
        {"business_timezone": "missing/zone"},
        {"allow_empty": 1},
    ],
)
def test_invalid_contracts_fail_before_runtime(spec, changes):
    """Invalid versions, names or modes cannot reach a Spark action or sink."""
    with pytest.raises((ValueError, TypeError)):
        replace(spec, **changes)


def test_business_month_preserves_dst_instants(spec):
    """UTC bounds preserve local calendar boundaries across a DST transition."""
    zone = ZoneInfo("Europe/Vilnius")
    actual = replace(
        spec,
        period_start=datetime(2026, 3, 1, tzinfo=zone),
        period_end=datetime(2026, 4, 1, tzinfo=zone),
        business_timezone="Europe/Vilnius",
    )
    assert actual.period_start_utc == datetime(2026, 2, 28, 22, tzinfo=UTC)
    assert actual.period_end_utc == datetime(2026, 3, 31, 21, tzinfo=UTC)


def test_local_engine_rejected_before_reading_source(spec):
    """A batch must never collect a Spark source into a local engine implicitly."""
    with pytest.raises(ValueError, match="engine='spark'"):
        run_batch(
            None,
            spec,
            source="default.source",
            bundle=cast(Any, object()),
            options=ExecutionOptions("pandas"),
        )


@pytest.mark.parametrize("master", ["local-cluster[2,1,1024]", "spark://cluster:7077"])
def test_local_admission_cannot_protect_distributed_drivers(spec, tmp_path, master):
    """A driver's local lock file must never imply distributed writer exclusion."""
    from skyulf.integrations.databricks.admission import LocalTableLock

    spark = SimpleNamespace(sparkContext=SimpleNamespace(master=master))
    with pytest.raises(ValueError, match="distributed drivers"):
        run_batch(
            spark,
            spec,
            source="default.source",
            bundle=cast(Any, object()),
            options=ExecutionOptions("spark"),
            admission=LocalTableLock(tmp_path),
        )


def test_nonexistent_local_boundary_is_rejected(spec):
    """A DST gap cannot silently shift the intended calendar period."""
    with pytest.raises(ValueError, match="valid local instant"):
        replace(spec, period_start=datetime(2026, 3, 29, 3, 30, tzinfo=ZoneInfo("Europe/Vilnius")))


def test_direct_sink_rejects_local_admission_on_distributed_runtime(spec, tmp_path):
    """The lower-level writer must enforce the same admission boundary as the runner."""
    from skyulf.integrations.databricks.admission import LocalTableLock
    from skyulf.integrations.databricks.delta import publish_replace_period

    spark = SimpleNamespace(sparkContext=SimpleNamespace(master="spark://cluster:7077"))
    with pytest.raises(ValueError, match="distributed drivers"):
        publish_replace_period(
            spark, object(), spec, manifest={}, admission=LocalTableLock(tmp_path)
        )


def test_local_publication_mode_is_distinct_from_spark_inference(spec) -> None:
    """Local model output must declare its actual execution mode."""
    local = replace(spec, mode="local_pipeline")
    assert local.mode == "local_pipeline"
    with pytest.raises(ValueError, match="Spark inference mode"):
        run_batch(
            None,
            local,
            source="default.source",
            bundle=cast(Any, object()),
            options=ExecutionOptions("spark"),
        )
