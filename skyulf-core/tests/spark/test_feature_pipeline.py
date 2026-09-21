"""Portable native FE must retain keyed predictions through engine and partition changes."""

import json
import math

import pandas as pd
import polars as pl
import pytest

from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.preprocessing.pipeline import FeatureEngineer


def test_compact_unicode_pipeline_runs_with_received_budget(spark):
    """Spark apply must not expand Unicode escaping and reject an already bounded pipeline."""
    column = "ölçüm" * 50
    fe = FeatureEngineer(
        [{"name": "fill", "transformer": "SimpleImputer", "params": {"columns": [column]}}]
    )
    fe.fit_transform(pd.DataFrame({column: [1.0, 3.0]}))
    payload = json.dumps(
        json.loads(fe.export_state()), ensure_ascii=False, separators=(",", ":")
    ).encode()
    restored = FeatureEngineer.from_state(
        payload,
        frame_spec=FrameSpec(("id",)),
        execution_options=ExecutionOptions("spark", state_max_bytes=len(payload)),
    )
    frame = spark.createDataFrame([(1, float("nan"))], ["id", column])
    assert restored.transform(frame).first()[column] == 2.0


@pytest.fixture(scope="module", params=["pandas", "polars", "spark"])
def fitted_payload(request, spark):
    """Fit once per source engine so every destination sees the same training-only state."""
    local = pd.DataFrame({"id": [1, 2, 3], "x": [1.0, float("nan"), 3.0], "label": [0, 1, 0]})
    data = local if request.param == "pandas" else pl.from_pandas(local)
    if request.param == "spark":
        data = spark.createDataFrame(
            [(1, 1.0, 0), (2, None, 1), (3, 3.0, 0)], "id long, x double, label long"
        )
    fe = FeatureEngineer(
        [
            {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}},
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
        ],
        frame_spec=FrameSpec(("id",), "label") if request.param == "spark" else None,
        execution_options=ExecutionOptions("spark") if request.param == "spark" else None,
    )
    fe.fit_transform(data)
    return fe.export_state()


@pytest.mark.parametrize("apply_engine", ["pandas", "polars", "spark"])
def test_full_chain_cross_engine_round_trip(spark, fitted_payload, apply_engine, tmp_path):
    """Byte persistence must preserve train-only imputation then scaling across all engines."""
    path = tmp_path / "features.json"
    path.write_bytes(fitted_payload)
    fe = FeatureEngineer.from_state(
        path.read_bytes(),
        frame_spec=FrameSpec(("key",)) if apply_engine == "spark" else None,
        execution_options=ExecutionOptions(apply_engine),
    )
    local = pd.DataFrame({"key": [20, 10, 30], "x": [3.0, 1.0, float("nan")]})
    data = {
        "pandas": lambda: local,
        "polars": lambda: pl.from_pandas(local),
        "spark": lambda: spark.createDataFrame(
            [(20, 3.0), (10, 1.0), (30, None)], "key long, x double"
        ).repartition(2),
    }[apply_engine]()
    output = fe.transform(data)
    rows = {
        "pandas": lambda: output.to_dict("records"),
        "polars": lambda: output.to_dicts(),
        "spark": lambda: [r.asDict() for r in output.collect()],
    }[apply_engine]()
    assert {row["key"]: row["x"] for row in rows} == pytest.approx(
        {10: -math.sqrt(1.5), 20: math.sqrt(1.5), 30: 0.0}
    )
    assert fe.export_state() == fitted_payload


@pytest.mark.parametrize("partitions", [1, 2, 7])
def test_repartition_reversal_and_repeated_transform(spark, fitted_payload, partitions):
    """Rows are identified by keys and repeated apply must leave learned state untouched."""
    fe = FeatureEngineer.from_state(
        fitted_payload, frame_spec=FrameSpec(("id",)), execution_options=ExecutionOptions("spark")
    )
    data = spark.createDataFrame([(1, 1.0), (2, 3.0), (3, None)], "id long, x double")
    expected = {1: -math.sqrt(1.5), 2: math.sqrt(1.5), 3: 0.0}
    for frame in (
        data.repartition(partitions),
        data.orderBy("id", ascending=False).repartition(partitions),
    ):
        actual = {row.id: row.x for row in fe.transform(frame).collect()}
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)
    assert fe.export_state() == fitted_payload


def test_spark_failed_refit_preserves_exported_pipeline(spark, fitted_payload):
    """The last successful Spark fit must remain exportable after a failed refit."""
    fe = FeatureEngineer.from_state(
        fitted_payload, frame_spec=FrameSpec(("id",)), execution_options=ExecutionOptions("spark")
    )
    duplicate = spark.createDataFrame([(1, 1.0), (1, 3.0)], "id long, x double")
    with pytest.raises(ValueError, match="unique"):
        fe.fit_transform(duplicate)
    assert fe.export_state() == fitted_payload


def test_large_pipeline_forbids_unbounded_collection(spark, monkeypatch):
    """Only bounded aggregate or validation rows may cross the driver boundary."""
    data = spark.range(10000).selectExpr(
        "id", "CASE WHEN id % 3 = 0 THEN NULL ELSE CAST(id AS DOUBLE) END AS x"
    )
    original_collect = type(data).collect
    original_first = type(data).first
    bounded_depth = 0
    returned_sizes = []

    def bounded_first(frame):
        """The node's first() boundary limits aggregate results to one row."""
        nonlocal bounded_depth
        bounded_depth += 1
        try:
            return original_first(frame)
        finally:
            bounded_depth -= 1

    def guarded_collect(frame):
        """Permit only explicit one-row validation limits or the bounded aggregate path."""
        plan = frame._jdf.queryExecution().logical().toString()
        assert bounded_depth or plan.startswith("GlobalLimit 1"), "Unbounded input collection"
        rows = original_collect(frame)
        returned_sizes.append(len(rows))
        assert len(rows) <= 1
        return rows

    def forbidden(*args, **kwargs):
        """Distributed frames cannot enter local preprocessing during this gate."""
        pytest.fail("Local conversion in native pipeline")

    with monkeypatch.context() as patch:
        patch.setattr(type(data), "collect", guarded_collect)
        patch.setattr(type(data), "first", bounded_first)
        for name in ("toPandas", "toArrow", "toLocalIterator"):
            patch.setattr(type(data), name, forbidden)
        fe = FeatureEngineer(
            [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}},
                {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
            ],
            frame_spec=FrameSpec(("id",)),
            execution_options=ExecutionOptions("spark"),
        )
        trained, _ = fe.fit_transform(data)
        restored = FeatureEngineer.from_state(
            fe.export_state(),
            frame_spec=FrameSpec(("id",)),
            execution_options=ExecutionOptions("spark"),
        )
        output = restored.transform(data.repartition(7))
    # Explicit terminal aggregate remains distributed; no training samples are collected.
    stats = output.selectExpr(
        "count(*) AS rows", "avg(x) AS mean", "var_pop(x) AS variance"
    ).first()
    assert stats.rows == 10000
    assert stats.mean == pytest.approx(0.0, abs=1e-12)
    assert stats.variance == pytest.approx(1.0, rel=1e-10, abs=1e-12)
    assert returned_sizes and trained.columns == ["id", "x"]
