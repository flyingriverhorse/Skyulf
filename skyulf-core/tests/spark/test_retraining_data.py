"""Distributed freshness parity over actual Spark execution and metadata splits."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from skyulf.integrations.databricks.data.training.retraining_data import row_identity_counts
from skyulf.integrations.databricks.data.training.spark_retraining_data import (
    _assessment,
    _counts,
    _eligible,
    _overlay_weights,
    _pandas_source_types,
    _training_partition,
)
from skyulf.integrations.databricks.training.fitting.candidate import (
    TrainingSpec,
    split_labeled_snapshot,
)


def _spec(**changes):
    """Give parity cases the same explicit split and safety policy as local training."""
    return replace(
        TrainingSpec(
            table="a.b.c",
            version=1,
            record_key_columns=("id",),
            input_columns=("x",),
            target_column="y",
            max_rows=100,
            max_bytes=100000,
        ),
        **changes,
    )


@pytest.mark.parametrize("partitions", [1, 3])
@pytest.mark.parametrize("stratify", [False, True])
def test_split_digest_matches_local(spark, partitions, stratify):
    """Executor splitting and streamed digests preserve exact local fit values across shuffles."""
    spec = _spec(stratify=stratify)
    source = pd.DataFrame({"id": range(20), "x": range(20), "y": [0, 1] * 10})
    local, _, _ = split_labeled_snapshot(source, spec)
    frame = spark.createDataFrame(source).repartition(partitions)
    counts = _counts(_training_partition(frame, spec), ["x", "y"])
    result = _assessment(counts, _counts(frame, ["x", "y"]), ["x", "y"])
    payload = json.dumps(
        {"columns": ["x", "y"], "rows": sorted(row_identity_counts(local, ["x", "y"]).items())},
        separators=(",", ":"),
    )
    assert result == {
        "status": "no_new_training_data",
        "training_rows": 16,
        "changed_rows": 0,
        "content_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def test_duplicate_multiset_and_dtype_widening(spark):
    """Count extra duplicate values while treating integer and floating representations equally."""
    old = spark.createDataFrame([(1, 0), (2, 1)], "x long, y long")
    current = spark.createDataFrame([(1.0, 0.0), (1.0, 0.0), (2.0, 1.0)], "x double, y double")
    result = _assessment(_counts(current, ["x", "y"]), _counts(old, ["x", "y"]), ["x", "y"])
    assert result["changed_rows"] == 1
    assert result["training_rows"] == 3


def test_weight_only_eligibility_counterfactual(spark):
    """Historical values admitted only by changed weights are protected by maximum counts."""
    spec = _spec(
        weight_column="weight",
        pre_split_steps=(
            {
                "name": "weighted_filter",
                "transformer": "ManualBounds",
                "params": {"bounds": {"weight": {"lower": 1}}},
            },
        ),
    )
    old = spark.createDataFrame(
        [(i, i, i % 2, 0.0 if i < 10 else 1.0) for i in range(20)],
        "id long, x long, y long, weight double",
    )
    current = spark.createDataFrame([(i, i, i % 2, 1.0) for i in range(20)], old.schema)
    counterfactual = _eligible(_overlay_weights(old, current, ("id",), {"weight"}), spec)
    result = _assessment(
        _counts(_training_partition(_eligible(current, spec), spec), ["x", "y"]),
        _counts(counterfactual, ["x", "y"]),
        ["x", "y"],
    )
    assert result["changed_rows"] == 0


def test_missing_target_filter_and_dedup_conflict(spark):
    """Missing labels require explicit exclusion and contradictory dedup targets fail closed."""
    spec = _spec(
        pre_split_steps=(
            {"name": "missing", "transformer": "DropMissingRows", "params": {"subset": ["y"]}},
        )
    )
    frame = spark.createDataFrame([(1, 1, None), (2, 1, 0), (3, 1, 1)], "id long, x long, y long")
    eligible = _eligible(frame, spec)
    assert eligible.count() == 2
    conflict = replace(
        spec,
        pre_split_steps=(
            {"name": "dedup", "transformer": "Deduplicate", "params": {"subset": ["x"]}},
        ),
    )
    with pytest.raises(ValueError, match="conflicting target"):
        _eligible(eligible, conflict)


def test_temporal_aging_excludes_future_labels(spark):
    """Only mature labels inside the advanced training boundary count as fresh."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    spec = _spec(
        split_strategy="temporal",
        test_size=None,
        random_state=None,
        stratify=None,
        event_column="event",
        start=start,
        holdout_start=start + timedelta(days=15),
        cutoff=start + timedelta(days=20),
        filter_unavailable_results=True,
        result_available_at_column="available",
        result_cutoff=start + timedelta(days=20),
    )
    base = int(start.timestamp() * 1000000)
    day = 86400 * 1000000
    rows = [(i, i, i % 2, base + i * day, base + (30 if i == 12 else i) * day) for i in range(20)]
    frame = spark.createDataFrame(rows, "id long, x long, y long, event long, available long")
    train = _training_partition(_eligible(frame, spec), spec)
    old = frame.where("id < 10")
    result = _assessment(_counts(train, ["x", "y"]), _counts(old, ["x", "y"]), ["x", "y"])
    assert result["changed_rows"] == 4
    assert result["training_rows"] == 14


def test_nullable_large_integer_matches_materialized_source(spark):
    """An unchanged nullable LONG must retain the trainer's column-wide float64 rounding."""
    spec = _spec()
    records = [{"id": i, "x": None if i == 0 else 9007199254740993, "y": i % 2} for i in range(20)]
    local_source = pd.DataFrame.from_records(records)
    local_train, _, _ = split_labeled_snapshot(local_source, spec)
    frame = spark.createDataFrame(records, "id long, x long, y long").repartition(3)
    frame = _pandas_source_types(frame, spec)
    train = _training_partition(_eligible(frame, spec), spec)
    result = _assessment(
        _counts(train, ["x", "y"]),
        _counts(spark.createDataFrame(local_source), ["x", "y"]),
        ["x", "y"],
    )
    payload = json.dumps(
        {
            "columns": ["x", "y"],
            "rows": sorted(row_identity_counts(local_train, ["x", "y"]).items()),
        },
        separators=(",", ":"),
    )
    assert result["changed_rows"] == 0
    assert result["content_sha256"] == hashlib.sha256(payload.encode()).hexdigest()
    assert frame.schema["x"].dataType.typeName() == "double"
    assert frame.schema["id"].dataType.typeName() == "long"


def test_prepared_temporal_source_uses_normalized_timestamp_rules(spark, monkeypatch):
    """Weight replay must not apply original string parsing to cached UTC timestamps."""
    from skyulf.integrations.databricks.data.training import spark_retraining_data
    from skyulf.integrations.databricks.data.training.training_dates import TrainingDateSpec

    start = datetime(2026, 1, 1, tzinfo=UTC)
    spec = _spec(
        split_strategy="temporal",
        test_size=None,
        random_state=None,
        stratify=None,
        event_column="event",
        start=start,
        holdout_start=start + timedelta(days=15),
        cutoff=start + timedelta(days=20),
        weight_column="weight",
        event_time_parsing=TrainingDateSpec(format="%Y-%m-%d %H:%M:%S", timezone="UTC"),
        pre_split_steps=(
            {
                "name": "weight",
                "transformer": "ManualBounds",
                "params": {"bounds": {"weight": {"lower": 1}}},
            },
        ),
    )
    historical = spark.createDataFrame(
        [(i, i, i % 2, start + timedelta(days=i), 1.0) for i in range(20)],
        "id long, x long, y long, event timestamp, weight double",
    )
    seen = historical.where("id < 15").select("x", "y")
    frames = {"source": historical, "seen": seen}
    monkeypatch.setattr(
        spark_retraining_data, "read_reference_population", lambda session, record: frames[record]
    )
    evidence = {"prepared_reference": {"source": "source", "seen": "seen"}}
    baseline = spark_retraining_data._baseline(spark, evidence, spec, spec, historical, ["x", "y"])
    result = _assessment(_counts(seen, ["x", "y"]), baseline, ["x", "y"])
    assert result["changed_rows"] == 0
    assert result["training_rows"] == 15
