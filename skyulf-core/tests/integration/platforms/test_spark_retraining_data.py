"""Executor metadata splitting preserves local training eligibility contracts."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from skyulf.integrations.databricks.data.training.spark_retraining_data import (
    _metadata_train_ordinals,
    _pandas_source_types,
    _require_supported_recipe,
    _row_hash,
)
from skyulf.integrations.databricks.training.fitting.candidate import (
    TrainingSpec,
    partition_training_rows,
)


def _spec(**changes):
    """Keep split fixtures independent of cloud configuration and model storage."""
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


@pytest.mark.parametrize("stratify", [False, True])
def test_exact_split_ordinals(stratify):
    """Random metadata splitting must match sklearn membership rather than Spark hashes."""
    spec = _spec(stratify=stratify)
    frame = pd.DataFrame({"_ordinal": range(20), "y": [0, 1] * 10})
    train, _ = partition_training_rows(frame, spec)
    assert _metadata_train_ordinals(frame, spec) == train["_ordinal"].tolist()


def test_group_split_ordinals():
    """Whole group isolation must retain the trainer's exact selected groups."""
    spec = _spec(group_column="group")
    frame = pd.DataFrame(
        {"_ordinal": range(20), "y": [0, 1] * 10, "group": [i // 2 for i in range(20)]}
    )
    train, _ = partition_training_rows(frame, spec)
    assert _metadata_train_ordinals(frame, spec) == train["_ordinal"].tolist()


def test_small_split_fails_closed():
    """An ineligible fit population cannot become a ready freshness request."""
    with pytest.raises(ValueError, match="at least two"):
        _metadata_train_ordinals(pd.DataFrame({"_ordinal": range(3), "y": range(3)}), _spec())


def test_dtype_widening_preserves_hash():
    """Pandas numeric widening cannot manufacture changed feature-target rows."""
    assert _row_hash([1, None, 2]) == _row_hash([1.0, float("nan"), 2.0])


def test_nullable_integer_source_inference(monkeypatch):
    """Native casting must reproduce pandas rounding before rows are filtered or split."""
    from skyulf.integrations.databricks.data.training import spark_retraining_data

    fields = [
        SimpleNamespace(name=name, dataType=SimpleNamespace(typeName=lambda: "long"))
        for name in ("id", "x", "y")
    ]
    source = MagicMock()
    source.schema.fields = fields
    source.columns = [field.name for field in fields]
    source.agg.return_value.first.return_value = {"id": 0, "x": 1, "y": 0}
    functions = MagicMock()
    columns = {name: MagicMock() for name in source.columns}
    functions.col.side_effect = columns.__getitem__
    monkeypatch.setattr(spark_retraining_data, "_functions", lambda: functions)
    _pandas_source_types(source, _spec())
    value = 9007199254740993
    local = pd.DataFrame.from_records([{"x": value}, {"x": None}])
    assert _row_hash([value]) != _row_hash([local.iloc[0, 0]])
    assert _row_hash([float(value)]) == _row_hash([local.iloc[0, 0]])
    columns["x"].cast.assert_called_once_with("double")
    columns["id"].cast.assert_not_called()


def test_historical_transport_does_not_reapply_string_parsing(monkeypatch):
    """Prepared timestamps already own UTC normalization regardless of raw source string rules."""
    from skyulf.integrations.databricks.data.training import spark_retraining_data
    from skyulf.integrations.databricks.data.training.training_dates import TrainingDateSpec

    spec = _spec(
        weight_column="weight",
        event_column="event",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        cutoff=datetime(2026, 1, 21, tzinfo=UTC),
        event_time_parsing=TrainingDateSpec(format="%Y-%m-%d %H:%M:%S", timezone="UTC"),
        pre_split_steps=(
            {
                "name": "weight",
                "transformer": "ManualBounds",
                "params": {"bounds": {"weight": {"lower": 1}}},
            },
        ),
    )
    frame = MagicMock()
    normalizer = MagicMock(return_value=frame)
    monkeypatch.setattr(spark_retraining_data, "_functions", MagicMock)
    monkeypatch.setattr(spark_retraining_data, "_counts", lambda *args: frame)
    monkeypatch.setattr(spark_retraining_data, "read_reference_population", lambda *args: frame)
    monkeypatch.setattr(spark_retraining_data, "normalize_training_dates", normalizer)
    monkeypatch.setattr(spark_retraining_data, "_pandas_source_types", lambda *args: frame)
    monkeypatch.setattr(spark_retraining_data, "_overlay_weights", lambda *args: frame)
    monkeypatch.setattr(spark_retraining_data, "_eligible", lambda *args: frame)
    evidence = {"prepared_reference": {"seen": {}, "source": {}}}
    spark_retraining_data._baseline(None, evidence, spec, spec, frame, ["x", "y"])
    assert normalizer.call_args.kwargs["event_spec"] == TrainingDateSpec()
    assert normalizer.call_args.kwargs["result_spec"] == TrainingDateSpec()


def test_assessment_does_not_require_serverless_unsupported_persistence(monkeypatch):
    """Freshness must run on the same serverless compute as the monitoring job."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from skyulf.integrations.databricks.data.training import spark_retraining_data as module

    spec = _spec()
    evidence = {"prepared_reference_source_table_id": "identity", "model_version": "1"}
    artifact = SimpleNamespace(manifest=SimpleNamespace(fitted_engine="pandas"))
    counts = Mock()
    counts.cache.side_effect = ValueError("DataFrame.cache unsupported on serverless")
    monkeypatch.setattr(
        module,
        "load_spark_monitoring_reference",
        Mock(return_value=(artifact, spec, None, evidence)),
    )
    monkeypatch.setattr(module, "table_identity", Mock(return_value="identity"))
    monkeypatch.setattr(module, "resolve_training_spec", Mock(return_value=spec))
    monkeypatch.setattr(module, "phase_training_spec", Mock(return_value=spec))
    for name in ("_read_source", "_eligible", "_training_partition", "_baseline"):
        monkeypatch.setattr(module, name, Mock(return_value=object()))
    monkeypatch.setattr(module, "_counts", Mock(return_value=counts))
    monkeypatch.setattr(
        module, "_assessment", Mock(return_value={"status": "ready", "changed_rows": 1})
    )
    result = module.assess_spark_training_data(
        None,
        None,
        {"engine": "pandas", "training_table": spec.table},
        datetime(2026, 10, 5, tzinfo=UTC),
    )
    assert result["status"] == "ready"
    counts.cache.assert_not_called()
    counts.unpersist.assert_not_called()


def test_fixed_edits_fail_explicitly():
    """Unproved normalization parity must never authorize a training request."""
    spec = _spec(
        pre_split_steps=(
            {
                "name": "cast",
                "transformer": "Casting",
                "params": {"columns": ["x"], "target_type": "float"},
            },
        )
    )
    with pytest.raises(ValueError, match="Unsupported Spark freshness pre_split"):
        _require_supported_recipe(spec)


def test_missing_target_is_not_eligible():
    """Unfiltered missing targets cannot be admitted by a random metadata split."""
    frame = pd.DataFrame({"_ordinal": range(20), "y": [None] + [1] * 19})
    with pytest.raises(ValueError, match="nonnull targets"):
        _metadata_train_ordinals(frame, _spec())


def test_training_weight_validation_is_preserved():
    """All-zero fitting weights cannot produce an eligible retraining request."""
    frame = pd.DataFrame({"_ordinal": range(20), "y": [0, 1] * 10, "weight": [0.0] * 20})
    with pytest.raises(ValueError, match="positive"):
        _metadata_train_ordinals(frame, _spec(weight_column="weight"))


def test_temporal_metadata_split_and_group_overlap():
    """Advancing the event boundary admits old holdout, while grouped overlap stays invalid."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    spec = _spec(
        split_strategy="temporal",
        test_size=None,
        random_state=None,
        stratify=None,
        event_column="event",
        start=start,
        holdout_start=start + timedelta(days=10),
        cutoff=start + timedelta(days=20),
    )
    frame = pd.DataFrame(
        {
            "_ordinal": range(20),
            "y": [0, 1] * 10,
            "event": [int((start + timedelta(days=i)).timestamp() * 1000000) for i in range(20)],
        }
    )
    assert _metadata_train_ordinals(frame, spec) == list(range(10))
    assert _metadata_train_ordinals(
        frame, replace(spec, holdout_start=start + timedelta(days=15))
    ) == list(range(15))
    frame["group"] = 1
    with pytest.raises(ValueError, match="disjoint groups"):
        _metadata_train_ordinals(frame, replace(spec, group_column="group"))
