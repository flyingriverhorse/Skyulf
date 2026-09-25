"""Bounded local-engine training and monthly scoring on pinned UC snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline
from skyulf.integrations.databricks import local_batch
from skyulf.integrations.databricks.local_batch import (
    LocalSourceSpec,
    fit_local_workflow,
    read_local_source,
    score_local_source,
)
from skyulf.integrations.databricks.local_sdk import (
    InputSource,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    prepare_local_workflow,
)


class _Rows:
    """Record Spark operations before yielding a bounded set of fake rows."""

    def __init__(self, rows, operations):
        """Retain the source rows and shared operation log."""
        self.rows = rows
        self.operations = operations

    def where(self, expression):
        """Record period filtering ahead of driver transfer."""
        self.operations.append(("where", expression))
        return self

    def select(self, *columns):
        """Record the narrow projection ahead of driver transfer."""
        self.operations.append(("select", columns))
        return self

    def orderBy(self, *columns):
        """Record deterministic key order ahead of driver transfer."""
        self.operations.append(("order", columns))
        return self

    def limit(self, count):
        """Record the finite cap before iteration."""
        self.operations.append(("limit", count))
        return self

    def toLocalIterator(self):
        """Yield only after snapshot, filter, projection and limit."""
        self.operations.append(("iterator", None))
        return iter(self.rows)


class _Reader:
    """Stand in for the versioned Delta reader without cloud access."""

    def __init__(self, rows, operations):
        """Retain fake rows and operations."""
        self.rows = rows
        self.operations = operations

    def format(self, value):
        """Require Delta format."""
        self.operations.append(("format", value))
        return self

    def option(self, key, value):
        """Require a concrete version before table access."""
        self.operations.append(("option", key, value))
        return self

    def table(self, name):
        """Return the fake source frame."""
        self.operations.append(("table", name))
        return _Rows(self.rows, self.operations)


class _Spark:
    """Expose only the versioned reader used by this workflow."""

    def __init__(self, rows, operations):
        """Construct the fake reader."""
        self.read = _Reader(rows, operations)


def _spec(**changes):
    """Keep snapshot and period boundaries explicit in every test."""
    values: dict[str, Any] = {
        "table": "workspace.test.score_source",
        "version": 2,
        "period_start": datetime(2026, 1, 1, tzinfo=UTC),
        "period_end": datetime(2026, 2, 1, tzinfo=UTC),
        "record_key_columns": ("entity_id",),
        "input_columns": ("x",),
        "max_rows": 3,
        "max_bytes": 1024,
    }
    values.update(changes)
    return LocalSourceSpec(**values)


def test_reader_pins_filters_and_limits_before_local_iteration() -> None:
    """A monthly request must never pull an unrestricted UC table to the driver."""
    operations = []
    frame = read_local_source(_Spark([{"entity_id": "b", "x": 2.0}], operations), _spec())
    assert frame.to_dict("records") == [{"entity_id": "b", "x": 2.0}]
    assert operations[:3] == [
        ("format", "delta"),
        ("option", "versionAsOf", 2),
        ("table", "workspace.test.score_source"),
    ]
    assert [name for name, *_ in operations[3:]] == [
        "where",
        "select",
        "order",
        "limit",
        "iterator",
    ]
    assert ("limit", 4) in operations


def test_reader_rejects_row_and_byte_overruns() -> None:
    """The requested cap must fail closed even when a source yields excess rows."""
    rows = [{"entity_id": str(i), "x": float(i)} for i in range(4)]
    with pytest.raises(ValueError, match="max_rows"):
        read_local_source(_Spark(rows, []), _spec())
    with pytest.raises(ValueError, match="max_bytes"):
        read_local_source(_Spark(rows[:1], []), _spec(max_bytes=1))


def test_fit_and_score_local_workflow_keeps_keys_outside_model(tmp_path) -> None:
    """A loaded fitted pipeline must score the new table without labels or refitting."""
    train = pd.DataFrame({"x": np.arange(8, dtype="float64"), "target": np.arange(8) * 2.0})
    path = tmp_path / "fitted"
    fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        SplitDataset(train=train.iloc[:6], test=train.iloc[6:]),
        target_column="target",
        artifact_path=path,
        max_rows=8,
        max_bytes=4096,
    )
    artifact = load_local_pipeline(path)
    config = LocalWorkflowConfig(
        runtime="databricks",
        engine="pandas",
        source=InputSource(kind="uc_table", table="workspace.test.score_source", version=2),
        model=ModelSelection(kind="local_pipeline", path=str(path)),
        sink=OutputSink(kind="return_frame"),
    )
    prepared = prepare_local_workflow(config)
    rows = [{"entity_id": "new-1", "x": 2.0}, {"entity_id": "new-2", "x": 4.0}]
    scored = score_local_source(_Spark(rows, []), _spec(), prepared)
    assert artifact.manifest.input_columns == ("x",)
    assert scored.predictions["entity_id"].tolist() == ["new-1", "new-2"]
    np.testing.assert_allclose(scored.predictions["prediction"], [4.0, 8.0])
    assert scored.diagnostics["source_version"] == 2


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_heldout_regression_metrics_use_saved_pipeline_and_200_test_rows(tmp_path, engine) -> None:
    """A fitted model must be measured on the reserved labels, not its training rows."""
    x = np.arange(1000, dtype="float64")
    frame = pd.DataFrame({"x": x, "target": 2 * x + 1})
    native = frame if engine == "pandas" else pl.from_pandas(frame)
    artifact = fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        SplitDataset(train=native[:800], test=native[800:]),
        target_column="target",
        artifact_path=tmp_path / engine,
        max_rows=1000,
        max_bytes=100_000,
    )

    metrics = local_batch.evaluate_local_holdout(artifact, native[800:], target_column="target")

    assert set(metrics) == {
        "heldout_mae",
        "heldout_mse",
        "heldout_rmse",
        "heldout_r2",
        "heldout_mape",
        "heldout_explained_variance",
    }
    assert metrics["heldout_mae"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["heldout_rmse"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["heldout_r2"] == pytest.approx(1.0)
    assert metrics["heldout_mse"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["heldout_mape"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["heldout_explained_variance"] == pytest.approx(1.0)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_heldout_classification_metrics_use_reserved_labels(tmp_path, engine) -> None:
    """Accuracy and F1 must expose label drift in the 200 rows kept out of fit."""
    x = np.r_[np.full(400, -1.0), np.full(400, 1.0)]
    train = pd.DataFrame({"x": x, "target": np.r_[np.zeros(400), np.ones(400)]})
    test_x = np.r_[np.full(100, -1.0), np.full(100, 1.0)]
    test_y = np.r_[np.ones(20), np.zeros(80), np.zeros(20), np.ones(80)]
    test = pd.DataFrame({"x": test_x, "target": test_y.astype("int64")})
    train["target"] = train["target"].astype("int64")
    native_train = train if engine == "pandas" else pl.from_pandas(train)
    native_test = test if engine == "pandas" else pl.from_pandas(test)
    artifact = fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "logistic_regression"}},
        SplitDataset(train=native_train, test=native_test),
        target_column="target",
        artifact_path=tmp_path / engine,
        max_rows=1000,
        max_bytes=100_000,
    )

    metrics = local_batch.evaluate_local_holdout(artifact, native_test, target_column="target")

    assert metrics["heldout_accuracy"] == pytest.approx(0.8)
    assert metrics["heldout_f1"] == pytest.approx(0.8)
    assert metrics["heldout_f1_weighted"] == pytest.approx(0.8)
    assert metrics["heldout_balanced_accuracy"] == pytest.approx(0.8)
    assert metrics["heldout_precision"] == pytest.approx(0.8)
    assert metrics["heldout_recall"] == pytest.approx(0.8)
    assert metrics["heldout_roc_auc"] > 0.5
    assert np.isfinite(metrics["heldout_log_loss"]) and metrics["heldout_log_loss"] > 0.0


def test_scoring_rejects_changed_identity_and_raw_column_order_before_read(tmp_path) -> None:
    """A mismatched snapshot or input order must never trigger a Spark action."""
    train = pd.DataFrame(
        {
            "x": np.arange(8, dtype="float64"),
            "z": np.arange(8, dtype="float64") * 3,
            "target": np.arange(8, dtype="float64") * 2,
        }
    )
    path = tmp_path / "fitted"
    fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        SplitDataset(train=train.iloc[:6], test=train.iloc[6:]),
        target_column="target",
        artifact_path=path,
        max_rows=8,
        max_bytes=4096,
    )
    prepared = prepare_local_workflow(
        LocalWorkflowConfig(
            runtime="databricks",
            engine="pandas",
            source=InputSource(kind="uc_table", table="workspace.test.score_source", version=2),
            model=ModelSelection(kind="local_pipeline", path=str(path)),
            sink=OutputSink(kind="return_frame"),
        )
    )
    operations = []
    source = _Spark([], operations)
    with pytest.raises(ValueError, match="pinned Delta table and version"):
        score_local_source(source, _spec(version=3, input_columns=("x", "z")), prepared)
    with pytest.raises(ValueError, match="raw column order"):
        score_local_source(source, _spec(input_columns=("z", "x")), prepared)
    assert operations == []


def test_source_spec_rejects_ambiguous_identity_before_spark() -> None:
    """Invalid identifiers and naive dates cannot reach a remote read."""
    with pytest.raises(ValueError):
        _spec(table="workspace.test.score_source;DROP TABLE x")
    with pytest.raises(ValueError, match="timezone-aware"):
        _spec(period_start=datetime(2026, 1, 1))


def test_real_spark_iterator_filters_requested_month(monkeypatch) -> None:
    """The real Spark API must apply the month predicate before local scoring."""
    spark_sql = pytest.importorskip("pyspark.sql")
    monkeypatch.setenv("SPARK_LOCAL_IP", "127.0.0.1")
    spark = (
        spark_sql.SparkSession.builder.master("local[1]")
        .appName("skyulf-sm24a-local-read")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    try:
        frame = spark.createDataFrame(
            [
                ("jan", datetime(2026, 1, 10, tzinfo=UTC), 1.0),
                ("feb", datetime(2026, 2, 10, tzinfo=UTC), 2.0),
            ],
            "entity_id string, event_time timestamp, x double",
        )

        class Reader:
            """Return the real Spark frame after accepting the pinned selector."""

            def format(self, value):
                """Accept Delta selection from the adapter."""
                assert value == "delta"
                return self

            def option(self, name, value):
                """Check that the source version is explicit."""
                assert (name, value) == ("versionAsOf", 2)
                return self

            def table(self, name):
                """Yield the known local source frame."""
                assert name == "workspace.test.score_source"
                return frame

        result = read_local_source(SimpleNamespace(read=Reader()), _spec())
        assert result.to_dict("records") == [{"entity_id": "jan", "x": 1.0}]
    finally:
        spark.stop()


def test_reader_rejects_null_record_key_columns_before_scoring() -> None:
    """A missing business key cannot be repaired after local prediction."""
    with pytest.raises(ValueError, match="row keys must not be null"):
        read_local_source(_Spark([{"entity_id": None, "x": 2.0}], []), _spec())


def test_scoring_rejects_output_that_exceeds_local_byte_budget(tmp_path) -> None:
    """Probability columns must stay inside the declared local memory cap."""
    frame = pd.DataFrame(
        {"x": [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0], "target": [0, 0, 0, 0, 1, 1, 1, 1]}
    )
    path = tmp_path / "classifier"
    fit_local_workflow(
        {"preprocessing": [], "modeling": {"type": "logistic_regression"}},
        SplitDataset(train=frame.iloc[[0, 1, 2, 4, 5, 6]], test=frame.iloc[[3, 7]]),
        target_column="target",
        artifact_path=path,
        max_rows=8,
        max_bytes=4096,
    )
    prepared = prepare_local_workflow(
        LocalWorkflowConfig(
            runtime="databricks",
            engine="pandas",
            source=InputSource(
                kind="uc_table",
                table="workspace.test.score_source",
                version=2,
                max_rows=3,
                max_bytes=180,
            ),
            model=ModelSelection(kind="local_pipeline", path=str(path)),
            sink=OutputSink(kind="return_frame"),
        )
    )
    rows = [{"entity_id": 1, "x": 2.0}, {"entity_id": 2, "x": 4.0}]
    with pytest.raises(ValueError, match="Prediction result exceeds max_bytes"):
        score_local_source(_Spark(rows, []), _spec(max_bytes=180), prepared)
