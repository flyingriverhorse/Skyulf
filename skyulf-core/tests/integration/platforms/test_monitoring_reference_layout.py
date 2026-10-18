"""Reference layouts preserve training membership and exact stored source populations."""

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.monitoring import reference_layout
from skyulf.integrations.databricks.observability.monitoring.monitoring_config import MonitorConfig
from skyulf.integrations.databricks.observability.monitoring.monitoring_source_evidence import (
    source_evidence,
)
from skyulf.integrations.databricks.observability.monitoring.reference_layout import (
    reference_population_records,
)
from skyulf.integrations.databricks.observability.monitoring.spark import (
    spark_monitoring_reference as reference,
)
from skyulf.integrations.databricks.training.fitting.candidate import TrainingSpec


@pytest.fixture
def prepared_reference(monkeypatch):
    """Exercise receipt production with real frames and specs at the external storage boundary."""
    source = pd.DataFrame(
        {"id": [10, 20, 30, 40], "x": [1.0, 1.0, 3.0, 4.0], "y": [2.0, 2.0, 6.0, 8.0]}
    )
    written = {}

    def prepare(
        *,
        temporal=False,
        collision=False,
        model_version="1",
        target="y",
        trace=None,
        replace_on_acquire=False,
    ):
        """Return actual serialized membership while replacing only remote dependencies."""
        frame = source.rename(columns={"y": target}).copy()
        inputs = ["x"]
        if collision:
            frame["is_train"] = [False, True, False, True]
            inputs.append("is_train")
        split: dict[str, Any] = {}
        if temporal:
            frame["event"] = pd.date_range("2026-01-01", periods=4, tz="UTC")
            split = {
                "split_strategy": "temporal",
                "event_column": "event",
                "start": datetime(2026, 1, 1, tzinfo=UTC),
                "holdout_start": datetime(2026, 1, 4, tzinfo=UTC),
                "cutoff": datetime(2026, 1, 5, tzinfo=UTC),
                "test_size": None,
                "random_state": None,
                "stratify": None,
            }
        spec = TrainingSpec(
            table="a.b.training",
            version=1,
            record_key_columns=("id",),
            input_columns=tuple(inputs),
            target_column=target,
            max_rows=100,
            max_bytes=10000,
            **split,
        )
        artifact = SimpleNamespace(
            manifest=SimpleNamespace(fitted_engine="pandas", project_source_sha256=None)
        )
        evidence = {
            "training_run_id": "run",
            "model_version": model_version,
            "model_name": "a.b.model",
        }
        config = MonitorConfig(
            environment="test",
            project="layout",
            model_name="a.b.model",
            model_version=model_version,
            source_table=spec.table,
            prediction_table="a.b.predictions",
            execution_engine="spark",
            reference_namespace="a.monitoring",
        )
        held = {"active": False, "source_id": "source-id"}

        def observe(stage, result):
            """Record whether potentially long computation runs inside the shared admission."""
            if trace is not None:
                trace.append((stage, held["active"]))
            return result

        @contextmanager
        def admission(*args, **kwargs):
            """Bound publication precisely and optionally race replacement before it begins."""
            held["active"] = True
            if replace_on_acquire:
                held["source_id"] = "replacement-id"
            try:
                yield "owner"
            finally:
                held["active"] = False

        monkeypatch.setattr(
            reference,
            "load_monitoring_artifact",
            lambda *a, **k: observe("load", (artifact, spec, {}, evidence)),
        )
        monkeypatch.setattr(reference, "ensure_owned_object", lambda *a: False)
        monkeypatch.setattr(reference, "ensure_monitoring_store", lambda *a: "a.monitoring")
        monkeypatch.setattr(reference, "hold_reference_mutation", admission)
        monkeypatch.setattr(reference, "require_reference_mutation", Mock())
        monkeypatch.setattr(reference, "table_identity", lambda *a: held["source_id"])
        monkeypatch.setattr(
            reference, "read_training_snapshot", lambda *a: observe("source", frame)
        )
        monkeypatch.setattr(reference, "make_registry_client", Mock())
        monkeypatch.setattr(reference, "require_mlflow", Mock())
        monkeypatch.setattr(
            reference,
            "reference_document",
            lambda *a: source_evidence(frame, spec.source_columns, spec.dataset_id),
        )
        monkeypatch.setattr(reference, "validate_training_evidence", Mock())
        monkeypatch.setattr(
            reference,
            "split_labeled_snapshot",
            lambda *a, **k: (frame.head(3), frame.tail(1), None),
        )
        monkeypatch.setattr(reference, "read_snapshot", lambda *a: SimpleNamespace(schema=None))
        monkeypatch.setattr(
            reference,
            "_prepared_frame",
            lambda spark, data, *a: observe(
                "frame",
                SimpleNamespace(
                    data=data,
                    schema=SimpleNamespace(jsonValue=lambda: {"columns": list(data.columns)}),
                ),
            ),
        )
        monkeypatch.setattr(
            reference,
            "measure_holdout_values",
            lambda *a: observe("holdout", {"values": {"rmse": 0.1}}),
        )

        def write(spark, name, data):
            """Capture the values that real Delta writes would persist."""
            observe("write", None)
            written[name] = getattr(data, "data", data).copy(deep=True)
            return {"name": name, "id": f"id-{name}", "version": 0}

        monkeypatch.setattr(reference, "_write_population", write)
        spark = SimpleNamespace(
            createDataFrame=lambda rows, schema: pd.DataFrame(rows, columns=["payload", "digest"])
        )
        receipt = reference.prepare_spark_monitoring_reference(spark, config)
        return receipt, written

    return prepare


def test_expensive_reference_preparation_precedes_shared_admission(prepared_reference):
    """Unrelated projects must remain able to activate while a source or model is being loaded."""
    trace = []
    prepared_reference(trace=trace)
    assert all(not held for stage, held in trace if stage != "write")
    assert [held for stage, held in trace if stage == "write"] == [True, True, True]


def test_source_replacement_before_publication_cannot_create_a_receipt(prepared_reference):
    """Unlocked source reconstruction must be rebound to its original physical ID under admission."""
    trace = []
    with pytest.raises(ValueError, match="source was replaced"):
        prepared_reference(trace=trace, replace_on_acquire=True)
    assert not any(stage == "write" for stage, _ in trace)


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("collision", [False, True])
def test_population_receipt_preserves_membership_and_feature_names(
    prepared_reference, temporal, collision
):
    """A single population must retain duplicate training rows without overwriting user features."""
    receipt, written = prepared_reference(temporal=temporal, collision=collision)
    assert receipt["format"] == 2
    assert set(receipt["tables"]) == {"population", "source"}
    assert len(written) == 3
    membership = receipt["membership_column"]
    population = written[receipt["tables"]["population"]["name"]]
    assert population[membership].tolist() == ([True] * 3 if temporal else [True] * 3 + [False])
    assert population.loc[population[membership], "x"].tolist() == [1.0, 1.0, 3.0]
    assert population["x"].tolist() == ([1.0, 1.0, 3.0] if temporal else [1.0, 1.0, 3.0, 4.0])
    if collision:
        assert membership != "is_train"
        assert population["is_train"].tolist() == (
            [False, True, False] if temporal else [False, True, False, True]
        )
    assert receipt["holdout"] == {"values": {"rmse": 0.1}}


def test_format_one_population_bindings_are_preserved():
    """Existing persisted receipts must keep their original tables and version pins."""
    tables = {
        name: {"name": f"a.b.old_{name}", "id": f"old-{name}", "version": 0}
        for name in ("train", "seen", "source")
    }
    assert reference_population_records({"format": 1, "tables": tables}) == tables


def test_format_two_bindings_select_train_and_hide_membership(prepared_reference):
    """Logical reader records must share one physical population with different projections."""
    receipt, _ = prepared_reference(collision=True)
    records = reference_population_records(receipt)
    assert records["train"]["name"] == records["seen"]["name"]
    assert records["train"]["id"] == records["seen"]["id"]
    assert records["train"]["train_only"] is True
    assert records["seen"]["train_only"] is False
    assert records["train"]["membership_column"] == "_is_train"
    assert records["source"] == receipt["tables"]["source"]


def test_equal_source_versions_need_two_tables_per_receipt_plus_one_source(prepared_reference):
    """Repeated versions with identical source populations must actually reuse the same table name."""
    first, _ = prepared_reference(model_version="1")
    second, written = prepared_reference(model_version="2")
    assert first["tables"]["source"] == second["tables"]["source"]
    assert first["tables"]["population"]["name"] != second["tables"]["population"]["name"]
    assert len(written) == 5


def test_distinct_target_projections_do_not_claim_a_single_shared_source(prepared_reference):
    """Four target-specific snapshots need twelve tables until a canonical-source design exists."""
    sources = set()
    for index in range(4):
        receipt, written = prepared_reference(
            model_version=str(index + 1), target=f"target_{index}"
        )
        sources.add(receipt["tables"]["source"]["name"])
    assert len(sources) == 4
    assert len(written) == 12


def test_unknown_reference_format_is_rejected():
    """A future layout cannot be interpreted silently as an older population contract."""
    with pytest.raises(ValueError, match="Unsupported"):
        reference_population_records({"format": 3, "tables": {}})


def test_source_identity_shares_only_equal_verified_populations():
    """Model identity and row order may differ while the immutable source content stays equal."""
    frame = pd.DataFrame({"id": [1, 2, 2, 3], "x": [1.0, 3.0, 3.0, None]})
    schema = {
        "type": "struct",
        "fields": [{"name": "id", "type": "long"}, {"name": "x", "type": "double"}],
    }
    first = reference_layout.source_snapshot_identity(
        "table-id", 3, source_evidence(frame, ("id", "x"), "first-model"), schema
    )
    second = reference_layout.source_snapshot_identity(
        "table-id", 3, source_evidence(frame.iloc[::-1], ("id", "x"), "second-model"), schema
    )
    assert first == second
    assert reference_layout.shared_source_table_name(
        "a.monitoring", first
    ) == reference_layout.shared_source_table_name("a.monitoring", second)


@pytest.mark.parametrize(
    "change", ["values", "multiplicity", "schema", "table_id", "version", "projection"]
)
def test_source_identity_separates_different_content_and_schema(change):
    """Equal Delta version and column names cannot merge different labels, windows or samples."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 2.0]})
    schema = {"type": "struct", "fields": [{"name": "x", "type": "double"}]}
    first = reference_layout.source_snapshot_identity(
        "table-id", 3, source_evidence(frame, ("x",), "dataset"), schema
    )
    if change == "values":
        frame.loc[0, "x"] = 9.0
    if change == "multiplicity":
        frame = frame.iloc[:2]
    if change == "schema":
        schema = {"type": "struct", "fields": [{"name": "x", "type": "decimal(10,2)"}]}
    if change == "projection":
        frame = frame.rename(columns={"x": "other_target"})
    second = reference_layout.source_snapshot_identity(
        "other-id" if change == "table_id" else "table-id",
        4 if change == "version" else 3,
        source_evidence(frame, tuple(frame.columns), "dataset"),
        schema,
    )
    assert reference_layout.shared_source_table_name(
        "a.monitoring", first
    ) != reference_layout.shared_source_table_name("a.monitoring", second)


class Population:
    """Exercise writer multiset comparisons without requiring a JVM in local contract tests."""

    def __init__(self, rows):
        """Keep nulls and duplicate multiplicity exactly as Spark exceptAll does."""
        self.rows = rows
        self.schema = ("id long", "x double")

    def exceptAll(self, other):  # noqa: N802 - Spark API
        """Return the positive multiset difference rather than dropping repeated values."""
        from collections import Counter

        return Population(list((Counter(self.rows) - Counter(other.rows)).elements()))

    def limit(self, count):
        """Bound the difference probe to the same single-row existence check as production."""
        return Population(self.rows[:count])

    def count(self):
        """Return remaining row count for the writer's difference predicate."""
        return len(self.rows)


@pytest.mark.parametrize("winner", ["equal", "different", "foreign", "modified", "replaced"])
def test_concurrent_source_creator_is_validated_before_reuse(monkeypatch, winner):
    """An atomic CREATE loser must prove the winner's content, ownership and immutable identity."""
    from test_reference_cleanup import FakeSpark

    spark = FakeSpark()
    name = "a.monitoring.monitor_src_test"
    verified = Population([(1, 2.0), (1, 2.0), (2, None)])
    saved = Population(list(reversed(verified.rows)))

    def create(spark, name, frame):
        """Place the concurrent winner between the initial absence check and verification."""
        spark.add(name, owner="foreign" if winner == "foreign" else reference.OWNER)
        if winner == "different":
            saved.rows = saved.rows[:-1]
        if winner == "modified":
            spark.tables[name]["version"] = 1

    def snapshot(*args):
        """Simulate replacement after the writer binds the first Delta ID."""
        if winner == "replaced":
            spark.tables[name]["id"] = "replacement-id"
        return saved

    monkeypatch.setattr(reference, "_create_population", create)
    monkeypatch.setattr(reference, "read_snapshot", snapshot)
    if winner == "equal":
        assert reference._write_population(spark, name, verified) == {
            "name": name,
            "id": f"id-{name}",
            "version": 0,
        }
    else:
        with pytest.raises(ValueError):
            reference._write_population(spark, name, verified)


def test_membership_name_avoids_case_insensitive_spark_columns():
    """Spark's default case-insensitive resolver must not merge an existing feature with the flag."""
    train = pd.DataFrame({"x": [1], "IS_TRAIN": [7]})
    population, marker = reference_layout.combine_reference_population(
        train, train, ["x", "IS_TRAIN"], "random"
    )
    assert marker == "_is_train"
    assert population["IS_TRAIN"].tolist() == [7, 7]
    assert population[marker].tolist() == [True, False]
