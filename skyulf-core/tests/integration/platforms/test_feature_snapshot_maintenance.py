"""Maintenance may advance Delta history without changing the frozen feature population."""

from collections import Counter
from types import SimpleNamespace

import pytest

from skyulf.integrations.databricks.feature_store import snapshots


class Relation:
    """Evaluate exact bag equality while exposing only a bounded count to the driver."""

    def __init__(self, rows, schema=("id", "amount")):
        self.rows = list(rows)
        self.schema = schema

    def exceptAll(self, other):
        """Retain duplicate multiplicity, including a changed duplicate row."""
        return Relation((Counter(self.rows) - Counter(other.rows)).elements())

    def limit(self, count):
        """Bound evidence collection independently of the distributed scan size."""
        return Relation(self.rows[:count], self.schema)

    def count(self):
        """Return the bounded mismatch cardinality."""
        return len(self.rows)


class Reader:
    """Resolve immutable versioned relations without consulting the current head."""

    def __init__(self, versions):
        self.versions = versions
        self.version = None

    def format(self, name):
        """Retain the Delta reader interface."""
        assert name == "delta"
        return self

    def option(self, name, value):
        """Record the explicit versionAsOf boundary."""
        assert name == "versionAsOf"
        self.version = value
        return self

    def table(self, name):
        """Reject missing retained versions instead of falling back to the latest data."""
        assert name == "catalog.features.values"
        return self.versions[self.version]


@pytest.fixture
def snapshot_case(monkeypatch):
    """Pin one unchanged table identity across a maintenance commit."""
    expected = {"table_name": "catalog.features.values", "table_id": "fixed", "version": 4}
    current = {**expected, "version": 5}
    versions = {
        4: Relation([(1, 2), (2, None), (1, 2)]),
        5: Relation([(2, None), (1, 2), (1, 2)]),
    }
    monkeypatch.setattr(snapshots, "snapshot_table", lambda *args: dict(current))
    return SimpleNamespace(read=Reader(versions)), expected, current, versions


def test_maintenance_version_preserves_exact_feature_population(snapshot_case):
    """Physical row reordering must not force retraining when schema and bag contents match."""
    spark, expected, _, _ = snapshot_case
    snapshots.validate_snapshots(spark, [expected])
    assert expected["version"] == 4


@pytest.mark.parametrize(
    "rows",
    [
        [(1, 3), (2, None), (1, 2)],
        [(1, 2), (2, None)],
        [(1, 2), (2, None), (1, 2), (3, 4)],
        [(1, 2), (2, None), (2, None)],
    ],
)
def test_feature_mutations_remain_rejected(snapshot_case, rows):
    """Updates, removals, inserts and duplicate-count changes cannot hide behind maintenance."""
    spark, expected, _, versions = snapshot_case
    versions[5] = Relation(rows)
    with pytest.raises(ValueError, match="changed.*retrain"):
        snapshots.validate_snapshots(spark, [expected])


@pytest.mark.parametrize("change", ["identity", "schema", "head", "missing_history"])
def test_maintenance_proof_fails_closed(snapshot_case, monkeypatch, change):
    """A data comparison cannot authorize replacement, schema drift or a moving table head."""
    spark, expected, current, versions = snapshot_case
    if change == "identity":
        current["table_id"] = "replacement"
    elif change == "schema":
        versions[5].schema = ("id", "renamed")
    elif change == "head":
        heads = iter([dict(current), {**current, "version": 6}])
        monkeypatch.setattr(snapshots, "snapshot_table", lambda *args: next(heads))
    else:
        del versions[4]
    with pytest.raises((ValueError, KeyError)):
        snapshots.validate_snapshots(spark, [expected])
