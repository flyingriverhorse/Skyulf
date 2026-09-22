"""Real Delta control-table admission across independent Spark sessions."""

import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from uuid import uuid4

import pytest

from skyulf.integrations.databricks.admission import BatchConflictError


def _provider(spark, table):
    """Load the optional provider without importing Spark in the base test lane."""
    module = importlib.import_module("skyulf.integrations.databricks.delta_admission")
    return module.DeltaTableAdmission(spark, table)


@pytest.fixture
def control(delta_spark):
    """Provision one immutable target binding, as an operator must before use."""
    table = "default.admission_" + uuid4().hex
    target = str(uuid4())
    delta_spark.createDataFrame([(target, None)], "target_id string, owner string").write.format(
        "delta"
    ).saveAsTable(table)
    try:
        yield delta_spark, table, target
    finally:
        delta_spark.sql(f"DROP TABLE IF EXISTS {table}").collect()


def test_provider_is_available():
    """Distributed batch callers need a concrete provider they can import."""
    assert importlib.util.find_spec("skyulf.integrations.databricks.delta_admission") is not None


def test_held_claim_rejects_second_session_then_releases(control):
    """A shared table serializes publishers even with separate Spark sessions."""
    spark, table, target = control
    first = _provider(spark, table)
    second = _provider(spark.newSession(), table)
    with first.hold(target):
        token = spark.table(table).first().owner
        assert token
        with pytest.raises(BatchConflictError), second.hold(target):
            pytest.fail("A second publisher entered the held critical section.")
        assert spark.table(table).first().owner == token
    assert spark.table(table).first().owner is None
    with second.hold(target):
        assert spark.table(table).first().owner != token
    assert first.local_only is False


def test_body_failure_releases_claim(control):
    """A failed publication must permit a deliberate subsequent retry."""
    spark, table, target = control
    with (
        pytest.raises(RuntimeError, match="publication failed"),
        _provider(spark, table).hold(target),
    ):
        raise RuntimeError("publication failed")
    assert spark.table(table).first().owner is None


@pytest.mark.parametrize("malformation", ["empty", "duplicate", "wrong_target", "wrong_schema"])
def test_malformed_control_is_rejected_without_writing(control, malformation):
    """Malformed authority state must not be repaired or claimed implicitly."""
    spark, table, target = control
    if malformation == "empty":
        spark.sql(f"DELETE FROM {table}").collect()
    elif malformation == "duplicate":
        spark.sql(f"INSERT INTO {table} SELECT * FROM {table}").collect()
    elif malformation == "wrong_target":
        target = str(uuid4())
    else:
        spark.sql(f"ALTER TABLE {table} ADD COLUMN unexpected STRING").collect()
    before = spark.table(table).collect()
    version = spark.sql(f"DESCRIBE HISTORY {table}").first().version
    with pytest.raises(ValueError), _provider(spark, table).hold(target):
        pytest.fail("Malformed admission entered the critical section.")
    assert spark.table(table).collect() == before
    assert spark.sql(f"DESCRIBE HISTORY {table}").first().version == version


def test_release_never_clears_another_owner(control):
    """Even an unauthorized owner change cannot make cleanup clear someone else."""
    spark, table, target = control
    with pytest.raises(BatchConflictError), _provider(spark, table).hold(target):
        spark.sql(f"UPDATE {table} SET owner = 'other-owner'").collect()
    assert spark.table(table).first().owner == "other-owner"


def test_release_predicate_preserves_owner_changed_after_read(control, monkeypatch):
    """Cleanup's UPDATE must check ownership even after its read saw the old token."""
    spark, table, target = control
    provider = _provider(spark, table)
    update = provider._update
    calls = 0

    def change_owner_before_release(statement, target_id, token):
        """Inject a real Delta change between the release check and release write."""
        nonlocal calls
        calls += 1
        if calls == 2:
            spark.sql(f"UPDATE {table} SET owner = 'racing-owner'").collect()
        return update(statement, target_id, token)

    monkeypatch.setattr(provider, "_update", change_owner_before_release)
    with provider.hold(target):
        assert spark.table(table).first().owner
    assert spark.table(table).first().owner == "racing-owner"


def test_existing_owner_has_no_expiry(control):
    """Orphaned ownership requires manual recovery rather than automatic takeover."""
    spark, table, target = control
    spark.sql(f"UPDATE {table} SET owner = 'orphaned-owner'").collect()
    with pytest.raises(BatchConflictError), _provider(spark, table).hold(target):
        pytest.fail("An orphaned claim was stolen.")
    assert spark.table(table).first().owner == "orphaned-owner"


def test_lost_acquisition_acknowledgement_retains_claim(control, monkeypatch):
    """An uncertain committed claim must block publication and require manual recovery."""
    spark, table, target = control
    provider = _provider(spark, table)
    update = provider._update

    def lose_acknowledgement(statement, target_id, token):
        """Commit the real Delta UPDATE before simulating a transport failure."""
        update(statement, target_id, token)
        raise RuntimeError("acquisition acknowledgement lost")

    monkeypatch.setattr(provider, "_update", lose_acknowledgement)
    entered = False
    with (
        pytest.raises(RuntimeError, match="acquisition acknowledgement lost"),
        provider.hold(target),
    ):
        entered = True
    assert entered is False
    owner = spark.table(table).first().owner
    assert owner is not None
    with pytest.raises(BatchConflictError), _provider(spark.newSession(), table).hold(target):
        pytest.fail("An uncertain committed claim admitted another publisher.")
    assert spark.table(table).first().owner == owner


def test_cached_free_row_cannot_admit_a_second_owner(control):
    """Cached availability must not bypass committed ownership from another session."""
    spark, table, target = control
    other = spark.newSession()
    other.sql(f"CACHE TABLE {table}").collect()
    assert other.table(table).first().owner is None
    try:
        with (
            _provider(spark, table).hold(target),
            pytest.raises(BatchConflictError),
            _provider(other, table).hold(target),
        ):
            pytest.fail("Cached free state admitted a second owner.")
    finally:
        other.sql(f"UNCACHE TABLE {table}").collect()
    assert spark.table(table).first().owner is None


def test_concurrent_requests_have_exactly_one_owner(control, monkeypatch):
    """Two real Spark requests may race, but only one may enter the critical section."""
    spark, table, target = control
    providers = [_provider(spark.newSession(), table) for _ in range(2)]
    free_reads = Barrier(2)
    loser_finished = Event()

    def synchronize_first_read(provider):
        """Force both contenders to observe availability before either can UPDATE."""
        state = provider._state
        first_read = True

        def read_state(target_id):
            """Read actual Delta state and synchronize only the pre-acquisition read."""
            nonlocal first_read
            owner = state(target_id)
            if first_read:
                first_read = False
                assert owner is None
                free_reads.wait(timeout=60)
            return owner

        monkeypatch.setattr(provider, "_state", read_state)

    for provider in providers:
        synchronize_first_read(provider)

    def contend(provider):
        """Keep a winning claim until its racing peer has rejected admission."""
        try:
            with provider.hold(target):
                assert loser_finished.wait(timeout=120), "Both publishers entered or peer hung."
                return "owned"
        except BatchConflictError:
            loser_finished.set()
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(contend, providers))
    assert sorted(results) == ["owned", "rejected"]
    assert spark.table(table).first().owner is None


def test_replaced_control_table_is_rejected(control):
    """A long-lived provider must never silently bind a new control-table identity."""
    spark, table, target = control
    provider = _provider(spark, table)
    spark.sql(f"DROP TABLE {table}").collect()
    spark.createDataFrame([(target, None)], "target_id string, owner string").write.format(
        "delta"
    ).saveAsTable(table)
    with pytest.raises(BatchConflictError), provider.hold(target):
        pytest.fail("Replaced control authority admitted an existing provider.")
    assert spark.table(table).first().owner is None


def test_missing_control_table_is_not_created(delta_spark):
    """Admission must never manufacture an authority when provisioning is missing."""
    analysis_error = importlib.import_module("pyspark.errors").AnalysisException
    table = "default.missing_admission_" + uuid4().hex
    with pytest.raises(analysis_error):
        _provider(delta_spark, table)
    assert not delta_spark.catalog.tableExists(table)


def test_non_delta_control_table_is_rejected(delta_spark):
    """A superficially correct Parquet row cannot provide transactional admission."""
    table = "default.parquet_admission_" + uuid4().hex
    delta_spark.createDataFrame(
        [(str(uuid4()), None)], "target_id string, owner string"
    ).write.format("parquet").saveAsTable(table)
    try:
        with pytest.raises(ValueError, match="Delta"):
            _provider(delta_spark, table)
        assert delta_spark.table(table).first().owner is None
    finally:
        delta_spark.sql(f"DROP TABLE {table}").collect()
