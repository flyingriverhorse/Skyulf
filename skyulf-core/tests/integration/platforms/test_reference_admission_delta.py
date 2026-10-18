"""Real Delta concurrency proves the reference authority's conditional write boundary."""

import contextvars
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock, local
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from skyulf.integrations.databricks.observability.monitoring.reference_admission import (
    CONTROL_ROW,
    hold_reference_mutation,
    recover_reference_mutation,
    require_reference_mutation,
)


@pytest.fixture
def admission_store(delta_spark):
    """Create only a uniquely named disposable authority owned by this test."""
    from skyulf.integrations.databricks.lifecycle.retraining_requests import REQUEST_SCHEMA
    from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
        OWNER,
        PROPERTY,
    )

    catalog = os.environ.get("SKYULF_TEST_CATALOG", "spark_catalog")
    schema = f"reference_admission_test_{uuid4().hex}"
    namespace = f"{catalog}.{schema}"
    delta_spark.sql(f"CREATE SCHEMA {namespace}").collect()
    try:
        delta_spark.sql(
            f"CREATE TABLE {namespace}.retraining_requests ({REQUEST_SCHEMA}) USING DELTA "
            f"TBLPROPERTIES ('{PROPERTY}'='{OWNER}', 'delta.isolationLevel'='Serializable')"
        ).collect()
        yield namespace
    finally:
        assert (
            schema.startswith("reference_admission_test_")
            and len(schema) == len("reference_admission_test_") + 32
        )
        delta_spark.sql(f"DROP SCHEMA {namespace} CASCADE").collect()


def test_native_simultaneous_first_claim_has_exactly_one_owner(delta_spark, admission_store):
    """Two first MERGEs cannot create duplicate control rows or enter concurrently."""
    before_merge, acquired = Barrier(2), Barrier(2)
    entered, rejected = [], []

    def concurrent_sql(statement, **kwargs):
        """Make both claimants read the absent row before their conditional writes."""
        if statement.startswith("MERGE"):
            before_merge.wait(timeout=60)
        return delta_spark.sql(statement, **kwargs)

    spark = SimpleNamespace(
        sql=concurrent_sql,
        table=delta_spark.table,
        createDataFrame=delta_spark.createDataFrame,
        catalog=delta_spark.catalog,
    )

    def claim():
        """Keep the winner held until the competing attempt has completed."""
        acquired_owner = False
        try:
            with hold_reference_mutation(spark, admission_store, wait_timeout_s=0) as owner:
                acquired_owner = True
                entered.append(owner)
                acquired.wait(timeout=90)
                require_reference_mutation(spark, admission_store, owner)
        except Exception as exc:  # noqa: BLE001 - Spark transports expose different Delta conflict types
            rejected.append(exc)
            if not acquired_owner:
                acquired.wait(timeout=90)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim) for _ in range(2)]
        for future in futures:
            future.result(timeout=180)
    assert len(entered) == len(rejected) == 1
    assert any(token in str(rejected[0]).lower() for token in ("concurrent", "held"))
    rows = delta_spark.table(f"{admission_store}.retraining_requests").collect()
    assert len(rows) == 1 and rows[0]["row_id"] == CONTROL_ROW
    assert rows[0]["status"] == "free" and rows[0]["request_id"] is None


def test_native_context_reentrancy_blocks_other_invocations(delta_spark, admission_store):
    """Real Delta ownership remains held across nested calls until the outer scope exits."""

    def competitor():
        """Use a fresh context to emulate a different job process."""
        with hold_reference_mutation(delta_spark, admission_store, wait_timeout_s=0):
            pytest.fail("independent operation entered")

    with hold_reference_mutation(delta_spark, admission_store) as owner:
        with hold_reference_mutation(delta_spark, admission_store) as nested:
            assert owner == nested
        with pytest.raises(RuntimeError, match="held"):
            contextvars.Context().run(competitor)
        require_reference_mutation(delta_spark, admission_store, owner)
    assert delta_spark.table(f"{admission_store}.retraining_requests").first()["status"] == "free"


def test_native_noop_claim_retries_after_competing_owner_finishes(delta_spark, admission_store):
    """A real successful no-op MERGE may observe a released rival at ownership readback."""
    claims = []

    def competing_sql(statement, **kwargs):
        """Finish a genuine rival between our free-row read and conditional-claim readback."""
        if statement.startswith("MERGE") and not claims:
            claims.append(statement)

            def finish_competitor():
                """Hold actual Delta authority while executing our losing SQL statement."""
                with hold_reference_mutation(delta_spark, admission_store, wait_timeout_s=0):
                    delta_spark.sql(statement, **kwargs).collect()

            contextvars.Context().run(finish_competitor)
            return SimpleNamespace(collect=lambda: [])
        if statement.startswith("MERGE"):
            claims.append(statement)
        return delta_spark.sql(statement, **kwargs)

    spark = SimpleNamespace(
        sql=competing_sql,
        table=delta_spark.table,
        createDataFrame=delta_spark.createDataFrame,
        catalog=delta_spark.catalog,
    )
    with hold_reference_mutation(spark, admission_store, wait_timeout_s=120) as owner:
        row = delta_spark.table(f"{admission_store}.retraining_requests").first()
        assert row["status"] == "held" and row["request_id"] == owner
        require_reference_mutation(spark, admission_store, owner)
    rows = delta_spark.table(f"{admission_store}.retraining_requests").collect()
    assert len(claims) == 2 and len(rows) == 1
    assert rows[0]["status"] == "free" and rows[0]["request_id"] is None


def test_native_simultaneous_claimants_wait_and_both_complete(
    delta_spark, admission_store, record_property
):
    """Serializable contention must let both jobs finish without concurrent ownership."""
    first_merge, state, mutex = Barrier(2), local(), Lock()
    active, completed, errors = set(), [], []

    def record_error(error):
        """Retain native metadata even when the controller successfully retries the claim."""
        evidence = {"type": f"{type(error).__module__}.{type(error).__name__}"}
        evidence["header"] = str(error).splitlines()[0]
        for name in ("getCondition", "getErrorClass"):
            try:
                getter = getattr(error, name, None)
                evidence[name] = getter() if getter is not None else None
            except Exception as failure:  # noqa: BLE001 - preserve unreadable metadata as evidence
                evidence[name] = {"failed": type(failure).__name__}
        with mutex:
            errors.append(evidence)

    def concurrent_sql(statement, **kwargs):
        """Race only the initial conditional writes, leaving retries free to progress."""
        if statement.startswith("MERGE") and not getattr(state, "attempted", False):
            state.attempted = True
            first_merge.wait(timeout=60)
        try:
            result = delta_spark.sql(statement, **kwargs)
        except Exception as error:
            record_error(error)
            raise
        if not statement.startswith("MERGE"):
            return result

        def collect():
            """Capture lazy transport failures without changing when the claim executes."""
            try:
                return result.collect()
            except Exception as error:
                record_error(error)
                raise

        return SimpleNamespace(collect=collect)

    spark = SimpleNamespace(
        sql=concurrent_sql,
        table=delta_spark.table,
        createDataFrame=delta_spark.createDataFrame,
        catalog=delta_spark.catalog,
    )

    def claim():
        """Verify unique real Delta authority while the other writer waits or retries."""
        with hold_reference_mutation(spark, admission_store, wait_timeout_s=120) as owner:
            with mutex:
                assert not active
                active.add(owner)
            require_reference_mutation(spark, admission_store, owner)
            with mutex:
                active.remove(owner)
                completed.append(owner)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(claim) for _ in range(2)]
            for future in futures:
                future.result(timeout=180)
    finally:
        record_property("delta_claim_errors", json.dumps(errors, sort_keys=True))
    rows = delta_spark.table(f"{admission_store}.retraining_requests").collect()
    assert len(set(completed)) == 2 and not active and len(rows) == 1
    assert rows[0]["status"] == "free" and rows[0]["request_id"] is None


def test_native_failed_owner_recovery_has_durable_intent_and_result(delta_spark, admission_store):
    """A verified recovery must atomically clear its token and retain both audit records."""
    with (
        pytest.raises(TimeoutError),
        hold_reference_mutation(
            delta_spark,
            admission_store,
            job_run_id=17,
            task_key="cleanup_references",
            workspace_host="https://test.cloud.databricks.com",
        ) as held_owner,
    ):
        require_reference_mutation(delta_spark, admission_store, held_owner)
        raise TimeoutError("remote response lost")
    owner = delta_spark.table(f"{admission_store}.retraining_requests").first()["request_id"]
    workspace = Mock()
    workspace.config.host = "https://test.cloud.databricks.com"
    workspace.jobs.get_run.return_value = SimpleNamespace(
        run_id=17,
        state=SimpleNamespace(life_cycle_state="TERMINATED"),
        tasks=[SimpleNamespace(task_key="cleanup_references")],
    )
    result = recover_reference_mutation(
        delta_spark, admission_store, workspace=workspace, owner=owner, confirm=True
    )
    assert result["status"] == "released"
    records = delta_spark.table(f"{admission_store}.reference_cleanup_audit").collect()
    assert {row["status"] for row in records} == {"intent", "released"}
    assert all(
        row["reference_id"] == owner and row["action"] == "release_admission" for row in records
    )
    with hold_reference_mutation(delta_spark, admission_store) as new_owner:
        assert new_owner != owner
