"""Reference cleanup deletes only provably owned, unneeded and old monitoring references."""

import json
import re
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from skyulf.integrations.databricks.observability.monitoring import reference_cleanup
from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
    MonitorConfig,
    json_digest,
)
from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
    OWNER,
)
from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_reference import (
    reference_table_name,
)

NS = "workspace.skyulf_monitoring"
NOW = datetime(2026, 10, 17, tzinfo=UTC)
OLD = NOW - timedelta(days=200)
VALUES = {
    "monitoring_catalog": "workspace",
    "monitoring_schema": "skyulf_monitoring",
    "monitoring_environment": "test",
    "monitoring_project": "air",
}


@pytest.fixture(autouse=True)
def storage_admission_boundary(monkeypatch):
    """Keep catalog decision tests independent of the separately tested Delta lock transport."""
    monkeypatch.setattr(
        reference_cleanup, "hold_reference_mutation", lambda *a: nullcontext("owner")
    )
    monkeypatch.setattr(reference_cleanup, "require_reference_mutation", lambda *a: None)


class Frame:
    """Rows with the small DataFrame surface the cleanup code reads and writes."""

    def __init__(
        self, spark: "FakeSpark", rows: list[dict], name: str | None = None, schema: str = ""
    ) -> None:
        """Keep rows, an optional table name for writes, and a schema string."""
        self.spark, self.rows, self.name = spark, rows, name
        self.schema = SimpleNamespace(simpleString=lambda: schema)

    def limit(self, count: int) -> "Frame":
        """Return at most count rows."""
        return Frame(self.spark, self.rows[:count], self.name, self.schema.simpleString())

    def select(self, *columns: str) -> "Frame":
        """Select every column; the fake keeps whole rows."""
        return self

    def collect(self) -> list[dict]:
        """Materialize rows."""
        return list(self.rows)

    def first(self) -> dict | None:
        """Return the first row or None."""
        return self.rows[0] if self.rows else None

    def writeTo(self, name: str) -> "Frame":  # noqa: N802 - Spark API name
        """Target a table for append."""
        return Frame(self.spark, self.rows, name)

    def append(self) -> None:
        """Append rows to the named table and log the write."""
        assert self.name is not None
        table = self.spark.tables[self.name]
        table["rows"].extend(self.rows)
        for row in self.rows:
            self.spark.log.append(f"AUDIT {row['action']} {row['status']} {row['table_name']}")


class Reader:
    """spark.read.format(...).option(...).table(...) for pinned snapshots."""

    def __init__(self, spark: "FakeSpark") -> None:
        """Bind the catalog."""
        self.spark = spark

    def format(self, value: str) -> "Reader":
        """Accept the delta format."""
        return self

    def option(self, key: str, value: Any) -> "Reader":
        """Accept a version pin."""
        return self

    def table(self, name: str) -> Frame:
        """Read table rows, then run an optional hook such as a concurrent replacement."""
        frame = Frame(self.spark, self.spark.tables[name]["rows"])
        if self.spark.after_read is not None:
            self.spark.after_read(name)
        return frame


class Catalog:
    """spark.catalog.tableExists over the fake tables."""

    def __init__(self, spark: "FakeSpark") -> None:
        """Bind the catalog."""
        self.spark = spark

    def tableExists(self, name: str) -> bool:  # noqa: N802 - Spark API name
        """Report whether a table exists."""
        return name in self.spark.tables


class FakeSpark:
    """An in-memory Unity Catalog schema that records every mutating statement."""

    def __init__(self) -> None:
        """Start with an empty catalog and log."""
        self.tables: dict[str, dict] = {}
        self.log: list[str] = []
        self.fail_drop: str | None = None
        self.before_drop: Any = None
        self.after_read: Any = None
        self.audit_owner = OWNER
        self.catalog = Catalog(self)
        self.read = Reader(self)

    def add(self, name, rows=(), *, owner=OWNER, created=OLD, size=100, table_id=None):
        """Create a table with an owner property, creation time and size."""
        self.tables[name] = {
            "rows": list(rows),
            "owner": owner,
            "created": created,
            "size": size,
            "id": table_id or f"id-{name}",
            "version": 0,
            "properties": {"delta.appendOnly": "true"},
            "schema": "",
        }

    def table(self, name: str) -> Frame:
        """Read a whole table."""
        table = self.tables[name]
        return Frame(self, table["rows"], schema=table["schema"])

    def createDataFrame(self, rows: list[dict], schema: str) -> Frame:  # noqa: N802
        """Wrap rows for an append."""
        return Frame(self, rows, schema=schema)

    def sql(self, statement: str) -> Frame:
        """Interpret the few statements cleanup issues."""
        name = _unquote(statement)
        if statement.startswith("SHOW TBLPROPERTIES"):
            return Frame(self, [{"value": self.tables[name]["owner"]}])
        if statement.startswith("DESCRIBE DETAIL"):
            table = self.tables[name]
            detail = {"format": "delta", "id": table["id"], "createdAt": table["created"]}
            detail |= {"sizeInBytes": table["size"], "properties": table["properties"]}
            return Frame(self, [detail])
        if statement.startswith("DESCRIBE HISTORY"):
            return Frame(self, [{"version": self.tables[name]["version"]}])
        if statement.startswith("SHOW TABLES"):
            schema = name
            return Frame(
                self,
                [
                    {"tableName": table.rsplit(".", 1)[1]}
                    for table in self.tables
                    if table.rsplit(".", 1)[0] == schema
                    and re.fullmatch(r"monitor_ref_.*_receipt", table.rsplit(".", 1)[1])
                ],
            )
        if statement.startswith("DROP TABLE"):
            if self.before_drop is not None:
                self.before_drop(name)
            if name == self.fail_drop:
                raise RuntimeError("drop failed")
            self.log.append(f"DROP {name}")
            del self.tables[name]
            return Frame(self, [])
        if statement.startswith("CREATE TABLE IF NOT EXISTS"):
            if name not in self.tables:
                self.log.append(f"CREATE {name}")
                self.add(name, owner=self.audit_owner)
                self.tables[name]["schema"] = reference_cleanup.AUDIT_SCHEMA
            return Frame(self, [])
        raise AssertionError(f"Unexpected SQL: {statement}")


def _unquote(statement: str) -> str:
    """Return the first backtick-quoted identifier of a statement."""
    match = re.search(r"((?:`[^`]+`\.?)+)", statement)
    assert match, statement
    return match.group(1).replace("`", "")


def _config(**changes) -> MonitorConfig:
    """A Spark monitor of project air in the shared store."""
    values: dict[str, Any] = {
        "environment": "test",
        "project": "air",
        "model_name": "workspace.air.no2_model",
        "model_version": "3",
        "source_table": "workspace.air.observations",
        "prediction_table": "workspace.air.predictions",
        "execution_engine": "spark",
        "reference_namespace": NS,
    }
    return MonitorConfig(**(values | changes))


def _prepare(spark: FakeSpark, config: MonitorConfig, version: str, **table) -> str:
    """Write a verified receipt and its three child tables for one model version."""
    evidence = {"model_name": config.model_name, "model_version": version, "run": version}
    tables = {}
    for key in reference_cleanup.CHILDREN:
        name = reference_table_name(config, evidence, key)
        spark.add(name, **table)
        tables[key] = {"name": name, "id": f"id-{name}", "version": 0}
    receipt = {"format": 1, "evidence": evidence, "tables": tables, "holdout": {}}
    name = reference_table_name(config, evidence, "receipt")
    spark.add(name, [{"payload": json.dumps(receipt), "digest": json_digest(receipt)}], **table)
    return name


def _store(spark: FakeSpark, requests: list[dict] | None = None) -> None:
    """Add owned request and inventory tables to the store."""
    spark.add(f"{NS}.retraining_requests", requests or [])


def _prepare_shared(spark, config, version):
    """Persist two format-2 tables while reusing one verifiable shared source identity."""
    from skyulf.integrations.databricks.observability.monitoring.reference_layout import (
        shared_source_table_name,
    )

    evidence = {"model_name": config.model_name, "model_version": version, "run": version}
    identity = {
        "format": 2,
        "table_id": "source-id",
        "version": 0,
        "columns": ["x"],
        "rows": 1,
        "content_sha256": "a" * 64,
        "schema": {"type": "struct", "fields": []},
    }
    source = shared_source_table_name(NS, identity)
    population = reference_table_name(config, evidence, "population")
    if source not in spark.tables:
        spark.add(source)
    spark.add(population)
    tables = {
        key: {"name": name, "id": f"id-{name}", "version": 0}
        for key, name in (("population", population), ("source", source))
    }
    receipt = {
        "format": 2,
        "evidence": evidence,
        "tables": tables,
        "holdout": {},
        "source_identity": identity,
        "source_table_id": "source-id",
        "membership_column": "is_train",
    }
    name = reference_table_name(config, evidence, "receipt")
    spark.add(name, [{"payload": json.dumps(receipt), "digest": json_digest(receipt)}])
    return name, source


def test_shared_source_survives_another_projects_receipt(monkeypatch):
    """One project's retirement cannot invalidate a reference owned by another project."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    old, source = _prepare_shared(spark, config, "1")
    foreign = _config(project="other", model_name="workspace.other.model", model_version="1")
    other, same_source = _prepare_shared(spark, foreign, "1")
    result = _run(spark, monkeypatch, [config, foreign], cleanup_confirm="true")
    assert result["deleted"] == [old]
    assert same_source == source
    assert source in spark.tables and other in spark.tables


def test_shared_source_drops_once_with_last_retiring_receipt(monkeypatch):
    """A batch can retire every user and delete the source once while retaining retry receipts."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    first, source = _prepare_shared(spark, config, "1")
    second, _ = _prepare_shared(spark, config, "2")
    result = _run(spark, monkeypatch, [config], cleanup_confirm="true")
    assert set(result["deleted"]) == {first, second}
    assert spark.log.count(f"DROP {source}") == 1
    last_receipt = result["deleted"][-1]
    assert spark.log.index(f"DROP {source}") < spark.log.index(f"DROP {last_receipt}")
    assert source not in spark.tables


def test_source_survives_receipt_excluded_by_first_plan(monkeypatch):
    """Intersecting plans must also protect the source when the second plan retires more users."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    first, source = _prepare_shared(spark, config, "1")
    second, _ = _prepare_shared(spark, config, "2")
    retained, retiring = sorted([first, second])
    version = json.loads(spark.tables[retained]["rows"][0]["payload"])["evidence"]["model_version"]
    plans = iter([{"champion": version}, {}])
    monkeypatch.setattr(reference_cleanup, "load_enrolled_models", lambda *a: [config.payload()])
    result = reference_cleanup.cleanup_references(
        spark, VALUES | {"cleanup_confirm": "true"}, registry=lambda _: next(plans), now=NOW
    )
    assert result["deleted"] == [retiring]
    assert retained in spark.tables and source in spark.tables


def test_malformed_foreign_receipt_blocks_shared_source_retirement(monkeypatch):
    """Unknown reverse references must fail closed instead of turning uncertainty into deletion."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    receipt, source = _prepare_shared(spark, config, "1")
    spark.add(f"{NS}.monitor_ref_corrupt_receipt", [{"payload": "broken", "digest": "bad"}])
    result = _run(spark, monkeypatch, [config], cleanup_confirm="true")
    assert result["deleted"] == []
    assert result["references"][0]["decision"] == "protected"
    assert "unknown" in result["references"][0]["reason"]
    assert receipt in spark.tables and source in spark.tables


@pytest.mark.parametrize("mutation", ["alias", "receipt"])
def test_cleanup_admission_excludes_new_dependency_after_final_plan(monkeypatch, mutation):
    """The original final-plan race must be rejected before a cooperating writer publishes."""
    from concurrent.futures import ThreadPoolExecutor

    from skyulf.integrations.databricks.observability.monitoring import reference_admission

    spark = FakeSpark()
    _store(spark)
    config = _config()
    receipt, source = _prepare_shared(spark, config, "1")
    state = {"owner": None}

    def acquire(owner, holder=None):
        """Represent the durable authority while exercising the real context and call ordering."""
        if state["owner"] is not None:
            raise RuntimeError("held by another operation")
        state["owner"] = owner

    def require(owner):
        """Check that each destructive operation still belongs to the original holder."""
        assert state["owner"] == owner

    authority = SimpleNamespace(
        acquire=acquire, require=require, release=lambda owner: state.update(owner=None)
    )
    monkeypatch.setattr(reference_admission, "_Authority", lambda *a: authority)
    monkeypatch.setattr(
        reference_cleanup, "hold_reference_mutation", reference_admission.hold_reference_mutation
    )
    monkeypatch.setattr(
        reference_cleanup,
        "require_reference_mutation",
        reference_admission.require_reference_mutation,
    )
    original = reference_cleanup._record
    blocked = []
    aliases = {}

    def concurrent_writer():
        """Attempt the operation at the exact point that previously admitted a lost dependency."""
        with reference_admission.hold_reference_mutation(spark, NS):
            if mutation == "alias":
                aliases[config.model_name] = {"champion": "1"}
            else:
                _prepare_shared(spark, _config(project="other"), "2")

    def audited(*args):
        """Start a separate caller after both plans, before any DROP is issued."""
        original(*args)
        if args[-2] == "intent":
            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(RuntimeError, match="held by another"):
                    pool.submit(concurrent_writer).result()
                blocked.append(mutation)

    monkeypatch.setattr(reference_cleanup, "_record", audited)
    result = _run(spark, monkeypatch, [config], aliases=aliases, cleanup_confirm="true")
    assert blocked == [mutation]
    assert result["deleted"] == [receipt]
    assert source not in spark.tables and aliases == {}


def test_confirmed_cleanup_carries_job_holder_into_admission(monkeypatch):
    """Scheduled cleanup failures must retain enough provenance for verified recovery."""
    spark = FakeSpark()
    _store(spark)
    holders = []

    def hold(spark, namespace, **holder):
        """Capture the real widget conversion at the admission boundary."""
        holders.append(holder)
        return nullcontext("owner")

    monkeypatch.setattr(reference_cleanup, "hold_reference_mutation", hold)
    _run(
        spark,
        monkeypatch,
        [_config()],
        cleanup_confirm="true",
        job_run_id="42",
        task_key="cleanup_references",
        workspace_host="https://workspace.example/",
    )
    assert holders == [
        {
            "job_run_id": 42,
            "task_key": "cleanup_references",
            "workspace_host": "https://workspace.example",
        }
    ]


def test_recovery_audit_shares_owned_table_and_keeps_terminal_proof():
    """Explicit recovery must record which durable holder was released and why."""
    spark = FakeSpark()
    owner = "b" * 32
    proof = {"job_run_id": 42, "life_cycle_state": "TERMINATED"}
    for status in ("intent", "released"):
        reference_cleanup.append_reference_recovery(
            spark, NS, owner=owner, status=status, evidence=proof
        )
    rows = spark.tables[f"{NS}.{reference_cleanup.AUDIT_TABLE}"]["rows"]
    assert [row["status"] for row in rows] == ["intent", "released"]
    assert all(
        row["action"] == "release_admission" and row["reference_id"] == owner for row in rows
    )
    assert all(json.loads(row["reason"]) == proof for row in rows)


def test_recovery_intent_is_not_a_reference_deletion_intent():
    """A recovered guard token must never authorize resuming an incomplete reference deletion."""
    spark = FakeSpark()
    reference_cleanup.append_reference_recovery(
        spark, NS, owner="b" * 32, status="intent", evidence={}
    )
    assert reference_cleanup._audit_intents(spark, NS) == set()


def _run(spark, monkeypatch, configs, *, aliases=None, run_state=None, **values):
    """Run cleanup with an injected inventory, registry and job-run lookup."""
    monkeypatch.setattr(
        reference_cleanup,
        "load_enrolled_models",
        lambda spark, namespace: [config.payload() for config in configs],
    )
    return reference_cleanup.cleanup_references(
        spark,
        VALUES | values,
        registry=lambda name: dict((aliases or {}).get(name, {})),
        run_state=run_state or (lambda run_id: "TERMINATED"),
        now=NOW,
    )


def _decisions(result: dict) -> dict[str, str]:
    """Map each reference's model version to its decision or protection reason."""
    return {
        item["model_version"]: item["reason"] if item["decision"] == "protected" else "delete"
        for item in result["references"]
    }


def test_dry_run_lists_old_unneeded_versions_without_any_write(monkeypatch):
    """The default must only report, so nothing is created, written or dropped."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    old = _prepare(spark, config, "1")
    _prepare(spark, config, "3")
    result = _run(spark, monkeypatch, [config])
    assert result["status"] == "dry_run"
    assert _decisions(result) == {"1": "delete", "3": "enrolled model version"}
    deletable = next(item for item in result["references"] if item["decision"] == "deletable")
    assert deletable["receipt"] == old and deletable["size_bytes"] == 400
    assert spark.log == []


def test_confirm_drops_children_before_the_receipt_between_audit_rows(monkeypatch):
    """The receipt must stay until all three population tables have been dropped."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    old = _prepare(spark, config, "1")
    _prepare(spark, config, "3")
    result = _run(spark, monkeypatch, [config], cleanup_confirm="true")
    audit = f"{NS}.{reference_cleanup.AUDIT_TABLE}"
    children = [old.replace("_receipt", f"_{key}") for key in reference_cleanup.CHILDREN]
    assert spark.log == [
        f"CREATE {audit}",
        f"AUDIT delete intent {old}",
        *[f"DROP {name}" for name in children],
        f"DROP {old}",
        f"AUDIT delete deleted {old}",
    ]
    assert result["status"] == "deleted" and result["deleted"] == [old]


def test_final_audit_failure_does_not_undo_completed_deletion(monkeypatch):
    """An intent without a final audit row can describe an already completed deletion."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    old = _prepare(spark, config, "1")
    record = reference_cleanup._record

    def fail_final_audit(spark, audit, cleanup_id, item, status, reason):
        """Let the intent persist and fail only after the last table has been dropped."""
        if status == "deleted":
            raise RuntimeError("audit destination unavailable")
        return record(spark, audit, cleanup_id, item, status, reason)

    monkeypatch.setattr(reference_cleanup, "_record", fail_final_audit)
    with pytest.raises(RuntimeError, match="audit destination unavailable"):
        _run(spark, monkeypatch, [config], cleanup_confirm="true")
    assert f"AUDIT delete intent {old}" in spark.log
    assert not any(
        name in spark.tables
        for name in [
            old,
            *[old.replace("_receipt", f"_{key}") for key in reference_cleanup.CHILDREN],
        ]
    )
    assert spark.log[-1] == f"DROP {old}"


def test_other_projects_and_environments_are_never_candidates(monkeypatch):
    """Ownership comes from the hashed monitor identity, not the shared name prefix."""
    spark = FakeSpark()
    _store(spark)
    mine = _config()
    other_project = _config(project="bike", model_version="9")
    other_env = _config(environment="prod", model_version="9")
    _prepare(spark, other_project, "1")
    _prepare(spark, other_env, "1")
    result = _run(spark, monkeypatch, [mine, other_project, other_env], cleanup_confirm="true")
    assert result["references"] == [] and spark.log == []


@pytest.mark.parametrize(
    ("configs", "aliases", "requests", "run_state", "reason"),
    [
        (
            [_config(project="bike", model_name="workspace.air.no2_model", model_version="1")],
            None,
            None,
            None,
            "enrolled model version",
        ),
        ([], {"workspace.air.no2_model": {"challenger": "1"}}, None, None, "registry alias"),
        (
            [
                _config(
                    project="bike",
                    model_version="3",
                    label_table="workspace.air.labels",
                    result_available_at_column="known_at",
                    performance_policy={
                        "mode": "report",
                        "metric": "wape",
                        "direction": "lower",
                        "baseline": {"kind": "training_holdout", "model_version": "1"},
                        "tolerance": 0.2,
                        "tolerance_mode": "relative",
                        "window_hours": 24,
                        "label_delay_hours": 6,
                        "minimum_labeled_rows": 20,
                        "minimum_label_coverage": 0.8,
                        "consecutive_windows": 2,
                    },
                )
            ],
            None,
            None,
            None,
            "performance baseline",
        ),
        ([], None, [{"request_id": "r1", "status": "intent", "run_id": None}], None, "r1"),
        ([], None, [{"request_id": "r2", "status": "submitted", "run_id": 7}], "RUNNING", "r2"),
        ([], None, [{"request_id": "r3", "status": "submitted", "run_id": 7}], "error", "r3"),
    ],
)
def test_each_dependency_protects_an_old_version(
    monkeypatch, configs, aliases, requests, run_state, reason
):
    """Enrollment, aliases, baselines and unfinished retraining each keep a reference."""
    spark = FakeSpark()
    _store(spark, requests)
    config = _config(enabled=False)
    _prepare(spark, config, "1")

    def state(run_id):
        """Report the configured run state, or fail like an unavailable Jobs API."""
        if run_state == "error":
            raise RuntimeError("jobs API unavailable")
        return run_state or "TERMINATED"

    result = _run(spark, monkeypatch, [config, *configs], aliases=aliases, run_state=state)
    assert reason in _decisions(result)["1"]
    confirmed = _run(
        spark,
        monkeypatch,
        [config, *configs],
        aliases=aliases,
        run_state=state,
        cleanup_confirm="true",
    )
    assert confirmed["deleted"] == [] and spark.log == []


def test_retention_counts_from_receipt_creation(monkeypatch):
    """A reference younger than the retention period stays."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    _prepare(spark, config, "1", created=NOW - timedelta(days=10))
    assert _decisions(_run(spark, monkeypatch, [config])) == {"1": "younger than 90 days"}
    result = _run(spark, monkeypatch, [config], reference_retention_days="7")
    assert _decisions(result) == {"1": "delete"}


def test_foreign_or_tampered_tables_are_skipped_not_deleted(monkeypatch):
    """A matching name without our owner, or a changed receipt, is reported and left alone."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    foreign = _prepare(spark, config, "1")
    spark.tables[foreign]["owner"] = "someone.else"
    tampered = _prepare(spark, config, "2")
    spark.tables[tampered]["rows"][0]["digest"] = "0" * 64
    result = _run(spark, monkeypatch, [config], cleanup_confirm="true")
    assert {item["receipt"] for item in result["skipped"]} == {foreign, tampered}
    assert result["deleted"] == [] and spark.log == []


def test_partial_failure_keeps_the_receipt_and_resumes_only_after_an_intent(monkeypatch):
    """A failed drop records the failure; a later run may finish only audited cleanups."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    old = _prepare(spark, config, "1")
    spark.fail_drop = old.replace("_receipt", "_seen")
    with pytest.raises(RuntimeError, match="drop failed"):
        _run(spark, monkeypatch, [config], cleanup_confirm="true")
    assert old in spark.tables
    assert spark.log[-1] == f"AUDIT delete failed {old}"
    spark.fail_drop = None
    result = _run(spark, monkeypatch, [config], cleanup_confirm="true")
    assert result["deleted"] == [old] and old not in spark.tables


def test_incomplete_reference_without_an_intent_is_left_for_review(monkeypatch):
    """A missing child without a cleanup record is not ours to finish."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    old = _prepare(spark, config, "1")
    del spark.tables[old.replace("_receipt", "_train")]
    result = _run(spark, monkeypatch, [config], cleanup_confirm="true")
    assert result["skipped"][0]["reason"].startswith("incomplete")
    assert old in spark.tables


def test_plan_is_rechecked_before_deleting(monkeypatch):
    """A version enrolled after the first plan must not be deleted by the confirm run."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    _prepare(spark, config, "1")
    calls = []

    def inventory(spark, namespace):
        """Enroll version 1 on the second read, like a concurrent activation."""
        calls.append(1)
        version = "3" if len(calls) == 1 else "1"
        return [_config(model_version=version).payload()]

    monkeypatch.setattr(reference_cleanup, "load_enrolled_models", inventory)
    result = reference_cleanup.cleanup_references(
        spark,
        VALUES | {"cleanup_confirm": "true"},
        registry=lambda name: {},
        run_state=lambda run_id: "TERMINATED",
        now=NOW,
    )
    assert result["deleted"] == [] and len(calls) == 2


@pytest.mark.parametrize(
    "values",
    [
        {"cleanup_confirm": "yes"},
        {"cleanup_confirm": "True"},
        {"reference_retention_days": "0"},
        {"reference_retention_days": "3651"},
        {"reference_retention_days": "7.5"},
    ],
)
def test_invalid_settings_fail_before_reading_the_store(values):
    """A typo must not silently become a dry run or a different retention."""
    with pytest.raises(ValueError, match="cleanup_confirm|reference_retention_days"):
        reference_cleanup.cleanup_references(FakeSpark(), VALUES | values, now=NOW)


def test_registry_failure_aborts_instead_of_deleting(monkeypatch):
    """An unreadable alias set must stop cleanup rather than treat the version as unused."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    _prepare(spark, config, "1")

    def broken(name):
        """Fail like an unavailable registry."""
        raise RuntimeError("registry down")

    monkeypatch.setattr(
        reference_cleanup, "load_enrolled_models", lambda spark, ns: [config.payload()]
    )
    with pytest.raises(RuntimeError, match="registry down"):
        reference_cleanup.cleanup_references(
            spark, VALUES | {"cleanup_confirm": "true"}, registry=broken, now=NOW
        )
    assert spark.log == []


def test_job_dispatches_cleanup_even_when_monitoring_is_disabled(monkeypatch):
    """A project that turned monitoring off must still be able to remove its references."""

    from skyulf.integrations.databricks.jobs.monitoring.spark_monitoring_job import (
        run_project_monitoring_notebook,
    )

    values = VALUES | {"monitoring_enabled": "false", "monitoring_action": "cleanup_references"}
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(getAll=lambda: values),
        jobs=SimpleNamespace(taskValues=SimpleNamespace(set=lambda **kwargs: None)),
    )
    calls = []
    monkeypatch.setattr(
        reference_cleanup,
        "cleanup_references",
        lambda spark, given: calls.append(given) or {"status": "dry_run"},
    )
    assert run_project_monitoring_notebook(FakeSpark(), dbutils) == {"status": "dry_run"}
    assert calls == [values]


def test_a_modified_child_is_skipped_not_deleted(monkeypatch):
    """An append after creation breaks immutability, even with the right owner and id."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    old = _prepare(spark, config, "1")
    spark.tables[old.replace("_receipt", "_seen")]["version"] = 1
    result = _run(spark, monkeypatch, [config], cleanup_confirm="true")
    assert "modified after creation" in result["skipped"][0]["reason"]
    assert result["deleted"] == [] and spark.log == []


def test_a_child_replaced_during_deletion_stops_before_dropping_it(monkeypatch):
    """Identity is checked again right before each DROP, and the receipt survives."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    old = _prepare(spark, config, "1")
    seen = old.replace("_receipt", "_seen")

    def replace_seen(name):
        """Swap the seen table for a same-owner replacement while train is dropped."""
        if name.endswith("_train"):
            spark.tables[seen]["id"] = "replacement"

    spark.before_drop = replace_seen
    with pytest.raises(ValueError, match="identity differs"):
        _run(spark, monkeypatch, [config], cleanup_confirm="true")
    assert seen in spark.tables and old in spark.tables
    assert spark.log[-1] == f"AUDIT delete failed {old}"


def test_missing_request_table_aborts_as_unknown_state(monkeypatch):
    """Without the request table, unfinished retraining cannot be ruled out."""
    spark = FakeSpark()
    config = _config()
    _prepare(spark, config, "1")
    with pytest.raises(ValueError, match="retraining_requests is missing"):
        _run(spark, monkeypatch, [config], cleanup_confirm="true")
    assert spark.log == []


def test_a_foreign_audit_table_created_concurrently_stops_deletion(monkeypatch):
    """If another writer wins the audit CREATE, cleanup must not append or drop."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    old = _prepare(spark, config, "1")
    spark.audit_owner = "someone.else"
    with pytest.raises(ValueError, match="ownership differs"):
        _run(spark, monkeypatch, [config], cleanup_confirm="true")
    assert old in spark.tables
    assert not any(entry.startswith(("DROP", "AUDIT")) for entry in spark.log)


def test_reference_listing_is_bounded(monkeypatch):
    """More receipts than the planning limit fail instead of being silently truncated."""
    spark = FakeSpark()
    _store(spark)
    config = _config()
    _prepare(spark, config, "1")
    _prepare(spark, config, "2")
    monkeypatch.setattr(reference_cleanup, "MAX_REFERENCES", 1)
    with pytest.raises(ValueError, match="too many references"):
        _run(spark, monkeypatch, [config])


def test_a_receipt_replaced_while_it_is_read_is_skipped(monkeypatch):
    """The payload read must belong to the receipt id that the plan later rechecks."""

    def replaced_after_read(spark, receipt):
        """Replace the receipt with a same-owner table right after its payload is read."""

        def hook(name):
            """Swap the id once, on the receipt's first read."""
            if name == receipt and spark.tables[receipt]["id"] != "replacement":
                spark.tables[receipt]["id"] = "replacement"

        spark.after_read = hook

    for confirm in ("false", "true"):
        spark = FakeSpark()
        _store(spark)
        config = _config()
        old = _prepare(spark, config, "1")
        replaced_after_read(spark, old)
        result = _run(spark, monkeypatch, [config], cleanup_confirm=confirm)
        if confirm == "false":
            assert result["skipped"] == [
                {"receipt": old, "reason": f"Reference table identity differs: {old}."}
            ]
        assert result["deleted"] == [] and spark.log == []
