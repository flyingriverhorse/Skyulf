"""Reference mutations serialize across jobs without an expiring owner lease."""

import asyncio
import contextvars
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.observability.monitoring import reference_admission as module


class Rows:
    """Expose only the Delta rows needed by the admission protocol."""

    def __init__(self, rows, schema="requests"):
        """Keep immutable snapshots so competing reads do not share row objects."""
        self.rows = [dict(row) for row in rows]
        self.schema = SimpleNamespace(simpleString=lambda: schema)

    def where(self, expression):
        """Filter the reserved row, leaving ordinary retraining records alone."""
        return Rows([r for r in self.rows if r["row_id"] == module.CONTROL_ROW])

    def limit(self, count):
        """Allow duplicate authority detection with a bounded read."""
        return Rows(self.rows[:count])

    def first(self):
        """Return one SQL metadata result."""
        return self.rows[0] if self.rows else None

    def collect(self):
        """Materialize a fresh snapshot."""
        return list(self.rows)


class Spark:
    """Model atomic conditional SQL changes; native Delta is verified separately."""

    def __init__(self):
        """Start with a provisioned empty authority and no owner."""
        self.rows = []
        self.identity = "table-1"
        self.isolation = "Serializable"
        self.owned = True
        self.exists = True
        self.schema = "requests"
        self.statements = []
        self.fail_acquire = False
        self.fail_release = False

    def table(self, name):
        """Read the current authority with its schema."""
        return Rows(self.rows, self.schema)

    def createDataFrame(self, data, schema):  # noqa: N802 - Spark API
        """Provide the expected schema without requiring local Java."""
        return Rows(data)

    def sql(self, statement, args=None):
        """Apply acquire/release atomically and expose ambiguous response failures."""
        self.statements.append(statement)
        args = args or {}
        if statement.startswith("SHOW TBLPROPERTIES"):
            return Rows([{"value": self.isolation}])
        if statement.startswith("MERGE"):
            old = next((r for r in self.rows if r["row_id"] == module.CONTROL_ROW), None)
            if old is None or old["status"] == "free":
                row = {
                    "row_id": module.CONTROL_ROW,
                    "job_id": None,
                    "request_id": args["owner"],
                    "status": "held",
                    "evidence_json": args["evidence"],
                    "created_at_ms": args["created_at"],
                    "submitted_at_ms": None,
                    "run_id": None,
                }
                self.rows = [r for r in self.rows if r["row_id"] != module.CONTROL_ROW] + [row]
            if self.fail_acquire:
                raise TimeoutError("acquire response lost")
        if statement.startswith("UPDATE"):
            if self.fail_release:
                raise TimeoutError("release response lost")
            for row in self.rows:
                if row["row_id"] == module.CONTROL_ROW and row["request_id"] == args["owner"]:
                    row.update(request_id=None, status="free")
        return Rows([])


@pytest.fixture
def authority(monkeypatch):
    """Replace metadata IO while exercising real claim/readback/release control flow."""
    spark = Spark()

    def ownership(spark, name):
        """Reject foreign tables exactly before mutation."""
        if not spark.owned:
            raise ValueError("ownership differs")
        return spark.exists

    monkeypatch.setattr(module, "ensure_owned_object", ownership)
    monkeypatch.setattr(module, "table_identity", lambda spark, name: spark.identity)
    monkeypatch.setattr(module, "_WAIT_TIMEOUT_S", 0)
    return spark


def test_guard_holds_nested_calls_and_releases_only_outer_owner(authority):
    """Nested preparation must not release the outer lifecycle mutation's authority."""
    with module.hold_reference_mutation(authority, "ops.monitoring") as owner:
        with module.hold_reference_mutation(authority, "ops.monitoring") as nested:
            assert nested == owner
            module.require_reference_mutation(authority, "ops.monitoring", owner)
        assert authority.rows[0]["request_id"] == owner
    assert authority.rows[0]["status"] == "free"
    assert sum(s.startswith("MERGE") for s in authority.statements) == 1


def test_independent_context_cannot_acquire_until_owner_releases(authority):
    """A concurrent cleanup cannot enter while a producer or activation holds the store."""

    def compete():
        """Run in another logical invocation rather than inheriting reentrant state."""
        with module.hold_reference_mutation(authority, "ops.monitoring"):
            pytest.fail("competing context entered")

    with (
        module.hold_reference_mutation(authority, "ops.monitoring"),
        pytest.raises(RuntimeError, match="held"),
    ):
        contextvars.Context().run(compete)
    with module.hold_reference_mutation(authority, "ops.monitoring") as owner:
        assert authority.rows[0]["request_id"] == owner


@pytest.mark.parametrize("failure", ["missing", "foreign", "isolation", "schema", "duplicate"])
def test_invalid_authority_fails_before_conditional_write(authority, failure):
    """Unknown authority must never authorize reference deletion."""
    if failure == "missing":
        authority.exists = False
    elif failure == "foreign":
        authority.owned = False
    elif failure == "isolation":
        authority.isolation = "WriteSerializable"
    elif failure == "schema":
        authority.schema = "wrong"
    else:
        authority.rows = [{"row_id": module.CONTROL_ROW}] * 2
    with (
        pytest.raises((ValueError, RuntimeError)),
        module.hold_reference_mutation(authority, "ops.monitoring"),
    ):
        pytest.fail("invalid authority entered")
    assert not any(s.startswith(("MERGE", "UPDATE")) for s in authority.statements)


def test_ambiguous_claim_is_not_stolen_or_automatically_cleared(authority):
    """A lost response may leave a live driver; elapsed time is not recovery evidence."""
    authority.fail_acquire = True
    with pytest.raises(TimeoutError), module.hold_reference_mutation(authority, "ops.monitoring"):
        pytest.fail("uncertain claim entered")
    authority.fail_acquire = False
    authority.rows[0]["created_at_ms"] = 1
    with (
        pytest.raises(RuntimeError, match="held"),
        module.hold_reference_mutation(authority, "ops.monitoring"),
    ):
        pytest.fail("stale owner stolen")
    assert authority.rows[0]["status"] == "held"


@pytest.mark.parametrize("mutation", ["identity", "owner", "deleted"])
def test_lost_authority_cannot_validate_or_clear_another_owner(authority, mutation):
    """Changed table identity or token invalidates the held mutation immediately."""
    with (
        pytest.raises(RuntimeError),
        module.hold_reference_mutation(authority, "ops.monitoring") as owner,
    ):
        if mutation == "identity":
            authority.identity = "replaced"
        elif mutation == "owner":
            authority.rows[0]["request_id"] = "b" * 32
        else:
            authority.rows.clear()
        with pytest.raises(RuntimeError):
            module.require_reference_mutation(authority, "ops.monitoring", owner)
    assert not any(s.startswith("UPDATE") for s in authority.statements)


def test_body_failure_keeps_authority_until_operator_verifies_remote_completion(authority):
    """A failed remote call may still commit after its client loses the response."""
    with (
        pytest.raises(ValueError, match="operation failed"),
        module.hold_reference_mutation(authority, "ops.monitoring") as owner,
    ):
        module.require_reference_mutation(authority, "ops.monitoring", owner)
        raise ValueError("operation failed")
    assert authority.rows[0]["status"] == "held"
    with (
        pytest.raises(RuntimeError, match="held"),
        module.hold_reference_mutation(authority, "ops.monitoring"),
    ):
        pytest.fail("uncertain mutation was released")


@pytest.mark.parametrize("nested", [False, True])
def test_validation_failure_releases_unmodified_authority(authority, nested):
    """A read-only validation failure must not block unrelated projects in the store."""
    with (
        pytest.raises(ValueError, match="invalid settings"),
        module.hold_reference_mutation(authority, "ops.monitoring"),
    ):
        if nested:
            with module.hold_reference_mutation(authority, "ops.monitoring"):
                raise ValueError("invalid settings")
        raise ValueError("invalid settings")
    assert authority.rows[0]["status"] == "free"


def test_nested_mutation_failure_retains_outer_authority(authority):
    """A nested remote write marks the original owner uncertain when it fails."""
    with (
        pytest.raises(TimeoutError),
        module.hold_reference_mutation(authority, "ops.monitoring"),
        module.hold_reference_mutation(authority, "ops.monitoring") as owner,
    ):
        module.require_reference_mutation(authority, "ops.monitoring", owner)
        raise TimeoutError("remote write response lost")
    assert authority.rows[0]["status"] == "held"


def test_guard_row_is_distinguished_from_actual_unfinished_retraining(authority):
    """Cleanup may ignore the control row only after validating its complete structure."""
    with module.hold_reference_mutation(authority, "ops.monitoring"):
        row = authority.rows[0]
        assert module.is_reference_control_row(row)
        assert not module.is_reference_control_row(row | {"row_id": "request:17:abc"})
        with pytest.raises(ValueError):
            module.is_reference_control_row(row | {"run_id": 17})
        with pytest.raises(ValueError):
            module.is_reference_control_row(row | {"evidence_json": json.dumps({})})
    assert module.is_reference_control_row(authority.rows[0])


HOLDER: dict[str, Any] = {
    "job_run_id": 17,
    "task_key": "cleanup_references",
    "workspace_host": "https://test.cloud.databricks.com",
}


def _failed_job(authority):
    """Retain an exact owner after an uncertain job mutation."""
    with (
        pytest.raises(TimeoutError),
        module.hold_reference_mutation(authority, "ops.monitoring", **HOLDER) as owner,
    ):
        module.require_reference_mutation(authority, "ops.monitoring", owner)
        raise TimeoutError("response lost")
    return authority.rows[0]["request_id"]


@pytest.mark.parametrize("state", ["RUNNING", "TERMINATING", "INTERNAL_ERROR", None])
def test_recovery_refuses_run_without_completed_execution_cleanup(authority, state):
    """A failed but still terminating job cannot authorize breaking its reference lock."""
    owner = _failed_job(authority)
    workspace = Mock()
    workspace.config.host = HOLDER["workspace_host"]
    workspace.jobs.get_run.return_value = SimpleNamespace(
        run_id=17, state=SimpleNamespace(life_cycle_state=state), tasks=[]
    )
    with pytest.raises(ValueError, match="TERMINATED"):
        module.recover_reference_mutation(
            authority, "ops.monitoring", workspace=workspace, owner=owner, confirm=True
        )
    assert authority.rows[0]["request_id"] == owner


def test_recovery_audits_verified_job_before_releasing_exact_token(authority, monkeypatch):
    """Only the confirmed owner in the same workspace can be released, with a durable intent."""
    from skyulf.integrations.databricks.observability.monitoring import reference_cleanup

    owner = _failed_job(authority)
    events = []

    def audit(spark, namespace, **record):
        """Check audit ordering against the actual persisted control state."""
        events.append((record["status"], authority.rows[0]["status"]))

    monkeypatch.setattr(reference_cleanup, "append_reference_recovery", audit)
    workspace = Mock()
    workspace.config.host = HOLDER["workspace_host"]
    task = SimpleNamespace(
        task_key="cleanup_references", state=SimpleNamespace(life_cycle_state="TERMINATED")
    )
    workspace.jobs.get_run.return_value = SimpleNamespace(
        run_id=17, state=SimpleNamespace(life_cycle_state="TERMINATED"), tasks=[task]
    )
    preview = module.recover_reference_mutation(
        authority, "ops.monitoring", workspace=workspace, owner=owner
    )
    assert preview["status"] == "ready" and events == []
    result = module.recover_reference_mutation(
        authority, "ops.monitoring", workspace=workspace, owner=owner, confirm=True
    )
    assert result["status"] == "released"
    assert events == [("intent", "held"), ("released", "free")]
    assert authority.rows[0]["request_id"] is None


@pytest.mark.parametrize("wrong", ["workspace", "owner", "run", "task"])
def test_recovery_requires_saved_holder_identity(authority, wrong):
    """A matching numeric run ID in another workspace or task is not sufficient evidence."""
    owner = _failed_job(authority)
    workspace = Mock()
    workspace.config.host = (
        HOLDER["workspace_host"] if wrong != "workspace" else "https://other.databricks.com"
    )
    workspace.jobs.get_run.return_value = SimpleNamespace(
        run_id=18 if wrong == "run" else 17,
        state=SimpleNamespace(life_cycle_state="TERMINATED"),
        tasks=[SimpleNamespace(task_key="other" if wrong == "task" else "cleanup_references")],
    )
    with pytest.raises((ValueError, RuntimeError)):
        module.recover_reference_mutation(
            authority,
            "ops.monitoring",
            workspace=workspace,
            owner="c" * 32 if wrong == "owner" else owner,
            confirm=True,
        )
    assert authority.rows[0]["request_id"] == owner


def test_holder_widgets_are_complete_or_absent():
    """Generated jobs supply recoverable provenance without accepting partial widget bindings."""
    assert module.reference_mutation_holder({}) == {}
    assert module.reference_mutation_holder(HOLDER | {"job_run_id": "17"}) == HOLDER
    with pytest.raises(ValueError):
        module.reference_mutation_holder({"job_run_id": "17"})


def test_waiting_writer_enters_after_verified_owner_releases(authority, monkeypatch):
    """Shared-store contention waits for release rather than failing unrelated projects."""
    original = _failed_job(authority)
    waited = []

    def release_after_wait(seconds):
        """Simulate the original operation finishing during bounded backoff."""
        waited.append(seconds)
        authority.rows[0].update(status="free", request_id=None)

    monkeypatch.setattr(module.time, "sleep", release_after_wait)
    with module.hold_reference_mutation(authority, "ops.monitoring", wait_timeout_s=30) as owner:
        assert owner != original and authority.rows[0]["request_id"] == owner
    assert waited and authority.rows[0]["status"] == "free"


def test_wait_timeout_names_holder_and_does_not_steal(authority):
    """A blocked operator can identify the exact job needing recovery without losing its token."""
    original = _failed_job(authority)
    with (
        pytest.raises(RuntimeError, match="job_run_id.*17.*task_key.*cleanup_references"),
        module.hold_reference_mutation(authority, "ops.monitoring", wait_timeout_s=0),
    ):
        pytest.fail("owner stolen")
    assert authority.rows[0]["request_id"] == original


@pytest.mark.parametrize("budget", [0, 30])
@pytest.mark.parametrize("released_before_read", [False, True])
def test_claim_race_uses_remaining_wait_budget(
    authority, monkeypatch, budget, released_before_read
):
    """A successful no-op MERGE is known contention, not an ambiguous failed write."""
    original_sql = authority.sql
    raced = []
    waited = []

    def competing_sql(statement, args=None):
        """Let another writer win after the free-row read and before our claim."""
        first_claim = statement.startswith("MERGE") and not raced
        if first_claim:
            assert args is not None
            raced.append(True)
            original_sql(statement, args | {"owner": "f" * 32})
        result = original_sql(statement, args)
        if first_claim and released_before_read:
            authority.rows[0].update(status="free", request_id=None)
        return result

    def release_after_wait(seconds):
        """Complete the competing writer without expiring or stealing its claim."""
        waited.append(seconds)
        authority.rows[0].update(status="free", request_id=None)

    monkeypatch.setattr(authority, "sql", competing_sql)
    monkeypatch.setattr(module.time, "sleep", release_after_wait)
    if budget == 0:
        with (
            pytest.raises(RuntimeError, match="wait budget exhausted.*owner="),
            module.hold_reference_mutation(authority, "ops.monitoring", wait_timeout_s=budget),
        ):
            pytest.fail("competing owner was stolen")
        expected = None if released_before_read else "f" * 32
        assert not waited and authority.rows[0]["request_id"] == expected
    else:
        with module.hold_reference_mutation(
            authority, "ops.monitoring", wait_timeout_s=budget
        ) as owner:
            assert owner != "f" * 32 and authority.rows[0]["request_id"] == owner
        assert bool(waited) is not released_before_read
        assert authority.rows[0]["status"] == "free"


def test_repeated_claim_races_share_one_wait_deadline(authority, monkeypatch):
    """Repeated winners cannot reset the caller's bounded waiting period."""
    original_sql = authority.sql
    clock = [0.0]

    def competing_sql(statement, args=None):
        """Have a new caller win every claim without returning a transport error."""
        if statement.startswith("MERGE"):
            assert args is not None
            original_sql(statement, args | {"owner": "f" * 32})
        return original_sql(statement, args)

    def release_after_wait(seconds):
        """Advance the shared clock and finish only the actual current owner."""
        clock[0] += seconds
        authority.rows[0].update(status="free", request_id=None)

    monkeypatch.setattr(authority, "sql", competing_sql)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", release_after_wait)
    monkeypatch.setattr(module.random, "uniform", lambda low, high: high)
    with (
        pytest.raises(RuntimeError, match="wait budget exhausted"),
        module.hold_reference_mutation(authority, "ops.monitoring", wait_timeout_s=2),
    ):
        pytest.fail("competing owner was stolen")
    assert clock[0] == 2 and authority.rows[0]["request_id"] == "f" * 32


@pytest.mark.parametrize("method", ["getCondition", "getErrorClass"])
@pytest.mark.parametrize(
    "condition",
    [
        "DELTA_CONCURRENT_APPEND",
        "DELTA_CONCURRENT_DELETE_READ",
        "DELTA_CONCURRENT_DELETE_DELETE",
        "DELTA_CONCURRENT_WRITE",
        "DELTA_CONCURRENT_APPEND.WITHOUT_HINT",
        "DELTA_CONCURRENT_DELETE_READ.ROW_LEVEL_CHANGES",
        "DELTA_CONCURRENT_DELETE_DELETE.WHOLE_TABLE_READ",
        "DELTA_CONCURRENT_DELETE_READ.ALLOTTED_TIME_EXCEEDED",
    ],
)
def test_definite_delta_conflict_retries_within_budget(authority, monkeypatch, method, condition):
    """Known aborted data writes can wait without mistaking a lost response for rejection."""
    original_sql, attempts, waits = authority.sql, [], []
    error = RuntimeError("definite data conflict")
    setattr(error, method, lambda: condition)

    def conflicting_sql(statement, args=None):
        """Reject the first claim without committing any ownership row."""
        if statement.startswith("MERGE"):
            attempts.append(True)
            if len(attempts) == 1:
                raise error
        return original_sql(statement, args)

    monkeypatch.setattr(authority, "sql", conflicting_sql)
    monkeypatch.setattr(module.time, "sleep", waits.append)
    with module.hold_reference_mutation(authority, "ops.monitoring", wait_timeout_s=30) as owner:
        assert authority.rows[0]["request_id"] == owner
    assert len(attempts) == 2 and waits and authority.rows[0]["status"] == "free"


@pytest.mark.parametrize(
    "condition",
    [
        None,
        "DELTA_METADATA_CHANGED",
        "DELTA_PROTOCOL_CHANGED",
        "DELTA_CONCURRENT_TRANSACTION",
        "DELTA_CONCURRENT_APPEND.METADATA_CHANGE",
        "DELTA_CONCURRENT_DELETE_READ.PROTOCOL_CHANGE",
        "DELTA_CONCURRENT_DELETE_DELETE.WHOLE_TABLE_REPLACE",
        "DELTA_CONCURRENT_APPEND.NEW_VARIANT",
        "DELTA_CONCURRENT_WRITE.NEW_VARIANT",
    ],
)
def test_unknown_or_structural_conflict_is_never_retried(authority, monkeypatch, condition):
    """Neither message text nor a shared error prefix proves a safe data-only retry."""
    attempts = []
    error = RuntimeError("DELTA_CONCURRENT_APPEND: response lost")
    if condition is not None:
        monkeypatch.setattr(error, "getCondition", lambda: condition, raising=False)

    def failing_sql(statement, args=None):
        """Reject only the claim so authority metadata remains inspectable."""
        attempts.append(statement)
        raise error

    original_sql = authority.sql
    monkeypatch.setattr(
        authority,
        "sql",
        lambda statement, args=None: (
            failing_sql(statement, args)
            if statement.startswith("MERGE")
            else original_sql(statement, args)
        ),
    )
    with (
        pytest.raises(RuntimeError) as caught,
        module.hold_reference_mutation(authority, "ops.monitoring", wait_timeout_s=30),
    ):
        pytest.fail("unknown claim result admitted")
    assert caught.value is error and len(attempts) == 1 and authority.rows == []


def test_conflict_with_own_token_is_not_assumed_uncommitted(authority, monkeypatch):
    """Contradictory readback preserves uncertain ownership for explicit recovery."""
    original_sql, attempts = authority.sql, []
    error = RuntimeError("purported non-commit with an owned row")
    monkeypatch.setattr(error, "getCondition", lambda: "DELTA_CONCURRENT_APPEND", raising=False)

    def contradictory_sql(statement, args=None):
        """Commit the claim before reporting a supposedly definite rejection."""
        result = original_sql(statement, args)
        if statement.startswith("MERGE"):
            attempts.append(True)
            raise error
        return result

    monkeypatch.setattr(authority, "sql", contradictory_sql)
    with (
        pytest.raises(RuntimeError) as caught,
        module.hold_reference_mutation(authority, "ops.monitoring", wait_timeout_s=30),
    ):
        pytest.fail("contradictory claim admitted")
    assert caught.value is error and len(attempts) == 1 and authority.rows[0]["status"] == "held"


def test_repeated_delta_conflicts_keep_original_deadline(authority, monkeypatch):
    """Repeated definite rejections cannot spin or renew the total wait budget."""
    original_sql, clock, waits = authority.sql, [0.0], []
    error = RuntimeError("definite data conflict")
    monkeypatch.setattr(error, "getCondition", lambda: "DELTA_CONCURRENT_APPEND", raising=False)

    def conflicting_sql(statement, args=None):
        """Keep rejecting the claim without a committed control row."""
        if statement.startswith("MERGE"):
            raise error
        return original_sql(statement, args)

    def advance(seconds):
        """Make bounded backoff observable without slowing down the test."""
        clock[0] += seconds
        waits.append(seconds)

    monkeypatch.setattr(authority, "sql", conflicting_sql)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", advance)
    monkeypatch.setattr(module.random, "uniform", lambda low, high: high)
    with (
        pytest.raises(RuntimeError) as caught,
        module.hold_reference_mutation(authority, "ops.monitoring", wait_timeout_s=2),
    ):
        pytest.fail("uncommitted claim admitted")
    assert caught.value is error and clock[0] == 2 and waits == [0.5, 1.0, 0.5]


def _connect_error(name, message, condition=None, *, module_name="delta.connect.exceptions"):
    """Model Delta Connect's message-only conversion without requiring a Spark runtime."""
    error_type = type(
        name,
        (RuntimeError,),
        {
            "__module__": module_name,
            "getCondition": lambda self: condition,
            "getErrorClass": lambda self: condition,
        },
    )
    return error_type(message)


@pytest.mark.parametrize(
    ("name", "condition"),
    [
        ("ConcurrentAppendException", "DELTA_CONCURRENT_APPEND.ROW_LEVEL_CHANGES"),
        ("ConcurrentDeleteReadException", "DELTA_CONCURRENT_DELETE_READ.WHOLE_TABLE_READ"),
        ("ConcurrentDeleteDeleteException", "DELTA_CONCURRENT_DELETE_DELETE"),
        ("ConcurrentWriteException", "DELTA_CONCURRENT_WRITE"),
    ],
)
def test_delta_connect_message_only_abort_retries(authority, monkeypatch, name, condition):
    """Delta Connect can drop structured fields while preserving the exact typed abort header."""
    error = _connect_error(name, f"[{condition}] Transaction conflict detected.")
    original_sql, attempts = authority.sql, []

    def conflicting_sql(statement, args=None):
        """Reject one real claim attempt, then let the normal ownership path finish."""
        if statement.startswith("MERGE"):
            attempts.append(True)
            if len(attempts) == 1:
                raise error
        return original_sql(statement, args)

    monkeypatch.setattr(authority, "sql", conflicting_sql)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    with module.hold_reference_mutation(authority, "ops.monitoring", wait_timeout_s=30):
        assert authority.rows[0]["status"] == "held"
    assert len(attempts) == 2 and authority.rows[0]["status"] == "free"


@pytest.mark.parametrize(
    ("name", "message", "condition", "module_name"),
    [
        (
            "ConcurrentAppendException",
            "[DELTA_CONCURRENT_APPEND.ROW_LEVEL_CHANGES] aborted",
            None,
            "user.code",
        ),
        (
            "RuntimeError",
            "[DELTA_CONCURRENT_APPEND.ROW_LEVEL_CHANGES] aborted",
            None,
            "delta.connect.exceptions",
        ),
        (
            "MetadataChangedException",
            "[DELTA_CONCURRENT_APPEND.ROW_LEVEL_CHANGES] aborted",
            None,
            "delta.connect.exceptions",
        ),
        (
            "ConcurrentAppendException",
            "transport error: [DELTA_CONCURRENT_APPEND]",
            None,
            "delta.connect.exceptions",
        ),
        (
            "ConcurrentAppendException",
            "[DELTA_CONCURRENT_APPEND]suffix",
            None,
            "delta.connect.exceptions",
        ),
        (
            "ConcurrentAppendException",
            "[DELTA_CONCURRENT_APPEND.METADATA_CHANGE] aborted",
            None,
            "delta.connect.exceptions",
        ),
        (
            "ConcurrentAppendException",
            "[DELTA_CONCURRENT_APPEND.PROTOCOL_CHANGE] aborted",
            None,
            "delta.connect.exceptions",
        ),
        (
            "ConcurrentAppendException",
            "[DELTA_CONCURRENT_APPEND.WHOLE_TABLE_REPLACE] aborted",
            None,
            "delta.connect.exceptions",
        ),
        (
            "ConcurrentAppendException",
            "[DELTA_CONCURRENT_APPEND.UNKNOWN] aborted",
            None,
            "delta.connect.exceptions",
        ),
        (
            "ConcurrentAppendException",
            "[DELTA_CONCURRENT_DELETE_READ] aborted",
            None,
            "delta.connect.exceptions",
        ),
        (
            "ConcurrentAppendException",
            "[DELTA_CONCURRENT_APPEND] aborted",
            "DELTA_METADATA_CHANGED",
            "delta.connect.exceptions",
        ),
    ],
)
def test_delta_connect_fallback_rejects_unverified_headers(name, message, condition, module_name):
    """A typed message fallback must not turn structural changes or arbitrary text into retries."""
    error = _connect_error(name, message, condition, module_name=module_name)
    assert not module._is_data_conflict(error)


@pytest.mark.parametrize(
    ("modern", "legacy"),
    [
        (None, "UNKNOWN"),
        (None, ""),
        (None, []),
        ("", None),
        (123, None),
        ("DELTA_CONCURRENT_APPEND", "DELTA_CONCURRENT_WRITE"),
    ],
)
def test_delta_connect_fallback_keeps_all_structured_metadata_authoritative(modern, legacy):
    """A valid-looking header cannot override either getter's contradictory metadata."""
    error = _connect_error("ConcurrentAppendException", "[DELTA_CONCURRENT_APPEND] aborted")
    error.getCondition = lambda: modern
    error.getErrorClass = lambda: legacy
    assert not module._is_data_conflict(error)


@pytest.mark.parametrize("name", ["getCondition", "getErrorClass"])
@pytest.mark.parametrize("broken", ["raises", "not_callable"])
def test_delta_connect_fallback_rejects_broken_metadata_getters(name, broken):
    """Metadata access failure must preserve uncertainty instead of enabling a retry."""
    error = _connect_error("ConcurrentAppendException", "[DELTA_CONCURRENT_APPEND] aborted")

    def unreadable():
        """Reproduce a remote exception whose metadata cannot be read."""
        raise RuntimeError("metadata unavailable")

    setattr(error, name, unreadable if broken == "raises" else "invalid")
    assert not module._is_data_conflict(error)


def test_delta_connect_fallback_rejects_leading_whitespace():
    """Only the exact exception header, not a nested or indented message, is recognized."""
    error = _connect_error("ConcurrentAppendException", " [DELTA_CONCURRENT_APPEND] aborted")
    assert not module._is_data_conflict(error)


def test_lazy_scope_does_not_acquire_for_evaluation_only(authority):
    """Expensive evaluation and rejected quality gates must not lock the shared store."""
    with (
        pytest.raises(ValueError, match="quality gate"),
        module.defer_reference_mutation(authority, "ops.monitoring"),
    ):
        assert authority.rows == []
        raise ValueError("quality gate")
    assert authority.rows == [] and authority.statements == []


def test_lazy_scope_keeps_first_alias_owner_until_phase_finishes(authority):
    """An alias mutation and its phase failure cleanup share one verified owner."""
    from skyulf.integrations.mlflow.lifecycle import promotion

    with module.defer_reference_mutation(authority, "ops.monitoring"):
        callback = promotion._BEFORE_ALIAS_WRITE.get()
        assert callback is not None
        callback()
        owner = authority.rows[0]["request_id"]
        callback()
        assert authority.rows[0]["request_id"] == owner
    assert authority.rows[0]["status"] == "free"


@pytest.mark.parametrize("incomplete", [None, "parent_end", "task_end", "task_state", "skipped"])
def test_internal_error_recovery_requires_completed_tasks(authority, incomplete):
    """A parent internal error is recoverable only with positive task termination evidence."""
    owner = _failed_job(authority)
    task = SimpleNamespace(
        task_key="cleanup_references",
        end_time=0 if incomplete == "task_end" else 100,
        state=SimpleNamespace(
            life_cycle_state={"task_state": "INTERNAL_ERROR", "skipped": "SKIPPED"}.get(
                incomplete, "TERMINATED"
            )
        ),
    )
    workspace = Mock()
    workspace.config.host = HOLDER["workspace_host"]
    workspace.jobs.get_run.return_value = SimpleNamespace(
        run_id=17,
        end_time=0 if incomplete == "parent_end" else 100,
        state=SimpleNamespace(life_cycle_state="INTERNAL_ERROR"),
        tasks=[task],
    )
    if incomplete is None:
        result = module.recover_reference_mutation(
            authority, "ops.monitoring", workspace=workspace, owner=owner
        )
        assert result["status"] == "ready"
        assert result["evidence"]["life_cycle_state"] == "INTERNAL_ERROR"
    else:
        with pytest.raises(ValueError, match="completed"):
            module.recover_reference_mutation(
                authority, "ops.monitoring", workspace=workspace, owner=owner
            )
    assert authority.rows[0]["request_id"] == owner


def test_copied_async_context_cannot_reuse_another_tasks_owner(authority):
    """Async child tasks inherit context variables but must acquire their own authority."""

    async def child():
        """Attempt ownership from the automatically copied child context."""
        with (
            pytest.raises(RuntimeError, match="held"),
            module.hold_reference_mutation(authority, "ops.monitoring"),
        ):
            pytest.fail("child reused parent authority")

    async def parent():
        """Keep the original owner alive while the child is rejected."""
        with module.hold_reference_mutation(authority, "ops.monitoring") as owner:
            await asyncio.create_task(child())
            assert authority.rows[0]["request_id"] == owner

    asyncio.run(parent())
    assert authority.rows[0]["status"] == "free"


def test_release_failure_preserves_original_validation_error(authority):
    """Recovery details must not hide the actual reason a model operation failed."""
    authority.fail_release = True
    with (
        pytest.raises(ValueError, match="invalid model settings") as error,
        module.hold_reference_mutation(authority, "ops.monitoring"),
    ):
        raise ValueError("invalid model settings")
    assert "release response lost" in " ".join(error.value.__notes__)
    assert authority.rows[0]["status"] == "held"
