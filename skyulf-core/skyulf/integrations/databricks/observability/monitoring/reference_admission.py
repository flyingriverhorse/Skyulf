"""Non-expiring store admission for reference creation, activation and deletion."""

import asyncio
import json
import random
import re
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from ...data.delta_io.delta import table_identity
from ...shared._contracts import table_name
from .monitoring_config import qualified_name
from .monitoring_store import ensure_owned_object

CONTROL_ROW = "control:reference_mutation"
_WAIT_TIMEOUT_S = 600
_DATA_CONFLICTS = {
    base + suffix
    for base in (
        "DELTA_CONCURRENT_APPEND",
        "DELTA_CONCURRENT_DELETE_READ",
        "DELTA_CONCURRENT_DELETE_DELETE",
    )
    for suffix in (
        "",
        ".ALLOTTED_TIME_EXCEEDED",
        ".ROW_LEVEL_CHANGES",
        ".WHOLE_TABLE_READ",
        ".WITHOUT_HINT",
        ".WITH_PARTITION_HINT",
    )
} | {
    "DELTA_CONCURRENT_WRITE",
    "DELTA_CONCURRENT_DELETE_READ.EMPTY_READ_PREDICATES",
    "DELTA_CONCURRENT_DELETE_DELETE.EMPTY_READ_PREDICATES",
}


def is_reference_control_row(row: Any) -> bool:
    """Recognize only a structurally valid control row, never a retraining request."""
    values = dict(row) if isinstance(row, Mapping) else row.asDict()
    if values.get("row_id") != CONTROL_ROW:
        return False
    if any(values.get(key) is not None for key in ("job_id", "run_id", "submitted_at_ms")):
        raise ValueError("Reference admission control row contains job fields.")
    _validate_control_state(values)
    evidence = json.loads(values["evidence_json"])
    _validate_control_evidence(evidence)
    return True


def _validate_control_evidence(evidence: dict) -> None:
    """Validate authority and recovery provenance without accepting unknown fields."""
    if (
        set(evidence) != {"version", "control_table_id", "holder"}
        or type(evidence["version"]) is not int
        or evidence["version"] != 1
        or not isinstance(evidence["control_table_id"], str)
        or not evidence["control_table_id"]
    ):
        raise ValueError("Reference admission control evidence differs.")
    if evidence["holder"] is not None:
        if not isinstance(evidence["holder"], dict) or set(evidence["holder"]) != {
            "job_run_id",
            "task_key",
            "workspace_host",
        }:
            raise ValueError("Reference admission holder differs.")
        _holder(**evidence["holder"])


def _validate_control_state(row: dict) -> None:
    """Reject malformed ownership and timestamps before accepting or ignoring a row."""
    status, owner = row.get("status"), row.get("request_id")
    valid_owner = isinstance(owner, str) and re.fullmatch(r"[0-9a-f]{32}", owner)
    valid_state = (status == "free" and owner is None) or (status == "held" and valid_owner)
    created = row.get("created_at_ms")
    if not valid_state or type(created) is not int or created <= 0:
        raise ValueError("Reference admission control state differs.")


def _contention_error(row: dict) -> RuntimeError:
    """Name the verified competing owner when the original wait budget is exhausted."""
    holder = json.loads(row["evidence_json"])["holder"]
    state = "is held by another operation" if row["status"] == "held" else "changed before readback"
    return RuntimeError(
        f"Reference mutation admission {state}; "
        f"wait budget exhausted. owner={row['request_id']}, holder={holder}. "
        "Inspect the holder before explicit reference recovery."
    )


def _is_data_conflict(error: Exception) -> bool:
    """Match exact Delta abort conditions, excluding structural and unknown suffixes."""
    try:
        conditions = _structured_conditions(error)
        if conditions:
            return all(
                isinstance(value, str) and value == conditions[0] and value in _DATA_CONFLICTS
                for value in conditions
            )
        return _connect_condition(error) in _DATA_CONFLICTS
    except Exception:  # noqa: BLE001 - unreadable metadata cannot authorize a retry
        return False


def _structured_conditions(error: Exception) -> list[Any]:
    """Keep every supplied condition authoritative, including malformed metadata."""
    conditions = []
    for name in ("getCondition", "getErrorClass"):
        getter = getattr(error, name, None)
        if getter is None:
            continue
        value = getter()  # Non-callable or failing getters must not permit message fallback.
        if value is not None:
            conditions.append(value)
    return conditions


def _connect_condition(error: Exception) -> str | None:
    """Read the typed header when Delta Connect's converter drops structured fields."""
    if type(error).__module__ != "delta.connect.exceptions":
        return None
    base = {
        "ConcurrentAppendException": "DELTA_CONCURRENT_APPEND",
        "ConcurrentDeleteReadException": "DELTA_CONCURRENT_DELETE_READ",
        "ConcurrentDeleteDeleteException": "DELTA_CONCURRENT_DELETE_DELETE",
        "ConcurrentWriteException": "DELTA_CONCURRENT_WRITE",
    }.get(type(error).__name__)
    match = re.match(r"\[([A-Z_]+(?:\.[A-Z_]+)?)\](?=\s|$)", str(error))
    if match is None or match[1].split(".", 1)[0] != base:
        return None
    return match[1]


class _Authority:
    """Use the existing Serializable request table without changing its schema."""

    def __init__(self, spark: Any, namespace: str) -> None:
        """Require a provisioned, owned authority; never adopt an unrelated table."""
        from ...lifecycle.retraining_requests import REQUEST_SCHEMA  # noqa: PLC0415 - module cycle

        self.spark = spark
        self.name = qualified_name(f"{namespace}.retraining_requests")
        self.quoted = table_name(self.name)
        if not ensure_owned_object(spark, self.name):
            raise ValueError("Initialize the monitoring store before reference mutations.")
        isolation = spark.sql(f"SHOW TBLPROPERTIES {self.quoted} ('delta.isolationLevel')").first()
        if isolation is None or isolation["value"] != "Serializable":
            raise ValueError("Reference mutations require Serializable Delta isolation.")
        expected = spark.createDataFrame([], schema=REQUEST_SCHEMA).schema.simpleString()
        if spark.table(self.name).schema.simpleString() != expected:
            raise ValueError("Reference admission table schema differs.")
        self.identity = table_identity(spark, self.name)

    def read(self) -> dict | None:
        """Bind every ownership read to the same physical singleton authority."""
        self._check_identity()
        rows = self.spark.table(self.name).where(f"row_id = '{CONTROL_ROW}'").limit(2).collect()
        if len(rows) > 1:
            raise ValueError("Reference admission identity is not unique.")
        self._check_identity()
        if not rows:
            return None
        row = dict(rows[0]) if isinstance(rows[0], Mapping) else rows[0].asDict()
        is_reference_control_row(row)
        if json.loads(row["evidence_json"])["control_table_id"] != self.identity:
            raise RuntimeError("Reference admission control table identity changed.")
        return row

    def _check_identity(self) -> None:
        """Never clear or reuse an authority that was replaced under the same name."""
        if table_identity(self.spark, self.name) != self.identity:
            raise RuntimeError("Reference admission table identity changed.")

    def require(self, owner: str) -> None:
        """Fail closed if ownership disappeared before a protected mutation."""
        row = self.read()
        if row is None or row["status"] != "held" or row["request_id"] != owner:
            raise RuntimeError("Reference mutation admission is no longer held by this owner.")

    def acquire(self, owner: str, holder: dict | None, wait_timeout_s: float) -> None:
        """Retry verified contention within one budget; never retry an ambiguous write."""
        deadline = time.monotonic() + wait_timeout_s
        delay = 0.5
        evidence = json.dumps(
            {"version": 1, "control_table_id": self.identity, "holder": holder}, sort_keys=True
        )
        while True:
            self._wait_available(max(0, deadline - time.monotonic()))
            try:
                self._claim(owner, evidence)
            except Exception as error:  # noqa: BLE001 - Spark exposes structured Delta aborts
                if not self._wait_conflict(error, owner, deadline, delay):
                    raise
                delay = min(delay * 2, 5.0)
                continue
            row = self.read()
            if row is None:
                raise RuntimeError("Reference mutation admission is no longer held by this owner.")
            if row["request_id"] == owner:
                return
            if time.monotonic() >= deadline:
                raise _contention_error(row)

    def _wait_conflict(self, error: Exception, owner: str, deadline: float, delay: float) -> bool:
        """Wait only after a definite abort with consistent identity and ownership evidence."""
        if not _is_data_conflict(error):
            return False
        row = self.read()
        remaining = deadline - time.monotonic()
        if remaining <= 0 or (row is not None and row["request_id"] == owner):
            return False
        time.sleep(min(remaining, random.uniform(delay / 2, delay)))
        return time.monotonic() < deadline

    def _claim(self, owner: str, evidence: str) -> None:
        """Attempt one conditional write and propagate any uncertain SQL failure."""
        self.spark.sql(
            f"MERGE INTO {self.quoted} t USING (SELECT '{CONTROL_ROW}' AS row_id, "
            "CAST(NULL AS BIGINT) AS job_id, :owner AS request_id, 'held' AS status, "
            ":evidence AS evidence_json, CAST(:created_at AS BIGINT) AS created_at_ms, "
            "CAST(NULL AS BIGINT) AS submitted_at_ms, CAST(NULL AS BIGINT) AS run_id) s "
            "ON t.row_id = s.row_id "
            "WHEN MATCHED AND t.status = 'free' AND t.request_id IS NULL THEN UPDATE SET * "
            "WHEN NOT MATCHED THEN INSERT *",
            args={"owner": owner, "evidence": evidence, "created_at": time.time_ns() // 1000000},
        ).collect()

    def _wait_available(self, timeout: float) -> None:
        """Wait for a verified owner to release, without expiring or stealing its token."""
        deadline = time.monotonic() + timeout
        delay = 0.5
        while (row := self.read()) is not None and row["status"] == "held":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _contention_error(row)
            time.sleep(min(remaining, random.uniform(delay / 2, delay)))
            delay = min(delay * 2, 5.0)

    def release(self, owner: str) -> None:
        """Clear only the verified token; never steal, expire or repair another owner."""
        self.require(owner)
        self.spark.sql(
            f"UPDATE {self.quoted} SET status = 'free', request_id = NULL "
            f"WHERE row_id = '{CONTROL_ROW}' AND status = 'held' AND request_id = :owner",
            args={"owner": owner},
        ).collect()
        row = self.read()
        if row is None or row["request_id"] == owner:
            raise RuntimeError("Reference admission release could not be verified.")


def _execution_identity() -> tuple[int, int | None]:
    """Do not inherit reentrant authority in another thread or copied asyncio task."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), id(task) if task is not None else None


@dataclass
class _Held:
    """Keep nested synchronous operations inside their original verified owner."""

    spark: Any
    namespace: str
    authority: _Authority
    owner: str
    execution: tuple[int, int | None]
    mutation_started: bool = False


_HELD: ContextVar[tuple[_Held, ...]] = ContextVar("skyulf_reference_admission", default=())


def _current(spark: Any, namespace: str) -> _Held | None:
    """Find a nested owner only in this execution and this store."""
    return next(
        (
            item
            for item in _HELD.get()
            if item.spark is spark
            and item.namespace == namespace
            and item.execution == _execution_identity()
        ),
        None,
    )


def require_reference_mutation(spark: Any, namespace: str, owner: str) -> None:
    """Verify ownership and mark a remote mutation as started before issuing its write."""
    held = _current(spark, namespace)
    if held is None or held.owner != owner:
        raise RuntimeError("Reference mutation requires this operation's held admission.")
    held.authority.require(owner)
    held.mutation_started = True


@contextmanager
def hold_reference_mutation(
    spark: Any,
    namespace: str,
    *,
    owner: str | None = None,
    job_run_id: int | None = None,
    task_key: str | None = None,
    workspace_host: str | None = None,
    wait_timeout_s: float | None = None,
) -> Iterator[str]:
    """Serialize participating reference writers and model activations in one store.

    Every participating project must use the same monitoring namespace. Direct
    SDK alias mutations must run inside this context too. External UI/MLflow
    writers, authority edits and table replacement require exclusive-writer ACLs;
    this cooperative protocol cannot protect against writers bypassing it.

    Ownership never expires. After a driver crash, mutation failure or uncertain acquisition,
    an operator must prove that the owner can no longer mutate before clearing
    its exact token. A failed claim never enters the body. Nested calls in the
    same thread/task reuse and revalidate the token without releasing it early.
    A validation failure before any marked write releases normally. Table writers
    must call ``require_reference_mutation`` immediately before writing; Skyulf
    registry alias writes are marked automatically by the scoped registry guard.
    """
    from ....mlflow.lifecycle.promotion import (  # noqa: PLC0415 - integration boundary
        guard_alias_mutations,
    )

    holder = _holder(job_run_id, task_key, workspace_host)
    timeout = _wait_timeout(wait_timeout_s)
    held = _current(spark, namespace)
    if held is not None:
        token = held.owner if owner is None else owner
        if token != held.owner:
            raise RuntimeError("Reference mutation requires this operation's held admission.")
        held.authority.require(token)
        yield token
        held.authority.require(token)
        return
    if owner is not None:
        raise RuntimeError("Cannot reuse a reference owner outside its original operation.")
    authority = _Authority(spark, namespace)
    token = uuid4().hex
    authority.acquire(token, holder, timeout)
    held = _Held(spark, namespace, authority, token, _execution_identity())
    state = _HELD.set((*_HELD.get(), held))
    try:
        with guard_alias_mutations(lambda: require_reference_mutation(spark, namespace, token)):
            yield token
    except BaseException as error:
        # A failed remote call may still complete after its response is lost.
        # Releasing here would allow cleanup to overlap that unconfirmed mutation.
        if not held.mutation_started:
            _release_failed_validation(held, error)
        raise
    else:
        authority.release(token)
    finally:
        _HELD.reset(state)


def _release_failed_validation(held: _Held, original: BaseException) -> None:
    """Keep the original failure visible if cleaning up its unused authority also fails."""
    try:
        held.authority.release(held.owner)
    except Exception as error:  # noqa: BLE001 - preserve the original failure and recovery evidence
        original.add_note(f"Reference admission release requires inspection: {error}")


def _wait_timeout(value: float | None) -> float:
    """Bound waiting without adding lease expiry or automatic stale-owner recovery."""
    if value is None:
        return _WAIT_TIMEOUT_S
    if type(value) not in {int, float} or not 0 <= value <= 3600:
        raise ValueError("Reference wait_timeout_s must be from 0 to 3600 seconds.")
    return value


@contextmanager
def defer_reference_mutation(spark: Any, namespace: str, **holder: Any) -> Iterator[None]:
    """Acquire at the first alias write, after expensive evaluation and validation.

    Keep the acquired scope through phase completion and failure restoration.
    A phase that never writes an alias neither creates nor waits for a lock.
    """
    from ....mlflow.lifecycle.promotion import (  # noqa: PLC0415 - integration boundary
        guard_alias_mutations,
    )

    owner = None
    with ExitStack() as stack:

        def before_write() -> None:
            """Enter once and mark every alias call before its network mutation."""
            nonlocal owner
            if owner is None:
                owner = stack.enter_context(hold_reference_mutation(spark, namespace, **holder))
            require_reference_mutation(spark, namespace, owner)

        stack.enter_context(guard_alias_mutations(before_write))
        yield


def _workspace_host(value: str | None) -> str:
    """Canonicalize a workspace origin without permitting credentials or URL paths."""
    if not isinstance(value, str):
        raise ValueError("Reference holder requires a workspace HTTPS origin.")
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.path not in {"", "/"}
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Reference holder requires a workspace HTTPS origin.")
    return f"https://{parts.netloc.lower()}"


def _holder(
    job_run_id: int | None, task_key: str | None, workspace_host: str | None
) -> dict | None:
    """Keep manual callers explicit while requiring complete job recovery provenance."""
    if job_run_id is None and task_key is None and workspace_host is None:
        return None
    if type(job_run_id) is not int or not 0 < job_run_id < 2**63:
        raise ValueError("Reference holder job_run_id must be a positive integer.")
    if not isinstance(task_key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", task_key):
        raise ValueError("Reference holder task_key must identify the running task.")
    return {
        "job_run_id": job_run_id,
        "task_key": task_key,
        "workspace_host": _workspace_host(workspace_host),
    }


def reference_mutation_holder(values: dict) -> dict:
    """Parse all-or-none job widgets for lock ownership and explicit recovery."""
    run_id = values.get("job_run_id")
    if isinstance(run_id, str) and re.fullmatch(r"[1-9][0-9]*", run_id):
        run_id = int(run_id)
    return _holder(run_id, values.get("task_key"), values.get("workspace_host")) or {}


def _terminated_run(workspace: Any, holder: dict) -> dict:
    """Require the exact workspace/run/task with completed execution cleanup."""
    if _workspace_host(workspace.config.host) != holder["workspace_host"]:
        raise ValueError("Reference recovery workspace differs from the saved holder.")
    run = workspace.jobs.get_run(run_id=holder["job_run_id"])
    state = _run_state(run)
    completed = state == "TERMINATED" or (state == "INTERNAL_ERROR" and _completed_tasks(run))
    if run.run_id != holder["job_run_id"] or not completed:
        raise ValueError(
            "Reference recovery requires TERMINATED execution or an INTERNAL_ERROR "
            "run with completed task termination evidence."
        )
    _verify_holder_task(run, holder, state)
    return {**holder, "life_cycle_state": state, "end_time": getattr(run, "end_time", None)}


def _verify_holder_task(run: Any, holder: dict, state: str | None) -> None:
    """Verify the holder actually ran, rather than mistaking a skipped task for a writer."""
    tasks = [task for task in run.tasks or [] if task.task_key == holder["task_key"]]
    if len(tasks) != 1:
        raise ValueError("Reference recovery task differs from the saved holder.")
    if state == "INTERNAL_ERROR" and _run_state(tasks[0]) != "TERMINATED":
        raise ValueError("Reference recovery requires a completed executing holder task.")


def _run_state(run: Any) -> str | None:
    """Read either an SDK enum or the corresponding Jobs API string."""
    state = getattr(getattr(run, "state", None), "life_cycle_state", None)
    return getattr(state, "value", state)


def _completed_tasks(run: Any) -> bool:
    """Do not treat an internal error alone as proof that workers have stopped."""
    end = getattr(run, "end_time", None)
    if type(end) is not int or end <= 0 or not run.tasks:
        return False
    for task in run.tasks:
        state, end = _run_state(task), getattr(task, "end_time", None)
        if state == "SKIPPED":
            continue
        if state != "TERMINATED" or type(end) is not int or end <= 0:
            return False
    return True


def recover_reference_mutation(
    spark: Any, namespace: str, *, workspace: Any, owner: str, confirm: bool = False
) -> dict:
    """Inspect or explicitly release a failed job's exact owner, with durable audit.

    First stop the original operation and inspect uncertain remote writes. Run
    this from a separate operator context, passing the saved owner token. The
    default only verifies readiness. ``confirm=True`` attests that no original
    writer can resume; the Jobs API must also show completed execution in the
    recorded workspace. An INTERNAL_ERROR parent additionally requires end times
    and completed tasks, not merely a terminal state name. Manual holders require verified
    process shutdown and operator SQL recovery, never an elapsed-time timeout.
    """
    from .reference_cleanup import append_reference_recovery  # noqa: PLC0415 - cycle boundary

    if type(confirm) is not bool:
        raise ValueError("Reference recovery confirm must be a bool.")
    if _current(spark, namespace) is not None:
        raise ValueError("Cannot recover reference admission from its active owner context.")
    authority = _Authority(spark, namespace)
    authority.require(owner)
    row = authority.read()
    assert row is not None  # require() verified the same held singleton
    holder = json.loads(row["evidence_json"])["holder"]
    if holder is None:
        raise ValueError("Manual reference owner requires verified operator recovery.")
    evidence = _terminated_run(workspace, holder) | {"control_table_id": authority.identity}
    result = {"status": "ready", "owner": owner, "evidence": evidence}
    if not confirm:
        return result
    append_reference_recovery(spark, namespace, owner=owner, status="intent", evidence=evidence)
    authority.release(owner)
    append_reference_recovery(spark, namespace, owner=owner, status="released", evidence=evidence)
    return result | {"status": "released"}
