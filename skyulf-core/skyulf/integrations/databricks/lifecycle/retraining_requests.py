"""Persist retraining intent and serialize automatic requests per training job."""

import json
import re
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from ..observability.monitoring.monitoring_config import qualified_name
from ..observability.monitoring.monitoring_store import (
    OWNER,
    PROPERTY,
    ensure_owned_object,
    merge_with_retry,
)
from ..shared._contracts import table_name

REQUEST_SCHEMA = (
    "row_id STRING, job_id BIGINT, request_id STRING, status STRING, evidence_json STRING, "
    "created_at_ms BIGINT, submitted_at_ms BIGINT, run_id BIGINT"
)


class _RequestStore:
    """Keep one atomic job gate alongside immutable request identities and evidence."""

    def __init__(self, spark: Any, namespace: str) -> None:
        """Create an owned Serializable table without adopting existing foreign objects."""
        self.spark = spark
        self.name = qualified_name(f"{namespace}.retraining_requests")
        if not ensure_owned_object(spark, self.name):
            spark.sql(
                f"CREATE TABLE IF NOT EXISTS {table_name(self.name)} ({REQUEST_SCHEMA}) "
                f"USING DELTA TBLPROPERTIES ('{PROPERTY}' = '{OWNER}', "
                "'delta.isolationLevel' = 'Serializable')"
            ).collect()
        ensure_owned_object(spark, self.name)
        isolation = spark.sql(
            f"SHOW TBLPROPERTIES {table_name(self.name)} ('delta.isolationLevel')"
        ).first()
        if isolation is None or isolation["value"] != "Serializable":
            raise ValueError("Retraining requests require Serializable Delta isolation.")
        expected = spark.createDataFrame([], schema=REQUEST_SCHEMA).schema.simpleString()
        if spark.table(self.name).schema.simpleString() != expected:
            raise ValueError("Retraining requests table schema differs.")

    def read(self, key: str) -> dict | None:
        """Fail closed on duplicate identities rather than choosing an arbitrary gate."""
        rows = self.spark.table(self.name).where(f"row_id = '{key}'").limit(2).collect()
        if len(rows) > 1:
            raise ValueError("Retraining request identity is not unique.")
        return rows[0].asDict() if rows else None

    def _merge(self, row: dict, matched: str, *, insert: bool = True) -> None:
        """Materialize caller evidence as typed data and retry only Delta write conflicts."""
        view = f"skyulf_retraining_{uuid4().hex}"
        self.spark.createDataFrame([row], schema=REQUEST_SCHEMA).createOrReplaceTempView(view)
        missing = "WHEN NOT MATCHED THEN INSERT *" if insert else ""
        try:
            merge_with_retry(
                self.spark,
                f"MERGE INTO {table_name(self.name)} t USING {view} s "
                f"ON t.row_id = s.row_id {matched} {missing}",
            )
        finally:
            self.spark.catalog.dropTempView(view)

    def claim(self, row: dict, cutoff_ms: int) -> dict:
        """Atomically acquire an unused gate or replace a submitted request after cooldown."""
        gate = row | {"row_id": f"job:{row['job_id']}"}
        self._merge(
            gate,
            "WHEN MATCHED AND t.request_id <> s.request_id AND t.status = 'submitted' "
            f"AND t.submitted_at_ms <= {cutoff_ms} THEN UPDATE SET *",
        )
        current = self.read(gate["row_id"])
        if current is None:
            raise ValueError("Retraining job gate disappeared after claim.")
        return current

    def record(self, row: dict) -> None:
        """Save intent once so subsequent observations cannot replace original evidence."""
        self._merge(row, "")

    def complete(self, row: dict) -> None:
        """Persist the run receipt before allowing the gate to enter its cooldown period."""
        matched = (
            "WHEN MATCHED AND t.request_id = s.request_id AND t.status = 'intent' "
            "THEN UPDATE SET t.status = s.status, t.run_id = s.run_id, "
            "t.submitted_at_ms = s.submitted_at_ms"
        )
        self._merge(row, matched, insert=False)
        self._merge(row | {"row_id": f"job:{row['job_id']}"}, matched, insert=False)


def _validate_request(job_id: int, request_id: str, cooldown_hours: float, now: datetime) -> None:
    """Reject unsafe identities and time bounds before contacting any cloud service."""
    if type(job_id) is not int or not 0 < job_id < 2**63:
        raise ValueError("Retraining job_id must be a positive 64-bit integer.")
    if type(request_id) is not str or not re.fullmatch(r"[0-9a-f]{64}", request_id):
        raise ValueError("Retraining request_id must be a lowercase SHA-256 digest.")
    _validate_cooldown(cooldown_hours)
    if not isinstance(now, datetime) or now.utcoffset() != timedelta(0):
        raise ValueError("Retraining now must be an aware UTC datetime.")


def _validate_cooldown(value: float) -> None:
    """Keep the cutoff finite and representable by the Delta BIGINT timestamp field."""
    if type(value) not in (int, float) or not 0 <= value <= (2**63 - 1) / 3600000:
        raise ValueError("Retraining cooldown_hours must be nonnegative and fit a 64-bit cutoff.")


def _result(status: str, row: dict) -> dict:
    """Expose the durable request and asynchronous run identifier to the parent task."""
    return {
        "status": status,
        "job_id": row["job_id"],
        "request_id": row["request_id"],
        "run_id": row["run_id"],
    }


def _blocking_gate(gate: dict | None, request_id: str, cutoff_ms: int) -> dict | None:
    """Distinguish an unresolved competing request from an ordinary job cooldown."""
    if gate is None or gate["request_id"] == request_id:
        return None
    if gate["status"] != "submitted":
        return _result("pending_request", gate)
    if gate["submitted_at_ms"] > cutoff_ms:
        return _result("cooldown", gate)
    return None


def _submit(workspace: Any, store: _RequestStore, row: dict, now_ms: int) -> dict:
    """Submit asynchronously; ambiguous failures deliberately leave a recoverable intent."""
    from databricks.sdk.service.jobs import QueueSettings  # noqa: PLC0415 - optional SDK

    store.record(row)
    response = workspace.jobs.run_now(
        job_id=row["job_id"],
        idempotency_token=row["request_id"],
        job_parameters={"lifecycle_action": "train"},
        queue=QueueSettings(enabled=True),
    ).response
    run_id = response.run_id
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("Retraining submission did not return a concrete run_id.")
    completed = row | {"status": "submitted", "run_id": run_id, "submitted_at_ms": now_ms}
    store.complete(completed)
    return _result("submitted", completed)


def _admit_request(
    workspace: Any, store: _RequestStore, row: dict, cutoff_ms: int, *, preview_only: bool = False
) -> dict:
    """Recheck the atomic claim after advisory cooldown and active-run checks."""
    gate = store.read(f"job:{row['job_id']}")
    request_id = row["request_id"]
    blocked = _blocking_gate(gate, request_id, cutoff_ms)
    if blocked is not None:
        return blocked
    retry = gate is not None and gate["request_id"] == request_id
    if not retry:
        active = workspace.jobs.list_runs(job_id=row["job_id"], active_only=True)
        if next(iter(active), None) is not None:
            return _result("active_run", row)
    if preview_only:
        return _result("ready", row)
    claimed = store.claim(row, cutoff_ms)
    if claimed["request_id"] != request_id:
        return _blocking_gate(claimed, request_id, cutoff_ms) or _result("pending_request", claimed)
    if claimed["status"] == "submitted":
        return _result("already_submitted", claimed)
    return claimed


def submit_retraining(
    spark: Any,
    workspace: Any,
    *,
    namespace: str,
    job_id: int,
    request_id: str,
    evidence: dict,
    cooldown_hours: float,
    now: datetime,
    preview_only: bool = False,
) -> dict:
    """Claim a durable job gate and submit exactly one token for this model/data identity.

    Include the target job in the caller's request digest. A pending request can be
    retried with the same identity after an ambiguous API failure; different requests
    stay blocked until that intent has a saved submission receipt or operator recovery.
    Queueing preserves this one request if a manual run races the active-run check;
    queued runs are included in the active guard for subsequent observations.
    Preview mode returns eligibility without claiming a gate or submitting a run;
    it is advisory, so callers must repeat these guards when actually submitting.
    """
    _validate_request(job_id, request_id, cooldown_hours, now)
    qualified_name(f"{namespace}.retraining_requests")
    evidence_json = json.dumps(evidence, sort_keys=True, allow_nan=False)
    settings = workspace.jobs.get(job_id=job_id).settings
    if settings is None or settings.max_concurrent_runs != 1:
        raise ValueError("Retraining job max_concurrent_runs must equal 1.")
    store = _RequestStore(spark, namespace)
    key = f"request:{job_id}:{request_id}"
    saved = store.read(key)
    if saved is not None and saved["status"] == "submitted":
        if not preview_only:
            store.complete(saved)
        return _result("already_submitted", saved)
    now_ms = int(now.timestamp() * 1000)
    cutoff_ms = max(-(2**63), now_ms - int(cooldown_hours * 3600000))
    row = saved or {
        "row_id": key,
        "job_id": job_id,
        "request_id": request_id,
        "status": "intent",
        "evidence_json": evidence_json,
        "created_at_ms": now_ms,
        "submitted_at_ms": None,
        "run_id": None,
    }
    claimed = _admit_request(workspace, store, row, cutoff_ms, preview_only=preview_only)
    if claimed["status"] != "intent":
        return claimed
    return _submit(workspace, store, claimed | {"row_id": key}, now_ms)
