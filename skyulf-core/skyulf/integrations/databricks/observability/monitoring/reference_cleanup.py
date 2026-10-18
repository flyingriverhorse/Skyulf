"""Retire prepared Spark monitoring references that no monitored model still needs.

Legacy references have train/seen/source tables; format 2 combines train/seen and shares
identical source populations. ``cleanup_references`` lists one project's references and, only with
``cleanup_confirm=true``, drops those that are provably owned, unneeded and old:

- Ownership is the reference identity recomputed from the project's own inventory rows and
  the receipt's evidence; a shared name prefix proves nothing.
- A version stays while any enabled monitor in the store uses it, a registry alias points at
  it, a performance policy names it as baseline, a retraining request is unfinished, or it is
  younger than ``reference_retention_days``.
- Unreadable inventory, registry or request state aborts the run. A foreign, tampered or
  incomplete reference is reported and left alone.
- Shared sources remain while any surviving receipt in any project references them. Unknown
  receipt state protects format-2 references. The last retiring receipt retains its shared
  source until its own deletion starts; the receipt stays until all dependent drops succeed.
- Deletion recomputes the plan first. It writes an audit intent, drops the population
  tables and then the receipt, and records the result. Population-drop failures retain the
  receipt; only an earlier intent permits resuming a partial deletion. The final audit write
  can fail after all tables are gone. Inspect table state after failure; there is no rollback
  or atomicity across tables. A non-expiring store admission excludes participating
  preparation and activation writers throughout confirmed cleanup. External writers must
  respect the documented exclusive-writer boundary.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from ...data.delta_io.delta import history
from ...shared._contracts import table_name
from .monitoring_config import MonitorConfig, json_digest, qualified_name
from .monitoring_registration import monitoring_destination
from .monitoring_sources import read_snapshot
from .monitoring_store import OWNER, PROPERTY, ensure_owned_object, load_enrolled_models
from .reference_admission import (
    hold_reference_mutation,
    is_reference_control_row,
    reference_mutation_holder,
    require_reference_mutation,
)
from .reference_layout import shared_source_table_name

CHILDREN = ("train", "seen", "source")
AUDIT_TABLE = "reference_cleanup_audit"
AUDIT_SCHEMA = (
    "audit_id STRING, cleanup_id STRING, reference_id STRING, table_name STRING, "
    "action STRING, status STRING, reason STRING, recorded_at TIMESTAMP"
)
TERMINAL_RUN_STATES = frozenset({"TERMINATED", "SKIPPED", "INTERNAL_ERROR"})
MAX_REFERENCES = 10000
MAX_ROWS = 100000
_PREFIX, _SUFFIX = "monitor_ref_", "_receipt"


@dataclass(frozen=True)
class Reference:
    """One owned receipt with verified population tables."""

    receipt: str
    reference_id: str
    model_name: str
    model_version: str
    tables: tuple[str, ...]
    ids: dict[str, str]
    created_at: datetime
    size_bytes: int
    shared_source: str | None = None


@dataclass(frozen=True)
class Protection:
    """Store-wide facts that keep a reference."""

    enrolled: frozenset[tuple[str, str]]
    baselines: frozenset[tuple[str, str]]
    aliases: dict[str, dict[str, str]]
    unresolved: tuple[str, ...]


def cleanup_settings(values: dict) -> tuple[bool, int]:
    """Parse ``cleanup_confirm`` and ``reference_retention_days`` strictly.

    Args:
        values: Monitoring job widget values.

    Returns:
        Whether to delete, and the retention period in days.

    Raises:
        ValueError: If confirm is not exactly ``true``/``false`` or days is not 1 to 3650.
    """
    confirm = values.get("cleanup_confirm", "false")
    if confirm not in {"true", "false"}:
        raise ValueError("cleanup_confirm must be true or false.")
    days = str(values.get("reference_retention_days", "90"))
    if not re.fullmatch(r"[1-9][0-9]{0,3}", days) or int(days) > 3650:
        raise ValueError("reference_retention_days must be an integer from 1 to 3650.")
    return confirm == "true", int(days)


def cleanup_references(
    spark: Any,
    values: dict,
    *,
    registry: Callable[[str], dict[str, str]] | None = None,
    run_state: Callable[[int], str] | None = None,
    now: datetime | None = None,
) -> dict:
    """List, and with ``cleanup_confirm=true`` drop, one project's unneeded references.

    Args:
        spark: Active Spark session.
        values: Job widgets: store location, ``monitoring_environment``,
            ``monitoring_project``, ``cleanup_confirm`` and ``reference_retention_days``.
        registry: Returns a registered model's aliases as alias to version; defaults to MLflow.
        run_state: Returns a job run's life-cycle state; defaults to the Databricks Jobs API.
        now: Evaluation time, defaulting to the current UTC time.

    Returns:
        The decision, reason and size of every owned reference, skipped references with a
        reason, and the receipts deleted by this run.

    Raises:
        ValueError: If settings or store state are invalid.
    """
    confirm, days = cleanup_settings(values)
    namespace = monitoring_destination(values)
    if namespace is None:
        return {"status": "disabled"}
    context = _Context(
        spark,
        namespace,
        values,
        registry or _mlflow_aliases,
        run_state or _job_run_state,
        now or datetime.now(UTC),
        days,
    )
    if not confirm:
        return context.plan() | {"status": "dry_run", "deleted": []}
    with hold_reference_mutation(spark, namespace, **reference_mutation_holder(values)) as owner:
        planned = context.plan()
        rechecked = context.plan()
        allowed = _deletable(planned) & _deletable(rechecked)
        references = [item for item in rechecked["references"] if item["receipt"] in allowed]
        return rechecked | {
            "status": "deleted",
            "deleted": _delete(spark, namespace, references, owner),
        }


def _deletable(plan: dict) -> set[str]:
    """Receipts a plan allows deleting."""
    return {item["receipt"] for item in plan["references"] if item["decision"] == "deletable"}


@dataclass(frozen=True)
class _Context:
    """Everything one planning pass reads."""

    spark: Any
    namespace: str
    values: dict
    registry: Callable[[str], dict[str, str]]
    run_state: Callable[[int], str]
    now: datetime
    days: int

    def plan(self) -> dict:
        """Read inventory, references and protections once and decide every reference."""
        configs = [
            MonitorConfig.from_dict(item)
            for item in load_enrolled_models(self.spark, self.namespace)
        ]
        owners = _owners(configs, self.namespace, self.values)
        intents = _audit_intents(self.spark, self.namespace)
        references, skipped, source_users = _references(self.spark, self.namespace, owners, intents)
        protection = Protection(
            enrolled=_enrolled(configs),
            baselines=_baselines(configs),
            aliases={
                model: {alias: str(version) for alias, version in self.registry(model).items()}
                for model in sorted({reference.model_name for reference in references})
            },
            unresolved=_unresolved(self.spark, self.namespace, self.run_state),
        )
        decided = [_decision(item, protection, self.now, self.days) for item in references]
        _shared_deletions(decided, source_users, bool(skipped))
        return {"namespace": self.namespace, "references": decided, "skipped": skipped}


def _owners(configs: list[MonitorConfig], namespace: str, values: dict) -> list[MonitorConfig]:
    """Requested project monitors, one per reference identity, with serving stripped."""
    wanted = (values["monitoring_environment"], values["monitoring_project"])
    owners = {}
    for config in configs:
        if (config.environment, config.project) == wanted and (
            config.reference_namespace == namespace
        ):
            stripped = replace(config, serving_endpoint=None)
            owners[stripped.monitor_id] = stripped
    return list(owners.values())


def _references(spark, namespace, owners, intents) -> tuple[list[Reference], list[dict], dict]:
    """Load the owned receipts in the store; report invalid ones without touching them."""
    from .spark.spark_monitoring_reference import reference_table_name  # noqa: PLC0415

    rows = spark.sql(f"SHOW TABLES IN {table_name(namespace)} LIKE '{_PREFIX}*{_SUFFIX}'")
    names = sorted(
        qualified_name(f"{namespace}.{row['tableName']}")
        for row in rows.limit(MAX_REFERENCES + 1).collect()
    )
    if len(names) > MAX_REFERENCES:
        raise ValueError("Monitoring store has too many references to plan at once.")
    references, skipped, source_users = [], [], {}
    for name in names:
        try:
            receipt, receipt_id = _receipt(spark, name)
            source = _shared_source(receipt, namespace)
            if source is not None:
                source_users.setdefault(source, set()).add(name)
            owner = next(
                (
                    o
                    for o in owners
                    if reference_table_name(o, receipt["evidence"], "receipt") == name
                ),
                None,
            )
            if owner is not None:
                references.append(_reference(spark, name, owner, receipt, receipt_id, intents))
        except ValueError as error:
            skipped.append({"receipt": name, "reason": str(error)})
    return references, skipped, source_users


def _shared_source(receipt: dict, namespace: str) -> str | None:
    """Verify a shared source binding before allowing it to contribute reverse references."""
    layout = receipt.get("format", 1)
    if layout == 1:
        return None
    if layout != 2:
        raise ValueError("Unsupported prepared monitoring reference format.")
    identity = receipt["source_identity"]
    if identity["format"] != 2 or identity["table_id"] != receipt["source_table_id"]:
        raise ValueError("Shared source identity differs from its receipt.")
    name = shared_source_table_name(namespace, identity)
    record = receipt["tables"]["source"]
    if record.get("name") != name or record.get("version") != 0 or not record.get("id"):
        raise ValueError("Shared source binding differs from its receipt.")
    return name


def _shared_deletions(references: list[dict], source_users: dict, unknown: bool) -> None:
    """Drop a shared source only with its final retiring receipt, under the store admission."""
    deleting = _deletable({"references": references})
    remaining = {name: users.copy() for name, users in source_users.items()}
    for item in references:
        source = item["shared_source"]
        if source is None:
            continue
        if unknown:
            item.update(decision="protected", reason="shared source reverse references are unknown")
        remaining[source].discard(item["receipt"])
        if item["decision"] != "deletable" or source_users[source] - deleting or remaining[source]:
            item["tables"] = [name for name in item["tables"] if name != source]


def _receipt(spark: Any, name: str) -> tuple[dict, str]:
    """Read one owned, never-modified receipt, verify its digest and return its Delta id.

    The id is checked again after the read, so the payload belongs to that table.
    """
    receipt_id = str(immutable_detail(spark, name)["id"])
    rows = read_snapshot(spark, name, 0).limit(2).collect()
    if len(rows) != 1:
        raise ValueError("Reference receipt is invalid.")
    receipt = json.loads(rows[0]["payload"])
    if rows[0]["digest"] != json_digest(receipt):
        raise ValueError("Reference receipt digest differs.")
    immutable_detail(spark, name, receipt_id)
    return receipt, receipt_id


def _reference(spark, name, owner, receipt, receipt_id, intents) -> Reference:
    """Verify the receipt's population tables against the names the owner would use."""
    from .spark.spark_monitoring_reference import reference_table_name  # noqa: PLC0415

    evidence = receipt["evidence"]
    reference_id = name.rsplit(".", 1)[1].removeprefix(_PREFIX).removesuffix(_SUFFIX)
    detail = immutable_detail(spark, name, receipt_id)
    existing, size = [], int(detail["sizeInBytes"])
    ids = {name: str(detail["id"])}
    source = _shared_source(receipt, str(owner.reference_namespace))
    keys = CHILDREN if source is None else ("population", "source")
    for key in keys:
        child_name = (
            source if key == "source" and source else reference_table_name(owner, evidence, key)
        )
        child = _child(spark, child_name, receipt["tables"].get(key) or {})
        if child is not None:
            existing.append(child[0])
            ids[child[0]] = child[1]
            size += child[2]
        elif reference_id not in intents:
            raise ValueError(f"incomplete reference without a cleanup intent: {key} is missing.")
    return Reference(
        receipt=name,
        reference_id=reference_id,
        model_name=str(evidence["model_name"]).lower(),
        model_version=str(evidence["model_version"]),
        tables=tuple(existing),
        ids=ids,
        created_at=detail["createdAt"],
        size_bytes=size,
        shared_source=source,
    )


def _child(spark, name, record) -> tuple[str, str, int] | None:
    """Return an existing population table's name, id and size, or None when dropped."""
    if record.get("name") != name:
        raise ValueError(f"Reference table name differs from its receipt: {name}.")
    if record.get("version") != 0:
        raise ValueError(f"Reference receipt records a mutable version: {name}.")
    if not spark.catalog.tableExists(name):
        return None
    detail = immutable_detail(spark, name, str(record.get("id")))
    return name, str(detail["id"]), int(detail["sizeInBytes"])


def immutable_detail(spark: Any, name: str, expected_id: str | None = None) -> Any:
    """Require an owned, append-only Delta table still at its first version.

    Args:
        spark: Active Spark session.
        name: Fully qualified table name.
        expected_id: The Delta table id it must still have, when known.

    Returns:
        The ``DESCRIBE DETAIL`` row: id, createdAt, sizeInBytes and properties.

    Raises:
        ValueError: If the table is missing, foreign, replaced, mutable or modified.
    """
    if not ensure_owned_object(spark, name):
        raise ValueError(f"Reference table disappeared: {name}.")
    detail = spark.sql(f"DESCRIBE DETAIL {table_name(name)}").first()
    if detail is None or detail["format"] != "delta":
        raise ValueError(f"Reference table is not Delta: {name}.")
    if expected_id is not None and str(detail["id"]) != expected_id:
        raise ValueError(f"Reference table identity differs: {name}.")
    if (detail["properties"] or {}).get("delta.appendOnly") != "true":
        raise ValueError(f"Reference table is not append-only: {name}.")
    latest = history(spark, name, limit=1).first()
    if latest is None or int(latest["version"]) != 0:
        raise ValueError(f"Reference table was modified after creation: {name}.")
    return detail


def _enrolled(configs: list[MonitorConfig]) -> frozenset[tuple[str, str]]:
    """Model versions that an enabled batch or serving monitor anywhere in the store uses."""
    return frozenset(
        (config.model_name.lower(), str(config.model_version))
        for config in configs
        if config.enabled and config.model_version is not None
    )


def _baselines(configs: list[MonitorConfig]) -> frozenset[tuple[str, str]]:
    """Model versions that any performance policy in the store uses as baseline."""
    found = set()
    for config in configs:
        version = (config.performance_policy or {}).get("baseline", {}).get("model_version")
        if version is not None:
            found.add((config.model_name.lower(), str(version)))
    return frozenset(found)


def _unresolved(spark: Any, namespace: str, run_state: Callable[[int], str]) -> tuple[str, ...]:
    """Requests whose train run may still read references: intent, or a non-terminal run."""
    name = qualified_name(f"{namespace}.retraining_requests")
    if not ensure_owned_object(spark, name):
        raise ValueError("retraining_requests is missing, so retraining dependencies are unknown.")
    rows = _bounded(spark.table(name), name)
    requests = (row for row in rows if not is_reference_control_row(row))
    return tuple(
        sorted(str(row["request_id"]) for row in requests if not _finished(row, run_state))
    )


def _finished(row: Any, run_state: Callable[[int], str]) -> bool:
    """A submitted request is finished only when its run is verifiably terminal."""
    if row["status"] != "submitted" or row["run_id"] is None:
        return False
    try:
        return run_state(int(row["run_id"])) in TERMINAL_RUN_STATES
    except Exception:  # noqa: BLE001 - an unknown run state must protect, never delete
        return False


def _decision(reference: Reference, protection: Protection, now: datetime, days: int) -> dict:
    """Return the first protection reason, or deletable."""
    reasons = (rule(reference, protection, now, days) for rule in _RULES)
    reason = next((found for found in reasons if found is not None), None)
    return {
        "receipt": reference.receipt,
        "reference_id": reference.reference_id,
        "model_name": reference.model_name,
        "model_version": reference.model_version,
        "tables": list(reference.tables),
        "ids": dict(reference.ids),
        "created_at": reference.created_at.isoformat(),
        "size_bytes": reference.size_bytes,
        "decision": "protected" if reason else "deletable",
        "reason": reason or f"unused and older than {days} days",
        "shared_source": reference.shared_source,
    }


def _requests_rule(reference, protection, now, days) -> str | None:
    """Keep everything while a retraining request may still read references."""
    if protection.unresolved:
        return f"unfinished retraining request {protection.unresolved[0]}"
    return None


def _enrolled_rule(reference, protection, now, days) -> str | None:
    """Keep a version that an enabled batch or serving monitor uses."""
    key = (reference.model_name, reference.model_version)
    return "enrolled model version" if key in protection.enrolled else None


def _alias_rule(reference, protection, now, days) -> str | None:
    """Keep a version that champion, challenger or any other alias points at."""
    aliases = protection.aliases.get(reference.model_name, {})
    names = sorted(name for name, version in aliases.items() if version == reference.model_version)
    return f"registry alias {names[0]}" if names else None


def _baseline_rule(reference, protection, now, days) -> str | None:
    """Keep a version that any performance policy in the store uses as baseline."""
    key = (reference.model_name, reference.model_version)
    return "performance baseline" if key in protection.baselines else None


def _retention_rule(reference, protection, now, days) -> str | None:
    """Keep a reference younger than the retention period."""
    created = reference.created_at
    created = created if created.tzinfo else created.replace(tzinfo=UTC)
    return f"younger than {days} days" if now - created < timedelta(days=days) else None


_RULES = (_requests_rule, _enrolled_rule, _alias_rule, _baseline_rule, _retention_rule)


def _audit_intents(spark: Any, namespace: str) -> set[str]:
    """Reference ids with a recorded cleanup intent, which may resume."""
    name = qualified_name(f"{namespace}.{AUDIT_TABLE}")
    if not ensure_owned_object(spark, name):
        return set()
    rows = _bounded(spark.table(name).select("reference_id", "status", "action"), name)
    return {
        str(row["reference_id"])
        for row in rows
        if row["status"] == "intent" and row["action"] == "delete"
    }


def _bounded(frame: Any, name: str) -> list:
    """Collect at most MAX_ROWS rows, failing instead of truncating."""
    rows = frame.limit(MAX_ROWS + 1).collect()
    if len(rows) > MAX_ROWS:
        raise ValueError(f"{name} has more rows than cleanup reads at once.")
    return rows


def _delete(spark: Any, namespace: str, references: list[dict], owner: str) -> list[str]:
    """Drop each reference's tables, receipt last, between audit intent and result rows."""
    if not references:
        return []
    require_reference_mutation(spark, namespace, owner)
    audit = _audit_table(spark, namespace)
    cleanup_id, deleted = uuid4().hex, []
    for item in references:
        require_reference_mutation(spark, namespace, owner)
        _record(spark, audit, cleanup_id, item, "intent", item["reason"])
        try:
            for name in [*item["tables"], item["receipt"]]:
                if name == item["shared_source"] and _source_still_used(spark, namespace, item):
                    continue
                immutable_detail(spark, name, item["ids"][name])
                require_reference_mutation(spark, namespace, owner)
                spark.sql(f"DROP TABLE {table_name(name)}")
        except Exception as error:
            _record(spark, audit, cleanup_id, item, "failed", str(error)[:1000])
            raise
        _record(spark, audit, cleanup_id, item, "deleted", item["reason"])
        deleted.append(item["receipt"])
    return deleted


def _audit_table(spark: Any, namespace: str) -> str:
    """Create or validate the same owned audit destination for retirement and guard recovery."""
    audit = qualified_name(f"{namespace}.{AUDIT_TABLE}")
    if not ensure_owned_object(spark, audit):
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {table_name(audit)} ({AUDIT_SCHEMA}) "
            f"USING DELTA TBLPROPERTIES ('{PROPERTY}' = '{OWNER}')"
        )
    expected = spark.createDataFrame([], AUDIT_SCHEMA).schema.simpleString()
    if not ensure_owned_object(spark, audit) or (
        spark.table(audit).schema.simpleString() != expected
    ):
        raise ValueError(f"Cleanup audit table differs: {audit}.")
    return audit


def append_reference_recovery(
    spark: Any, namespace: str, *, owner: str, status: str, evidence: dict
) -> None:
    """Audit explicit admission recovery without impersonating the failed operation's owner.

    The recovery caller must independently verify the holder has terminated and append
    intent before conditionally clearing that exact token, then append released afterward.
    Audit schema and ownership validation are identical to reference retirement.
    """
    if not re.fullmatch(r"[0-9a-f]{32}", owner) or status not in {"intent", "released"}:
        raise ValueError("Invalid reference admission recovery audit identity or status.")
    reason = json.dumps(evidence, sort_keys=True, allow_nan=False)
    item = {"reference_id": owner, "receipt": qualified_name(f"{namespace}.retraining_requests")}
    _record(
        spark,
        _audit_table(spark, namespace),
        owner,
        item,
        status,
        reason,
        action="release_admission",
    )


def _source_still_used(spark: Any, namespace: str, item: dict) -> bool:
    """Re-derive surviving users under admission, including receipts excluded by either plan."""
    rows = spark.sql(f"SHOW TABLES IN {table_name(namespace)} LIKE '{_PREFIX}*{_SUFFIX}'")
    names = rows.limit(MAX_REFERENCES + 1).collect()
    if len(names) > MAX_REFERENCES:
        raise ValueError("Monitoring store has too many references to plan at once.")
    for row in names:
        name = qualified_name(f"{namespace}.{row['tableName']}")
        if name != item["receipt"]:
            receipt, _ = _receipt(spark, name)
            if _shared_source(receipt, namespace) == item["shared_source"]:
                return True
    return False


def _record(spark, audit, cleanup_id, item, status, reason, *, action="delete") -> None:
    """Append one immutable audit row for a reference."""
    row = {
        "audit_id": uuid4().hex,
        "cleanup_id": cleanup_id,
        "reference_id": item["reference_id"],
        "table_name": item["receipt"],
        "action": action,
        "status": status,
        "reason": reason,
        "recorded_at": datetime.now(UTC),
    }
    spark.createDataFrame([row], AUDIT_SCHEMA).writeTo(audit).append()


def _mlflow_aliases(model_name: str) -> dict[str, str]:
    """Read a registered model's aliases from Unity Catalog; failures propagate."""
    from ....mlflow.shared._client import make_registry_client, require_mlflow  # noqa: PLC0415

    client = make_registry_client(require_mlflow(), "databricks", "databricks-uc")
    return dict(client.get_registered_model(model_name).aliases)


def _job_run_state(run_id: int) -> str:
    """Read a job run's life-cycle state; failures propagate and protect references."""
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    state = WorkspaceClient().jobs.get_run(run_id).state
    if state is None or state.life_cycle_state is None:
        raise ValueError(f"Job run {run_id} has no life-cycle state.")
    return str(getattr(state.life_cycle_state, "value", state.life_cycle_state))
