"""Non-expiring distributed publish admission through a preprovisioned Delta row."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from ._contracts import table_name
from .admission import BatchConflictError
from .delta import table_identity


class DeltaTableAdmission:
    """Serialize participating publishers using one shared Delta control table.

    Operators must provision exactly one row with ``target_id STRING`` bound to
    the immutable output Delta table ID and ``owner STRING`` initially NULL.
    All publishers of that target must use this same control table. Its identity,
    schema and target binding must remain fixed; external DDL, control-row edits
    and publishers bypassing this protocol are outside the guarantee.

    Ownership never expires. A crashed driver or an uncertain acquisition can
    leave an owner behind. Clear it manually only after proving that driver can
    no longer publish. This provider does not create tables, insert rows, retry
    conflicts or implement time-based leases. Spark SQL parameter markers require
    Spark 3.4 or later (including compatible Spark Connect runtimes).
    """

    local_only = False

    def __init__(self, spark: Any, control_table: str) -> None:
        """Bind to an existing Delta authority without creating or modifying it."""
        self._spark = spark
        self._table = control_table
        self._quoted_table = table_name(control_table)
        self._control_id = self._identity()

    def _identity(self) -> str:
        """Invalidate cached metadata and read the authority's current Delta ID."""
        self._spark.sql(f"REFRESH TABLE {self._quoted_table}").collect()
        return table_identity(self._spark, self._table)

    def _state(self, target_id: str) -> str | None:
        """Read fresh singleton ownership, rejecting malformed or rebound authority."""
        if self._identity() != self._control_id:
            raise BatchConflictError("Control table identity changed during admission.")
        frame = self._spark.table(self._table)
        fields = {field.name: field for field in frame.schema}
        if (
            len(frame.schema) != 2
            or set(fields) != {"target_id", "owner"}
            or any(field.dataType.typeName() != "string" for field in fields.values())
            or not fields["owner"].nullable
        ):
            raise ValueError(
                "Control table requires only target_id STRING and nullable owner STRING."
            )
        rows = frame.limit(2).collect()
        if len(rows) != 1 or rows[0]["target_id"] != target_id:
            raise ValueError("Control table must contain exactly one row bound to this target ID.")
        if self._identity() != self._control_id:
            raise BatchConflictError("Control table identity changed while reading ownership.")
        return rows[0]["owner"]

    def _update(self, statement: str, target_id: str, token: str) -> None:
        """Execute conditional ownership changes, preserving Delta conflict causes."""
        try:
            self._spark.sql(statement, args={"target_id": target_id, "token": token}).collect()
        except Exception as exc:  # noqa: BLE001 - Spark Connect and classic wrap Delta differently
            name = type(exc).__name__
            message = str(exc)
            if (
                "Concurrent" in name
                or name in {"MetadataChangedException", "ProtocolChangedException"}
                or any(
                    marker in message
                    for marker in (
                        "DELTA_CONCURRENT",
                        "DELTA_METADATA_CHANGED",
                        "DELTA_PROTOCOL_CHANGED",
                    )
                )
            ):
                raise BatchConflictError("Delta rejected a concurrent admission change.") from exc
            raise

    @contextmanager
    def hold(self, table_id: str) -> Iterator[None]:
        """Claim the target until context exit, clearing only this context's token."""
        if type(table_id) is not str or not table_id:
            raise ValueError("Admission requires a nonempty immutable target table ID.")
        if self._state(table_id) is not None:
            raise BatchConflictError("Another publisher holds this target's admission.")
        token = str(uuid4())
        self._update(
            f"UPDATE {self._quoted_table} SET owner = :token "
            "WHERE target_id = :target_id AND owner IS NULL",
            table_id,
            token,
        )
        if self._state(table_id) != token:
            raise BatchConflictError("Admission ownership could not be verified.")
        try:
            yield
        finally:
            if self._state(table_id) != token:
                raise BatchConflictError("Admission owner changed before release; nothing cleared.")
            self._update(
                f"UPDATE {self._quoted_table} SET owner = NULL "
                "WHERE target_id = :target_id AND owner = :token",
                table_id,
                token,
            )
            # A new publisher may acquire immediately after this release commits.
            if self._state(table_id) == token:
                raise BatchConflictError("Admission release could not be verified.")
