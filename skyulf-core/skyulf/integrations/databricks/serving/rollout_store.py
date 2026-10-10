"""Immutable MLflow rollout receipts with one verified current-pointer tag."""

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from ..shared.json_contracts import finite_json_digest

_POINTER = "skyulf.rollout.current"
_DIRECTORY = "rollout/receipts"


class MLflowRolloutStore:
    """Persist receipts through an explicit authenticated tracking client and run.

    All writes must be inside the controller's shared endpoint admission. The
    backend tag operation is not compare-and-swap and is never used as a lock.
    Restrict receipt/tag writes to those same authorized controller writers.
    """

    def __init__(self, client: Any, run_id: str) -> None:
        """Bind the caller's existing run without creating clients or fluent runs."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("An explicit rollout run_id is required.")
        self.client = client
        self.run_id = run_id

    def _pointer(self) -> str | None:
        """Fetch fresh durable identity before each read or write."""
        return self.client.get_run(self.run_id).data.tags.get(_POINTER)

    def require_empty(self) -> None:
        """Reject existing or interrupted rollout initialization on this run."""
        if self._pointer() is not None or self.client.list_artifacts(self.run_id, _DIRECTORY):
            raise ValueError("An existing rollout or incomplete receipt already occupies this run.")

    def _read(self, receipt_id: str) -> dict[str, Any]:
        """Download a bounded identifier into an ephemeral local directory."""
        if not isinstance(receipt_id, str) or not re.fullmatch(r"[0-9a-f]{32}", receipt_id):
            raise ValueError("Invalid rollout receipt identity.")
        with TemporaryDirectory(prefix="skyulf-rollout-") as directory:
            path = self.client.download_artifacts(
                self.run_id, f"{_DIRECTORY}/{receipt_id}.json", directory
            )
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Rollout receipt must be a JSON object.")
        return value

    def load(self) -> dict[str, Any]:
        """Verify the pointer, JSON digest, schema and explicit run before reuse."""
        raw = self._pointer()
        try:
            pointer = json.loads(raw) if isinstance(raw, str) else None
            if not isinstance(pointer, dict) or set(pointer) != {"receipt_id", "sha256"}:
                raise ValueError("Missing or malformed rollout receipt pointer.")
            value = self._read(pointer["receipt_id"])
            if finite_json_digest(value) != pointer["sha256"]:
                raise ValueError("Rollout receipt digest differs from its pointer.")
            if (
                type(value.get("schema_version")) is not int
                or value["schema_version"] != 1
                or value.get("run_id") != self.run_id
                or value.get("receipt_id") != pointer["receipt_id"]
            ):
                raise ValueError("Rollout receipt identity or schema differs.")
            return value
        except (KeyError, TypeError) as exc:
            raise ValueError("Missing or malformed rollout receipt.") from exc

    def write(self, value: dict[str, Any], *, expected_receipt_id: str | None) -> dict[str, Any]:
        """Read back immutable content before publishing and verifying its pointer."""
        if expected_receipt_id is None:
            self.require_empty()
        elif self.load()["receipt_id"] != expected_receipt_id:
            raise ValueError("Rollout current receipt changed before the write.")
        record: dict[str, Any] = value | {
            "schema_version": 1,
            "run_id": self.run_id,
            "receipt_id": uuid4().hex,
            "parent_receipt_id": expected_receipt_id,
        }
        digest = finite_json_digest(record)
        self.client.log_dict(self.run_id, record, f"{_DIRECTORY}/{record['receipt_id']}.json")
        if finite_json_digest(self._read(record["receipt_id"])) != digest:
            raise ValueError("Written rollout receipt failed artifact readback.")
        pointer = json.dumps({"receipt_id": record["receipt_id"], "sha256": digest})
        self.client.set_tag(self.run_id, _POINTER, pointer)
        readback = self.load()
        if finite_json_digest(readback) != digest:
            raise ValueError("Written rollout receipt failed pointer readback.")
        return readback
