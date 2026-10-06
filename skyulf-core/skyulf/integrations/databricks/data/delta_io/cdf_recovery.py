"""Pin optional full-snapshot recovery to a recognized lost CDF history interval."""

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ...shared._contracts import table_name
from ...shared.json_contracts import finite_json_digest
from ..admission import BatchConflictError

# These Delta conditions identify unavailable or retention-blocked CDF history, not CDF
# that was never enabled (DELTA_MISSING_CHANGE_DATA), or generic missing files.
# Databricks also documents manual removal as a possible cause of these errors;
# recovery still requires a readable pinned snapshot and unchanged table IDs.
_HISTORY_ERRORS = frozenset(
    {
        "DELTA_CHANGE_DATA_FILE_NOT_FOUND",
        "DELTA_TRUNCATED_TRANSACTION_LOG",
        "DELTA_MISSING_FILES_UNEXPECTED_VERSION",
        "DELTA_UNSUPPORTED_TIME_TRAVEL_BEYOND_DELETED_FILE_RETENTION_DURATION",
    }
)
_TEXT_FIELDS = (
    "source_table",
    "source_table_id",
    "target_table",
    "target_table_id",
    "model_name",
    "model_version",
    "model_digest",
)
_VERSION_FIELDS = ("target_version", "source_start_version", "source_end_version")
_REQUEST_FIELDS = frozenset(("version", "layout", *_TEXT_FIELDS, *_VERSION_FIELDS))
_CONNECT_READ_ERRORS = frozenset(
    {"FAILED_READ_FILE.DBR_FILE_NOT_EXIST", "FAILED_READ_FILE.FILE_NOT_EXIST"}
)
_JAVA_DELTA_CAUSE = re.compile(
    r"^Caused by: (?:com\.databricks\.sql\.transaction\.tahoe|org\.apache\.spark\.sql\.delta)"
    r"\.Delta(?:FileNotFound|IllegalState)Exception: \[(DELTA_[A-Z_]+)\](?:\s|$)",
    re.MULTILINE,
)


class CdfHistoryExpired(RuntimeError):
    """A CDF read or materialization encountered unavailable historical files/logs."""


class CdfRecoveryRequired(RuntimeError):
    """Carry the exact publication state required by a separate recovery task."""

    def __init__(self, request: dict[str, Any]) -> None:
        """Detach and validate the pinned request before exposing it to a caller."""
        self.request = validate_recovery_request(request)
        super().__init__(
            f"CDF history is unavailable for {self.request['source_table']}; "
            f"full recovery of {self.request['target_table']} at source version "
            f"{self.request['source_end_version']} requires explicit recovery."
        )


def _error_method(error: Any, name: str) -> Any:
    """Read Spark/Py4J structured evidence without depending on a specific runtime."""
    try:
        getter = getattr(error, name, None)
        return getter() if callable(getter) else None
    except Exception:  # noqa: BLE001 - unavailable bridge methods are not error evidence
        return None


def _has_history_error(error: Any) -> bool:
    """Inspect bounded explicit cause chains, never search free-form exception text."""
    pending = [error]
    seen: set[int] = set()
    for _ in range(32):
        if not pending:
            return False
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        conditions = _error_conditions(current)
        if _HISTORY_ERRORS.intersection(conditions):
            return True
        if _connect_history_error(current, conditions):
            return True
        pending.extend(
            (
                getattr(current, "__cause__", None),
                getattr(current, "java_exception", None),
                _error_method(current, "getCause"),
            )
        )
    return False


def _error_conditions(error: Any) -> tuple[str, ...]:
    """Keep only actual structured condition strings exposed by Spark error methods."""
    values = (_error_method(error, "getCondition"), _error_method(error, "getErrorClass"))
    return tuple(value for value in values if isinstance(value, str))


def _connect_history_error(error: Any, conditions: tuple[str, ...]) -> bool:
    """Read exact Java cause headers from Spark Connect's separate server stacktrace."""
    if not _CONNECT_READ_ERRORS.intersection(conditions):
        return False
    if not any(
        base.__module__ == "pyspark.errors.exceptions.connect"
        and base.__name__ == "SparkConnectGrpcException"
        for base in type(error).__mro__
    ):
        return False
    trace = _error_method(error, "getStackTrace")
    if trace is None:
        trace = getattr(error, "_stacktrace", None)
    if not isinstance(trace, str) or len(trace) > 1024 * 1024:
        return False
    return any(match[1] in _HISTORY_ERRORS for match in _JAVA_DELTA_CAUSE.finditer(trace))


@contextmanager
def normalize_cdf_error() -> Iterator[None]:
    """Normalize known history failures only inside a caller's CDF read boundary."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - preserve all non-CDF errors unchanged
        if _has_history_error(exc):
            raise CdfHistoryExpired(
                "Delta CDF history files or transaction log are unavailable."
            ) from exc
        raise


def _validate_request_values(request: dict[str, Any]) -> None:
    """Require bounded concrete identities and a forward source watermark interval."""
    for key in _TEXT_FIELDS:
        if type(request[key]) is not str or not request[key].strip():
            raise ValueError(f"CDF recovery {key} must be a nonempty string.")
    for key in _VERSION_FIELDS:
        if type(request[key]) is not int or request[key] < 0:
            raise ValueError(f"CDF recovery {key} must be a nonnegative integer.")
    if request["source_end_version"] <= request["source_start_version"]:
        raise ValueError("CDF recovery requires a forward source interval.")
    if not re.fullmatch(r"[1-9][0-9]*", request["model_version"]):
        raise ValueError("CDF recovery requires a concrete model_version.")
    if not re.fullmatch(r"[0-9a-f]{64}", request["model_digest"]):
        raise ValueError("CDF recovery requires a SHA256 model_digest.")


def validate_recovery_request(request: Any) -> dict[str, Any]:
    """Return a detached, strictly validated task payload with no optional identities."""
    if type(request) is not dict or set(request) != _REQUEST_FIELDS:
        raise ValueError("CDF recovery request fields differ from format version 1.")
    if type(request["version"]) is not int or request["version"] != 1:
        raise ValueError("CDF recovery request version must be 1.")
    if request["layout"] not in ("single_model", "model_set"):
        raise ValueError("CDF recovery layout must be single_model or model_set.")
    _validate_request_values(request)
    for key in ("source_table", "target_table", "model_name"):
        table_name(request[key])
    if (
        request["source_table"] == request["target_table"]
        or request["source_table_id"] == request["target_table_id"]
    ):
        raise ValueError("CDF recovery source and target must be distinct tables.")
    if len(json.dumps(request, allow_nan=False).encode()) > 40 * 1024:
        raise ValueError("CDF recovery request exceeds its 40 KiB task-value budget.")
    return dict(request)


def make_recovery_request(
    *,
    layout: str,
    source_table: str,
    source_table_id: str,
    target_table: str,
    target_table_id: str,
    target_version: int,
    source_start_version: int,
    source_end_version: int,
    model_name: str,
    model_version: str,
    model_digest: str,
) -> dict[str, Any]:
    """Create a versioned request whose start is the last committed source watermark."""
    return validate_recovery_request(
        {
            "version": 1,
            "layout": layout,
            "source_table": source_table,
            "source_table_id": source_table_id,
            "target_table": target_table,
            "target_table_id": target_table_id,
            "target_version": target_version,
            "source_start_version": source_start_version,
            "source_end_version": source_end_version,
            "model_name": model_name,
            "model_version": model_version,
            "model_digest": model_digest,
        }
    )


def recovery_request_digest(request: dict[str, Any]) -> str:
    """Hash canonical pinned content independently of task-value dictionary ordering."""
    return finite_json_digest(validate_recovery_request(request))


def recovery_receipt_fields(request: dict[str, Any]) -> dict[str, Any]:
    """Bind a committed full replacement to the exact recovery request it satisfied."""
    request = validate_recovery_request(request)
    return {
        "cdf_recovered": True,
        "cdf_recovery_request": request,
        "cdf_recovery_request_digest": recovery_request_digest(request),
    }


def validate_recovery_binding(
    request: dict[str, Any],
    *,
    layout: str,
    source_table: str,
    target_table: str,
    model_name: str,
    model_version: str,
    model_digest: str,
) -> dict[str, Any]:
    """Prevent a delayed task from following a new alias, artifact or table name."""
    expected = {
        "layout": layout,
        "source_table": source_table,
        "target_table": target_table,
        "model_name": model_name,
        "model_version": model_version,
        "model_digest": model_digest,
    }
    request = validate_recovery_request(request)
    if any(request[key] != value for key, value in expected.items()):
        raise BatchConflictError("CDF recovery model or table binding changed.")
    return request


def _check_recovered_model(request: dict[str, Any], previous: dict[str, Any]) -> None:
    """Require a completed recovery to have scored its pinned model generation."""
    prefix = "model_set" if request["layout"] == "model_set" else "model"
    for suffix in ("name", "version", "digest"):
        if previous.get(f"{prefix}_{suffix}") != request[f"model_{suffix}"]:
            raise BatchConflictError("CDF recovery previous model identity changed.")


def check_recovery_state(
    request: dict[str, Any],
    previous: dict[str, Any] | None,
    target_version: int,
) -> bool:
    """Accept a pinned base receipt or return true for only its exact committed replay."""
    request = validate_recovery_request(request)
    if previous is None:
        raise BatchConflictError("CDF recovery requires a trusted prior receipt.")
    for key in ("source_table_id", "target_table_id"):
        if previous.get(key) != request[key]:
            raise BatchConflictError("CDF recovery source or target table ID changed.")
    if previous.get("cdf_recovery_request_digest") == recovery_request_digest(request):
        _check_recovered_model(request, previous)
        expected = recovery_receipt_fields(request) | {
            "source_end_version": request["source_end_version"],
            "expected_target_version": request["target_version"],
        }
        if (
            any(previous.get(key) != value for key, value in expected.items())
            or target_version != request["target_version"] + 1
        ):
            raise BatchConflictError("CDF recovery committed request evidence differs.")
        return True
    if (
        target_version != request["target_version"]
        or previous.get("source_end_version") != request["source_start_version"]
    ):
        raise BatchConflictError("Target changed since the CDF recovery request was created.")
    return False
