"""Preserve an opt-in scoring boundary without adding a second checkpoint store."""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

from ...data.admission import BatchConflictError
from ...data.delta_io.cdf_recovery import CdfHistoryExpired, normalize_cdf_error
from ...data.delta_io.delta import table_identity


@dataclass(frozen=True)
class PredictionCheckpoint:
    """Pin previously admitted keys to one verified prediction commit and source watermark."""

    table: str
    table_id: str
    version: int
    source_version: int


def validate_initial_scoring(value: Any) -> str:
    """Require an explicit supported choice before source reads or target writes."""
    if value not in ("full_snapshot", "from_now"):
        raise ValueError("initial_scoring must be full_snapshot or from_now.")
    return value


def start_fields(previous: dict | None, mode: str, upper: int) -> dict[str, Any]:
    """Keep the original boundary immutable, treating legacy receipts as full snapshots."""
    validate_initial_scoring(mode)
    if previous is None:
        return (
            {"initial_scoring": mode, "scoring_start_version": upper} if mode == "from_now" else {}
        )
    if previous.get("initial_scoring", "full_snapshot") != mode:
        raise BatchConflictError(
            "initial_scoring differs from the committed target; use a new prediction target "
            "to choose a different starting policy."
        )
    boundary = previous.get("scoring_start_version")
    if mode == "full_snapshot":
        if "scoring_start_version" in previous:
            raise BatchConflictError("Unexpected scoring_start_version on a full_snapshot receipt.")
        return {}
    end = previous.get("source_end_version")
    if type(boundary) is not int or type(end) is not int or not 0 <= boundary <= end <= upper:
        raise BatchConflictError("Invalid scoring_start_version in the committed target receipt.")
    return {"initial_scoring": mode, "scoring_start_version": boundary}


def scoring_snapshot(
    spark: Any,
    source: str,
    upper: int,
    keys: tuple[str, ...],
    boundary: int,
    *,
    checkpoint: PredictionCheckpoint | None = None,
) -> Any:
    """Read only keys added after the start snapshot; never fall back to all history.

    Normal rebuilds reuse persisted prediction keys plus recent inserts. Only
    explicit CDF recovery needs the original source snapshot. Record keys must
    be immutable and never reused.
    """
    selected = spark.read.format("delta").option("versionAsOf", upper).table(source)
    if boundary == upper:
        return selected.limit(0)
    if checkpoint is not None:
        members = _rebuild_keys(spark, source, upper, keys, checkpoint)
        return selected.join(members, on=list(keys), how="left_semi")
    try:
        excluded = spark.read.format("delta").option("versionAsOf", boundary).table(source)
        return selected.join(excluded.select(*keys), on=list(keys), how="left_anti")
    except Exception as exc:  # noqa: BLE001 - retain the Spark cause and never broaden membership
        raise ValueError(
            f"Cannot read scoring_start_version={boundary} for from_now rebuild. "
            "Restore that Delta snapshot or use a new prediction target with an explicit "
            "starting policy; historical rows will not be scored automatically."
        ) from exc


def _rebuild_keys(
    spark: Any,
    source: str,
    upper: int,
    keys: tuple[str, ...],
    checkpoint: PredictionCheckpoint,
) -> Any:
    """Recover membership from output history, including excluded rows and unscored inserts."""
    if table_identity(spark, checkpoint.table) != checkpoint.table_id:
        raise BatchConflictError("Previous prediction checkpoint table identity changed.")
    members = (
        spark.read.format("delta")
        .option("versionAsOf", checkpoint.version)
        .table(checkpoint.table)
        .select(*keys)
    )
    if checkpoint.source_version < upper:
        changes = (
            spark.read.format("delta")
            .option("readChangeFeed", "true")
            .option("startingVersion", checkpoint.source_version + 1)
            .option("endingVersion", upper)
            .table(source)
        )
        members = members.unionByName(changes.where("_change_type = 'insert'").select(*keys))
    return members


@contextmanager
def snapshot_read_errors(boundary: int | None) -> Iterator[None]:
    """Keep lazy retention failures actionable without turning other errors into recovery."""
    try:
        with normalize_cdf_error() if boundary is not None else nullcontext():
            yield
    except Exception as exc:  # noqa: BLE001 - classify only structured retention/read errors
        if boundary is None or not (
            isinstance(exc, CdfHistoryExpired) or _missing_snapshot_files(exc)
        ):
            raise
        raise ValueError(
            f"from_now rebuild cannot read required Delta history (scoring_start_version={boundary}). "
            "Restore the required source snapshot/CDF and prediction checkpoint, or use a new "
            "prediction target with an explicit starting policy; historical rows will not be "
            "scored automatically."
        ) from exc


def _missing_snapshot_files(error: Any) -> bool:
    """Recognize missing data files only inside the already selected rebuild read."""
    conditions = {
        "DELTA_FILE_NOT_FOUND",
        "FAILED_READ_FILE.DBR_FILE_NOT_EXIST",
        "FAILED_READ_FILE.FILE_NOT_EXIST",
    }
    return any(
        callable(getter := getattr(error, name, None)) and getter() in conditions
        for name in ("getCondition", "getErrorClass")
    )
