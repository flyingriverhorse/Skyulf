"""Read and verify exact Delta feature snapshots without collecting table data."""

from typing import Any

from ..data.delta_io.delta import history, table_identity
from .config import FeatureTrainingSpec, validate_uc_name


def snapshot_table(spark: Any, name: str) -> dict[str, Any]:
    """Capture a stable table identity around its current Delta history version."""
    validate_uc_name(name)
    identity = table_identity(spark, name)
    latest = history(spark, name).orderBy("version", ascending=False).select("version").first()
    if latest is None:
        raise ValueError(f"Feature table {name!r} has no readable Delta history.")
    if table_identity(spark, name) != identity:
        raise ValueError(f"Feature table {name!r} identity changed while reading its version.")
    return {"table_name": name, "table_id": identity, "version": int(latest["version"])}


def snapshot_tables(spark: Any, spec: FeatureTrainingSpec) -> list[dict[str, Any]]:
    """Read each declared feature table exactly once in deterministic name order."""
    if not isinstance(spec, FeatureTrainingSpec):
        raise TypeError("spec must be FeatureTrainingSpec.")
    return [
        snapshot_table(spark, name) for name in sorted({item.table_name for item in spec.lookups})
    ]


def validate_snapshots(spark: Any, records: list[dict[str, Any]]) -> None:
    """Reject logical changes while admitting proven data-preserving Delta maintenance.

    A changed version requires exact schema and multiset equality against the
    retained training version. The distributed comparison can scan both full
    snapshots; only a bounded mismatch count reaches the driver. Missing history
    fails closed, and an operation name never substitutes for this proof.
    """
    if type(records) is not list or not records:
        raise ValueError("Feature snapshot records must be a nonempty list.")
    for expected in records:
        current = snapshot_table(spark, expected["table_name"])
        if current == expected:
            continue
        if not _unchanged_data(spark, expected, current):
            raise ValueError(
                f"Feature table {expected['table_name']!r} changed since training; "
                "retrain and register a refreshed feature package before scoring."
            )


def _unchanged_data(spark: Any, expected: dict[str, Any], current: dict[str, Any]) -> bool:
    """Prove retained snapshots agree without accepting table replacement or a moving head."""
    if current["table_id"] != expected["table_id"] or current["version"] <= expected["version"]:
        return False
    name = expected["table_name"]
    previous = spark.read.format("delta").option("versionAsOf", expected["version"]).table(name)
    latest = spark.read.format("delta").option("versionAsOf", current["version"]).table(name)
    if previous.schema != latest.schema:
        return False
    if previous.exceptAll(latest).limit(1).count() or latest.exceptAll(previous).limit(1).count():
        return False
    return snapshot_table(spark, name) == current
