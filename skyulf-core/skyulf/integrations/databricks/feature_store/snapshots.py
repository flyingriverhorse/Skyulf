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
    """Reject feature-only changes under the explicit training-snapshot policy."""
    if type(records) is not list or not records:
        raise ValueError("Feature snapshot records must be a nonempty list.")
    for expected in records:
        if snapshot_table(spark, expected["table_name"]) != expected:
            raise ValueError(
                f"Feature table {expected['table_name']!r} changed since training; "
                "retrain and register a refreshed feature package before scoring."
            )
