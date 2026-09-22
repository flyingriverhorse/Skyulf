"""Exercise retained, test-owned Delta tables through the public batch runner.

The caller supplies a pinned raw regression bundle with the known y=2*x oracle,
its concrete registry identity, and an existing schema or catalog.schema. This
probe creates three unique tables and retains them even on failure. It neither
creates schemas nor grants permissions. Runtime failures propagate unchanged;
passing this probe does not complete the full Databricks platform gate.
"""

import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from typing import Any
from uuid import uuid4

import numpy as np

from skyulf.core.execution import ExecutionOptions
from skyulf.inference.bundle import InferenceBundle
from skyulf.integrations.databricks.admission import BatchConflictError
from skyulf.integrations.databricks.batch import BatchResult, BatchSpec, run_batch
from skyulf.integrations.databricks.delta import table_identity
from skyulf.integrations.databricks.delta_admission import DeltaTableAdmission


def _receipt(result: BatchResult) -> dict[str, Any]:
    """Keep the verified commit and row counts without embedding runtime objects."""
    return {
        "commit_version": result.commit_version,
        "input_count": result.input_count,
        "output_count": result.output_count,
        "replayed": result.replayed,
    }


def run_smoke(
    spark: Any,
    *,
    namespace: str,
    bundle: InferenceBundle,
    model_name: str,
    model_version: str,
) -> dict[str, Any]:
    """Verify publication, retries, stale requests and explicit empty replacement.

    Supply a raw bundle whose only input feature is ``x`` and whose regression
    prediction is ``2*x``. Mean imputation and scaling are supported. Table names
    are printed before creation so a failed live run remains inspectable. Cleanup
    belongs to the caller and must use only the three names owned by this run.
    """
    if type(namespace) is not str or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", namespace
    ):
        raise ValueError("namespace must be a simple schema or catalog.schema.")
    identity = uuid4().hex
    names = {
        role: f"{namespace}.skyulf_delta_smoke_{identity}_{role}"
        for role in ("source", "target", "control")
    }
    spec = BatchSpec(
        period_start=datetime(2026, 1, 1, tzinfo=UTC),
        period_end=datetime(2026, 2, 1, tzinfo=UTC),
        as_of=datetime.now(UTC) + timedelta(minutes=10),
        row_keys=("id",),
        output_table=names["target"],
        model_name=model_name,
        model_version=model_version,
        source_version=0,
        code_version=version("skyulf-core"),
        run_id=f"smoke-{identity}",
        model_digest=bundle.semantic_digest,
        expected_target_version=0,
    )
    print("Test-owned Delta tables (retained for inspection): " + json.dumps(names), flush=True)
    february = (3, datetime(2026, 2, 1, tzinfo=UTC), 3.0)
    spark.createDataFrame(
        [
            (1, datetime(2026, 1, 1, tzinfo=UTC), 1.0),
            (2, datetime(2026, 1, 31, tzinfo=UTC), 2.0),
            february,
        ],
        "id long, event_time timestamp, x double",
    ).write.format("delta").saveAsTable(names["source"])
    spark.createDataFrame(
        [(9, datetime(2025, 12, 1, tzinfo=UTC), 99.0, "prior", model_name, model_version)],
        "id long, event_time timestamp, prediction double, __skyulf_run_id string, "
        "__skyulf_model_name string, __skyulf_model_version string",
    ).write.format("delta").saveAsTable(names["target"])
    prior = spark.table(names["target"]).first()
    spark.createDataFrame(
        [(table_identity(spark, names["target"]), None)], "target_id string, owner string"
    ).write.format("delta").saveAsTable(names["control"])
    admission = DeltaTableAdmission(spark, names["control"])

    def execute(request: BatchSpec) -> BatchResult:
        """Use the public batch API and verify ownership release after each success."""
        result = run_batch(
            spark,
            request,
            source=names["source"],
            bundle=bundle,
            options=ExecutionOptions("spark"),
            admission=admission,
        )
        if spark.table(names["control"]).first().owner is not None:
            raise AssertionError("Successful batch retained admission ownership.")
        return result

    first = execute(spec)
    rows = spark.table(names["target"]).orderBy("id").limit(4).collect()
    if [row.id for row in rows] != [1, 2, 9] or rows[-1] != prior:
        raise AssertionError("Publication changed row membership or the prior period.")
    np.testing.assert_allclose([row.prediction for row in rows], [2.0, 4.0, 99.0], atol=1e-10)
    replay = execute(spec)
    if (
        first.replayed
        or not replay.replayed
        or first.input_count != 2
        or first.output_count != 2
        or first.commit_version != replay.commit_version
        or first.manifest != replay.manifest
        or spark.table(names["target"]).orderBy("id").limit(4).collect() != rows
    ):
        raise AssertionError("Replay changed the original commit receipt or predictions.")
    try:
        execute(replace(spec, run_id=spec.run_id + "-stale"))
    except BatchConflictError as exc:
        if "Target version changed" not in str(exc):
            raise
    else:
        raise AssertionError("A stale expected target version was admitted.")
    if spark.table(names["control"]).first().owner is not None:
        raise AssertionError("Stale-request rejection retained admission ownership.")
    spark.createDataFrame([february], "id long, event_time timestamp, x double").write.format(
        "delta"
    ).mode("overwrite").saveAsTable(names["source"])
    empty_spec = replace(
        spec,
        run_id=spec.run_id + "-empty",
        source_version=1,
        expected_target_version=first.commit_version,
        as_of=datetime.now(UTC) + timedelta(minutes=10),
    )
    try:
        execute(empty_spec)
    except ValueError as exc:
        if "allow_empty" not in str(exc):
            raise
    else:
        raise AssertionError("Empty replacement did not require explicit consent.")
    if spark.table(names["target"]).orderBy("id").limit(4).collect() != rows:
        raise AssertionError("A rejected request changed target predictions.")
    empty = execute(replace(empty_spec, allow_empty=True))
    empty_replay = execute(replace(empty_spec, allow_empty=True))
    if (
        empty.replayed
        or not empty_replay.replayed
        or empty.commit_version != empty_replay.commit_version
        or empty.commit_version != first.commit_version + 1
        or empty.manifest != empty_replay.manifest
        or empty.input_count != 0
        or empty.output_count != 0
        or spark.table(names["target"]).limit(2).collect() != [prior]
    ):
        raise AssertionError("Explicit empty replacement or its replay changed the prior period.")
    return {
        "stage": "delta_publication",
        "platform_gate_complete": False,
        "checks": dict.fromkeys(
            (
                "publication",
                "replay",
                "stale_rejected",
                "empty_rejected",
                "empty_replacement",
                "empty_replay",
                "ownership_released",
                "prior_period_preserved",
            ),
            True,
        ),
        "model_name": model_name,
        "model_version": model_version,
        "model_digest": bundle.semantic_digest,
        "code_version": spec.code_version,
        "tables": {
            role: {
                "name": name,
                "id": table_identity(spark, name),
                "version": int(spark.sql(f"DESCRIBE HISTORY {name}").first().version),
                "count": spark.table(name).count(),
            }
            for role, name in names.items()
        },
        "receipts": {
            key: _receipt(result)
            for key, result in (
                ("first", first),
                ("replay", replay),
                ("empty", empty),
                ("empty_replay", empty_replay),
            )
        },
    }
