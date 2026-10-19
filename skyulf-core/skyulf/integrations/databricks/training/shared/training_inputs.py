"""Native MLflow Inputs for already materialized training and final holdout rows."""

import hashlib
import json
import os
from typing import Any

import polars as pl


def capture_training_uc_id(table: str) -> str | None:
    """Read UC object metadata only in a native Databricks runtime, never as a Delta ID."""
    catalog = table.split(".", 1)[0].casefold()
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION") or catalog in {
        "hive_metastore",
        "spark_catalog",
        "samples",
    }:
        return None
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415 - native runtime dependency

    identity = WorkspaceClient().tables.get(table).table_id
    if not isinstance(identity, str) or not identity:
        raise ValueError("Training source requires an inspectable Unity Catalog table ID.")
    return identity


def log_training_inputs(
    run: Any, fitted: Any, spec_payload: dict[str, Any], source_receipt: dict[str, Any]
) -> None:
    """Attach actual bounded split metadata to each model-producing run without source reads."""
    from mlflow.data.code_dataset_source import CodeDatasetSource  # noqa: PLC0415
    from mlflow.data.pandas_dataset import from_pandas  # noqa: PLC0415 - tracked fit boundary
    from mlflow.data.polars_dataset import from_polars  # noqa: PLC0415 - tracked fit boundary
    from mlflow.entities import Dataset, DatasetInput, InputTag  # noqa: PLC0415

    spec = fitted.spec
    attrs = fitted.source_frame.attrs
    provenance = {
        "version": 1,
        "source_table_id": attrs.get("source_table_id"),
        "source_uc_table_id": attrs.get("source_uc_table_id"),
        "training_spec": spec_payload,
        "training_evidence": fitted.evidence,
        "source_evidence": source_receipt,
        "training_selection": attrs.get("training_selection"),
        "feature_lookup": attrs.get("feature_lookup"),
        **({"training_labels": attrs["training_labels"]} if "training_labels" in attrs else {}),
        **(
            {"source_column_conversions": attrs["source_column_conversions"]}
            if "source_column_conversions" in attrs
            else {}
        ),
    }
    serialized = json.dumps(
        provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    run.client.log_dict(run.run_id, provenance, "dataset_inputs.json")
    source = _input_source(fitted, hashlib.sha256(serialized.encode()).hexdigest())
    roles = {
        "skyulf.feature_columns": json.dumps(list(spec.input_columns)),
        "skyulf.target_column": spec.target_column,
    }
    if spec.weight_column:
        roles["skyulf.weight_column"] = spec.weight_column
    inputs = []
    for context, frame in (("training", fitted.training_frame), ("holdout", fitted.holdout)):
        factory = from_polars if isinstance(frame, pl.DataFrame) else from_pandas
        data = factory(
            frame,
            source=CodeDatasetSource(tags={}),
            targets=spec.target_column,
            name=f"{spec.table}@{spec.version}/{context}",
        )
        payload = data.to_dict()
        # Explicit source serialization keeps version zero and avoids MLflow's UC lookup.
        payload.update(source_type="delta_table", source=source)
        payload["digest"] = hashlib.sha256(
            f"{source}:{context}:{data.digest}".encode()
        ).hexdigest()[:32]
        inputs.append(
            DatasetInput(
                Dataset(**payload),
                [InputTag("mlflow.data.context", context)]
                + [InputTag(key, value) for key, value in roles.items()],
            )
        )
    run.log_inputs(inputs)


def _input_source(fitted: Any, provenance_digest: str) -> str:
    """Keep snapshot and derivation evidence separate from native Unity Catalog identity."""
    spec = fitted.spec
    attrs = fitted.source_frame.attrs
    source = {
        "delta_table_name": spec.table,
        "delta_table_version": spec.version,
        "skyulf": {
            "source_table_id": attrs.get("source_table_id"),
            "dataset_id": spec.dataset_id,
            "provenance_artifact": "dataset_inputs.json",
            "provenance_sha256": provenance_digest,
        },
    }
    if attrs.get("source_uc_table_id"):
        source.update(is_databricks_uc_table=True, delta_table_id=attrs["source_uc_table_id"])
    return json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
