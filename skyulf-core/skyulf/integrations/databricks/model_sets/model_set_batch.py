"""Publish one complete bounded model-set result with a shared Delta receipt."""

import hashlib
import importlib
import json
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from skyulf.integrations.databricks.shared._local_frames import frame_bytes, output_scalar

from ...mlflow.registration.registry import ResolvedModel
from ..data.admission import BatchConflictError, PublishAdmission, validate_admission
from ..data.delta_io.cdf_recovery import (
    CdfHistoryExpired,
    CdfRecoveryRequired,
    check_recovery_state,
    make_recovery_request,
    normalize_cdf_error,
    recovery_receipt_fields,
    validate_recovery_binding,
)
from ..data.delta_io.delta import DeltaPublishError, table_identity
from ..feature_store.scoring import (
    feature_source_columns,
    validate_feature_continuation,
    validate_feature_receipt,
    validate_feature_snapshot,
    validate_feature_source,
)
from ..scoring.batch.spark_scoring import (
    DistributedRows,
    SparkSetExecution,
    complete_distributed_set,
    prepare_spark_set_execution,
    read_distributed_rows,
    score_distributed_set,
)
from ..scoring.incremental.local_incremental import (
    SourceChangeRequiresRebuild,
    bounded_frame,
    check_incremental_bootstrap,
    last_receipt,
    latest_source_version,
    require_incremental_change_feed,
    select_incremental_rows,
    validate_source_change_policy,
)
from ..scoring.shared.prediction_output import OUTPUT_TYPES, check_existing_table
from ..shared._contracts import column_name, table_name
from .model_set_output import (
    METADATA_COLUMNS,
    provision_publication_views,
    publication_columns,
    publication_policy,
    publication_views,
)

if TYPE_CHECKING:
    from ....inference.model_set import ModelSetArtifact

_METADATA = METADATA_COLUMNS


@dataclass(frozen=True)
class ModelSetBatchResult:
    """Describe the complete publication or an already committed source watermark."""

    source_end_version: int
    input_count: int
    output_count: int
    commit_version: int
    manifest: dict[str, Any] | None
    noop: bool


def _plan_publication(
    previous: dict[str, Any] | None,
    *,
    set_digest: str,
    mode: str,
    source_version: int,
    release_changed: bool = False,
) -> tuple[int | None, str, bool]:
    """Choose a complete snapshot or insert window without hiding set changes."""
    if mode not in {"incremental_append", "full_rebuild"}:
        raise ValueError("Model-set mode must be incremental_append or full_rebuild.")
    if previous is None:
        return None, "append", False
    prior = int(previous["source_end_version"])
    if source_version < prior:
        raise BatchConflictError("Source version moved behind the committed watermark.")
    changed = release_changed or previous["model_set_digest"] != set_digest
    if changed and mode == "full_rebuild":
        return None, "overwrite", False
    return prior, "append", prior == source_version


def _validate_request(
    model: ResolvedModel,
    artifact: "ModelSetArtifact",
    source: str,
    target: str,
    mode: str,
    max_rows: int,
    max_bytes: int,
) -> None:
    """Reject ambiguous destinations and identities before contacting Spark."""
    if not isinstance(model, ResolvedModel):
        raise TypeError("Model-set scoring requires a pinned ResolvedModel.")
    from ....inference.model_set import ModelSetArtifact  # noqa: PLC0415

    if not isinstance(artifact, ModelSetArtifact):
        raise TypeError("Model-set scoring requires a ModelSetArtifact.")
    if model.digest != artifact.manifest.set_sha256:
        raise ValueError("Pinned model-set digest differs from the loaded artifact.")
    _validate_model_reference(model)
    if table_name(source).casefold() == table_name(target).casefold():
        raise ValueError("Model-set output must differ from the source table.")
    _plan_publication(None, set_digest=model.digest, mode=mode, source_version=0)
    for limit in (max_rows, max_bytes):
        if type(limit) is not int or limit <= 0:
            raise ValueError("Model-set row and byte limits must be positive integers.")


def _validate_model_reference(model: ResolvedModel) -> None:
    """Require a registry identity whose URI cannot follow a mutable alias."""
    version = model.version
    if not isinstance(version, str) or not version.isascii() or not version.isdigit():
        raise ValueError("Model-set version must be a positive concrete version.")
    if int(version) <= 0 or model.model_uri != f"models:/{model.name}/{version}":
        raise ValueError("Model-set URI must use the pinned positive concrete version.")


def _table_columns(
    artifact: "ModelSetArtifact", publication: Any = None
) -> tuple[tuple[str, str, str], ...]:
    """Derive a fixed output schema that includes set provenance on every row."""
    from ....inference.model_set_scoring import model_set_output_schema  # noqa: PLC0415

    columns = []
    for spec in model_set_output_schema(artifact):
        column_name(spec.name)
        if spec.dtype not in OUTPUT_TYPES:
            raise ValueError(f"Unsupported model-set Delta dtype: {spec.dtype}.")
        kind, sql = OUTPUT_TYPES[spec.dtype]
        columns.append((spec.name, kind, sql))
    columns.extend((name, "string", "STRING") for name in _METADATA)
    if len({name.casefold() for name, _, _ in columns}) != len(columns):
        raise ValueError("Model-set output columns collide with metadata.")
    selected = publication_columns(artifact, publication)
    return tuple(column for column in columns if column[0] in selected)


def _provision(
    spark: Any, source: str, target: str, artifact: "ModelSetArtifact", publication: Any = None
) -> None:
    """Create only an absent empty target after validating source and exact output types."""
    columns = _table_columns(artifact, publication)
    frame = spark.table(source)
    validate_feature_source(frame, artifact)
    for name in feature_source_columns(artifact):
        column_name(name)
        if name not in frame.columns:
            raise ValueError(f"Scoring source is missing model-set input {name!r}.")
    for spec in artifact.manifest.record_key_schema:
        if not _key_type_matches(spec.dtype, frame.schema[spec.name].dataType.typeName()):
            raise ValueError("Source key types differ from the model-set manifest.")
    if not spark.catalog.tableExists(target):
        declaration = ", ".join(f"{column_name(name)} {sql}" for name, _, sql in columns)
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {table_name(target)} ({declaration}) USING DELTA"
        ).collect()
    check_existing_table(spark, target, columns)


def _key_type_matches(dtype: str, spark_type: str) -> bool:
    """Accept lossless Spark integer widening while keeping other key types exact."""
    if dtype == "int64":
        return spark_type in {"byte", "short", "integer", "long"}
    expected = OUTPUT_TYPES.get(dtype)
    return expected is not None and spark_type == expected[0]


def _previous_set_receipt(latest: Any, source_id: str, target_id: str) -> dict | None:
    """Require complete set provenance instead of accepting single-model watermarks."""
    previous = last_receipt(latest, source_id, target_id)
    if previous is not None:
        required = {"model_set_name", "model_set_version", "model_set_digest", "set_history"}
        if previous.get("artifact_kind") != "model_set" or not required.issubset(previous):
            raise BatchConflictError("Target receipt belongs to another scoring contract.")
    return previous


def _score_increment(
    artifact: "ModelSetArtifact",
    frame: Any,
    previous: dict | None,
    write_mode: str,
    max_rows: int,
    max_bytes: int,
) -> Any:
    """Propose all component predictions and temporal history before any output write."""
    if isinstance(frame, DistributedRows):
        return score_distributed_set(frame)
    from ....inference.model_set_scoring import score_model_set  # noqa: PLC0415

    state = previous.get("set_history") if previous and write_mode == "append" else None
    _reject_temporal_reset(previous, state, artifact, write_mode)
    if frame.empty:
        frame = _typed_empty_input(artifact)
    result = score_model_set(
        frame,
        artifact,
        max_rows=max_rows,
        max_bytes=max_bytes,
        history_state=state or None,
        bootstrap_history=not state,
    )
    _reject_temporal_reset(previous, result.history, artifact, write_mode)
    return result


def _typed_empty_input(artifact: "ModelSetArtifact") -> Any:
    """Restore declared types that an empty Spark row iterator cannot carry into pandas."""
    import pandas as pd  # noqa: PLC0415

    return pd.DataFrame(
        {
            column.name: _empty_input_column(column.dtype)
            for column in artifact.manifest.input_schema
        }
    )


def _empty_input_column(dtype: str) -> Any:
    """Preserve pandas and recorded Polars dtypes through an empty pandas bridge."""
    import pandas as pd  # noqa: PLC0415
    import polars as pl  # noqa: PLC0415

    if dtype in {"Date", "Categorical"}:
        polars_dtype = {"Date": pl.Date, "Categorical": pl.Categorical}[dtype]
        return pl.Series([], dtype=polars_dtype).to_pandas(use_pyarrow_extension_array=True)
    temporal = re.fullmatch(
        r"Datetime\(time_unit='(ms|us|ns)', time_zone=(None|'([^']+)')\)", dtype
    )
    if temporal:
        unit, _, timezone = temporal.groups()
        pandas_dtype = (
            f"datetime64[{unit}, {timezone}]" if timezone is not None else f"datetime64[{unit}]"
        )
        return pd.Series(dtype=pandas_dtype)
    aliases = {"String": "string", "Utf8": "string", "Boolean": "boolean"}
    return pd.Series(dtype=aliases.get(dtype, dtype))


def _reject_temporal_reset(
    previous: dict | None,
    history: Any,
    artifact: "ModelSetArtifact",
    write_mode: str,
) -> None:
    """Changing either side of a temporal release needs a complete history rebuild."""
    if (
        history
        and previous is not None
        and write_mode == "append"
        and previous["model_set_digest"] != artifact.manifest.set_sha256
    ):
        raise ValueError("Changing a temporal model set requires full_rebuild.")


def _publication_receipt(
    model: ResolvedModel,
    source_id: str,
    target_id: str,
    prior: int | None,
    upper: int,
    target_version: int,
    rows: int,
    state: dict,
    *,
    source_change_policy: str = "reject",
    source_rebuilt: bool = False,
    write_mode: str = "append",
    recovery_request: dict[str, Any] | None = None,
    execution: SparkSetExecution | None = None,
    feature_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind provenance, progress and component continuation to one atomic write."""
    receipt = {
        "skyulf_mode": "incremental_append",
        "artifact_kind": "model_set",
        "source_table_id": source_id,
        "target_table_id": target_id,
        "source_start_version": prior + 1 if prior is not None else None,
        "source_end_version": upper,
        "expected_target_version": target_version,
        "model_set_name": model.name,
        "model_set_version": model.version,
        "model_set_digest": model.digest,
        "input_count": rows,
        "output_count": rows,
        "set_history": state,
        "source_change_policy": source_change_policy,
        "source_rebuilt": source_rebuilt,
        "write_mode": write_mode,
    }
    if recovery_request is not None:
        receipt.update(recovery_receipt_fields(recovery_request))
    receipt.update(feature_evidence or {})
    if execution is not None:
        receipt.update(
            inference_mode="spark",
            spark_udf_env_manager=execution.env_manager,
            spark_udf_prediction_batch_rows=execution.prediction_batch_rows,
        )
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > 60 * 1024:
        raise ValueError("Model-set publication receipt exceeds its 60 KiB budget.")
    digest = hashlib.sha256(encoded).hexdigest()
    return receipt | {"request_digest": digest, "run_id": digest}


def _output_frame(
    spark: Any,
    frame: Any,
    artifact: "ModelSetArtifact",
    model: ResolvedModel,
    target: str,
    receipt: dict[str, Any],
    write_mode: str,
    max_bytes: int,
) -> Any:
    """Materialize one typed complete result and reject previously published keys."""
    if isinstance(frame, DistributedRows):
        return complete_distributed_set(
            spark,
            frame,
            target,
            receipt,
            _METADATA,
            artifact.manifest.record_key_columns,
            write_mode,
        )
    frame = frame.copy(deep=True)
    for name in _METADATA:
        frame[name] = receipt[name]
    if frame_bytes(frame) > max_bytes:
        raise ValueError("Model-set output exceeds max_bytes.")
    output = spark.createDataFrame(
        [
            tuple(output_scalar(value) for value in row)
            for row in frame.itertuples(index=False, name=None)
        ],
        schema=spark.table(target).select(*frame.columns).schema,
    )
    keys = list(artifact.manifest.record_key_columns)
    if (
        write_mode == "append"
        and output.select(*keys)
        .join(spark.table(target).select(*keys), on=keys, how="left_semi")
        .limit(1)
        .count()
    ):
        raise BatchConflictError("Source key already has a published model-set prediction.")
    return output


def _commit_set(
    spark: Any,
    output: Any,
    source: str,
    target: str,
    receipt: dict[str, Any],
    mode: str,
) -> int:
    """Commit once under admission and verify exact receipt after any write outcome."""
    source_id, target_id = receipt["source_table_id"], receipt["target_table_id"]
    if table_identity(spark, source) != source_id or table_identity(spark, target) != target_id:
        raise BatchConflictError("Source or target identity changed while scoring the set.")
    expected = receipt["expected_target_version"]
    if int(latest_source_version(spark, target)["version"]) != expected:
        raise BatchConflictError("Target changed while scoring the set.")
    validate_feature_receipt(spark, receipt)
    try:
        output.write.format("delta").mode(mode).option("mergeSchema", "false").option(
            "partitionOverwriteMode", "static"
        ).option("overwriteSchema", "false").option(
            "txnAppId", f"skyulf-model-set:{target_id}"
        ).option("txnVersion", expected + 1).option(
            "userMetadata", json.dumps(receipt, sort_keys=True, allow_nan=False)
        ).saveAsTable(target)
    except Exception as exc:  # noqa: BLE001 - preserve uncertain transport outcomes
        raise DeltaPublishError(
            "Model-set write outcome unknown; inspect target receipt before retry."
        ) from exc
    committed = latest_source_version(spark, target)
    recorded = _previous_set_receipt(committed, source_id, target_id)
    if recorded != receipt:
        raise DeltaPublishError("Model-set output has no matching committed receipt.")
    return int(committed["version"])


def run_model_set_batch(
    spark: Any,
    model: ResolvedModel,
    artifact: "ModelSetArtifact",
    *,
    source_table: str,
    prediction_table: str,
    admission: PublishAdmission | None,
    model_change_mode: str = "incremental_append",
    max_rows: int = 100_000,
    max_bytes: int = 64 * 1024 * 1024,
    publication: dict[str, Any] | None = None,
    source_change_policy: str = "reject",
    recovery_request: dict[str, Any] | None = None,
    inference_mode: str = "local",
    spark_udf_env_manager: str = "virtualenv",
    spark_udf_prediction_batch_rows: int = 10_000,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> ModelSetBatchResult:
    """Publish all keyed component and rule outcomes together, or publish nothing.

    Source corrections fail by default. With source_change_policy='rebuild_on_change',
    readable CDF updates/deletes trigger a complete pinned snapshot rebuild with
    the selected set, including temporal history. Inserts continue to append.
    All writers share non-expiring admission.
    Full rebuild replaces the complete target atomically on set-release changes;
    schema changes require a separately provisioned compatible target. Temporal
    set changes require full rebuild to reconstruct component history safely.
    """
    source_change_policy = validate_source_change_policy(source_change_policy)
    _validate_request(
        model, artifact, source_table, prediction_table, model_change_mode, max_rows, max_bytes
    )
    execution = prepare_spark_set_execution(
        spark,
        artifact,
        model.model_uri,
        inference_mode,
        spark_udf_env_manager,
        spark_udf_prediction_batch_rows,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
    publication = publication_policy(publication)
    publication_columns(artifact, publication)
    views = publication_views(artifact, prediction_table, publication)
    if any(view.name.casefold() == source_table.casefold() for view in views):
        raise ValueError("Prediction views must differ from the source table.")
    admission = validate_admission(spark, admission)
    source_id = table_identity(spark, source_table)
    require_incremental_change_feed(spark, source_table)
    _provision(spark, source_table, prediction_table, artifact, publication)
    target_id = table_identity(spark, prediction_table)
    with admission.hold(target_id):
        provision_publication_views(spark, prediction_table, views)
        return _run_admitted_set(
            spark,
            model,
            artifact,
            source_table,
            prediction_table,
            source_id,
            target_id,
            model_change_mode,
            max_rows,
            max_bytes,
            publication,
            source_change_policy,
            recovery_request,
            execution,
        )


def _run_admitted_set(
    spark: Any,
    model: ResolvedModel,
    artifact: "ModelSetArtifact",
    source: str,
    target: str,
    source_id: str,
    target_id: str,
    mode: str,
    max_rows: int,
    max_bytes: int,
    publication: dict[str, Any] | None = None,
    source_change_policy: str = "reject",
    recovery_request: dict[str, Any] | None = None,
    execution: SparkSetExecution | None = None,
) -> ModelSetBatchResult:
    """Hold the common target claim through snapshot selection and complete publication."""
    feature_evidence = validate_feature_snapshot(spark, artifact)
    if table_identity(spark, source) != source_id or table_identity(spark, target) != target_id:
        raise BatchConflictError("Source or target changed during model-set admission.")
    functions = importlib.import_module("pyspark.sql.functions")
    latest = latest_source_version(spark, target)
    previous = _previous_set_receipt(latest, source_id, target_id)
    check_incremental_bootstrap(spark, target, previous, functions)
    upper, prior, write_mode, noop = _set_read_plan(
        spark,
        model,
        source,
        target,
        source_id,
        target_id,
        mode,
        previous,
        int(latest["version"]),
        recovery_request,
    )
    validate_feature_continuation(previous, feature_evidence, rebuilding=write_mode == "overwrite")
    if noop:
        return ModelSetBatchResult(upper, 0, 0, int(latest["version"]), previous, True)
    frame, source_rebuilt = _read_set_frame(
        spark,
        model,
        artifact,
        source,
        target,
        source_id,
        target_id,
        prior,
        upper,
        int(latest["version"]),
        functions,
        source_change_policy,
        max_rows,
        max_bytes,
        execution,
    )
    if source_rebuilt:
        prior, write_mode = None, "overwrite"
    if frame.empty and write_mode != "overwrite":
        return ModelSetBatchResult(upper, 0, 0, int(latest["version"]), previous, True)
    scored = _score_increment(artifact, frame, previous, write_mode, max_rows, max_bytes)
    receipt = _publication_receipt(
        model,
        source_id,
        target_id,
        prior,
        upper,
        int(latest["version"]),
        len(frame),
        scored.history,
        source_change_policy=source_change_policy,
        source_rebuilt=source_rebuilt,
        write_mode=write_mode,
        recovery_request=recovery_request,
        execution=execution,
        feature_evidence=feature_evidence,
    )
    selected_columns = [
        name for name in publication_columns(artifact, publication) if name not in _METADATA
    ]
    output = _output_frame(
        spark,
        (
            scored.frame.select(selected_columns)
            if isinstance(scored.frame, DistributedRows)
            else scored.frame[selected_columns]
        ),
        artifact,
        model,
        target,
        receipt,
        write_mode,
        max_bytes,
    )
    committed = _commit_set(spark, output, source, target, receipt, write_mode)
    return ModelSetBatchResult(upper, len(frame), len(scored.frame), committed, receipt, False)


def _set_read_plan(
    spark: Any,
    model: ResolvedModel,
    source: str,
    target: str,
    source_id: str,
    target_id: str,
    mode: str,
    previous: dict | None,
    target_version: int,
    request: dict[str, Any] | None,
) -> tuple[int, int | None, str, bool]:
    """Freeze the recovery snapshot and reject stale requests before prediction work."""
    digest = _model_digest(model)
    if request is not None:
        request = validate_recovery_binding(
            request,
            layout="model_set",
            source_table=source,
            target_table=target,
            model_name=model.name,
            model_version=model.version,
            model_digest=digest,
        )
        if source_id != request["source_table_id"] or target_id != request["target_table_id"]:
            raise BatchConflictError("CDF recovery source or target table ID changed.")
        noop = check_recovery_state(request, previous, target_version)
        return request["source_end_version"], None, "overwrite", noop
    upper = int(latest_source_version(spark, source)["version"])
    prior, write_mode, noop = _plan_publication(
        previous,
        set_digest=digest,
        mode=mode,
        source_version=upper,
        release_changed=_release_changed(previous, model),
    )
    return upper, prior, write_mode, noop


def _read_set_frame(
    spark: Any,
    model: ResolvedModel,
    artifact: "ModelSetArtifact",
    source: str,
    target: str,
    source_id: str,
    target_id: str,
    prior: int | None,
    upper: int,
    target_version: int,
    functions: Any,
    policy: str,
    max_rows: int,
    max_bytes: int,
    execution: SparkSetExecution | None = None,
) -> tuple[Any, bool]:
    """Classify history loss only while reading an incremental window, never while writing."""
    try:
        selected, rebuilt = _select_set_source(spark, source, prior, upper, functions, policy)
        context = normalize_cdf_error() if prior is not None and not rebuilt else nullcontext()
        with context:
            validate_feature_source(selected, artifact)
            columns = feature_source_columns(artifact)
            keys = artifact.manifest.record_key_columns
            frame = (
                read_distributed_rows(selected, columns, keys, execution=execution)
                if execution is not None
                else bounded_frame(selected, columns, keys, max_rows, max_bytes)
            )
        return frame, rebuilt
    except CdfHistoryExpired as error:
        if prior is None:
            raise
        request = make_recovery_request(
            layout="model_set",
            source_table=source,
            source_table_id=source_id,
            target_table=target,
            target_table_id=target_id,
            target_version=target_version,
            source_start_version=prior,
            source_end_version=upper,
            model_name=model.name,
            model_version=model.version,
            model_digest=_model_digest(model),
        )
        raise CdfRecoveryRequired(request) from error


def _model_digest(model: ResolvedModel) -> str:
    """Require the artifact digest used to bind recovery across separate tasks."""
    if not isinstance(model.digest, str):
        raise ValueError("Model-set scoring requires a concrete artifact digest.")
    return model.digest


def _select_set_source(
    spark: Any, source: str, prior: int | None, upper: int, functions: Any, policy: str
) -> tuple[Any, bool]:
    """Recover only readable source corrections using the same pinned upper snapshot."""
    validate_source_change_policy(policy)
    try:
        return select_incremental_rows(spark, source, prior, upper, None, functions), False
    except SourceChangeRequiresRebuild:
        if policy == "reject":
            raise
        return select_incremental_rows(spark, source, None, upper, None, functions), True


def _release_changed(previous: dict | None, model: ResolvedModel) -> bool:
    """Include concrete registry identity even when two releases have identical bytes."""
    if previous is None:
        return False
    return (previous["model_set_name"], previous["model_set_version"]) != (
        model.name,
        model.version,
    )
