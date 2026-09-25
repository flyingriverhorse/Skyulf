"""Reusable bounded local training and scoring orchestration for Databricks jobs.

The caller serializes all lifecycle writes for each model and output. Importing
this module creates no Spark session, registry connection or cloud resource.
"""

import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ..mlflow.challenger import ChallengerLifecycle
from ..mlflow.promotion import (
    AliasChangeReceipt,
    ExclusiveAliasWriterAdmission,
    controlled_champion_version,
    initialize_champion,
    promote_candidate,
    rollback_promotion,
    stage_challenger,
)
from ..mlflow.registry import RegistryModelNotFoundError, resolve_model
from .admission import SingleWriterAdmission
from .local_approval import approve_local_candidate, reject_local_candidate
from .local_incremental import run_incremental_local_batch
from .local_retraining import (
    LocalTrainingSpec,
    read_training_snapshot,
    split_labeled_snapshot,
    train_local_candidate,
)
from .local_sdk import (
    InputSource,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    prepare_local_workflow,
)
from .prediction_output import (
    _IDENTIFIER,
    _TABLE_NAME,
    _activate_prediction_view,
    _managed_prediction_view_exists,
    _scoring_target,
    provision_prediction_table,
)
from .training_dates import training_date_spec

_TABLE_FIELDS = (
    "training_table",
    "score_source_table",
    "prediction_table",
    "model_name",
)


@dataclass(frozen=True, slots=True)
class AutoTrainingOutcome:
    """Expose both candidate evidence and the optional alias change in job output."""

    candidate: Any
    alias_change: AliasChangeReceipt | None


def _selection_mode(config: dict[str, Any]) -> str:
    """Reject a selection policy that could silently load a wrong model."""
    mode = config.get("model_selection_mode", "pinned_version")
    if mode not in ("pinned_version", "auto_champion"):
        raise ValueError("model_selection_mode must be pinned_version or auto_champion.")
    return mode


def _workflow_policies(config: dict[str, Any]) -> tuple[str, str]:
    """Validate independent policies while preserving legacy project behavior.

    Migrate both fields together and remove model_selection_mode. A mixed or
    partial configuration is ambiguous and must fail before any external work.
    """
    fields = {"score_model_selection", "promotion_policy"}
    supplied = fields.intersection(config)
    if supplied:
        if supplied != fields or "model_selection_mode" in config:
            raise ValueError(
                "Set score_model_selection and promotion_policy together, "
                "and remove legacy model_selection_mode."
            )
        selection = config["score_model_selection"]
        policy = config["promotion_policy"]
        if selection not in ("pinned_version", "champion"):
            raise ValueError("score_model_selection must be pinned_version or champion.")
        if policy not in ("manual_approval", "automatic"):
            raise ValueError("promotion_policy must be manual_approval or automatic.")
        return selection, policy
    mode = _selection_mode(config)
    if "model_selection_mode" in config:
        warnings.warn(
            "model_selection_mode is deprecated. Replace auto_champion with "
            "score_model_selection=champion and promotion_policy=automatic; "
            "replace pinned_version with score_model_selection=pinned_version "
            "and promotion_policy=manual_approval.",
            DeprecationWarning,
            stacklevel=3,
        )
    return (
        ("champion", "automatic")
        if mode == "auto_champion"
        else ("pinned_version", "manual_approval")
    )


def resolve_target_config(config: dict[str, Any], bindings: dict[str, str]) -> dict[str, Any]:
    """Bind only UC object names to one validated Bundle target."""
    for name in ("catalog", "input_schema", "output_schema", "metadata_schema"):
        if not _IDENTIFIER.fullmatch(bindings.get(name, "")):
            raise ValueError(f"Invalid {name} for a Unity Catalog identifier.")
    suffix = bindings.get("resource_suffix", "")
    if suffix and (not suffix.startswith("_") or not _IDENTIFIER.fullmatch(suffix)):
        raise ValueError("Invalid resource_suffix for a Unity Catalog identifier.")
    resolved = config.copy()
    for name in _TABLE_FIELDS:
        value = config[name]
        if type(value) is not str:
            raise ValueError(f"{name} must be a string.")
        for key, replacement in bindings.items():
            value = value.replace("{" + key + "}", replacement)
        if not _TABLE_NAME.fullmatch(value):
            raise ValueError(f"{name} must resolve to a three-part UC name.")
        if name in {"prediction_table", "model_name"}:
            schema = "output_schema" if name == "prediction_table" else "metadata_schema"
            expected = f"{bindings['catalog']}.{bindings[schema]}."
            if not value.startswith(expected):
                raise ValueError(f"{name} must use the active target's {schema}.")
            if suffix and not value.endswith(suffix):
                raise ValueError(f"{name} must include the active target's resource_suffix.")
        resolved[name] = value
    return resolved


def _scoring_config(config: dict[str, Any]) -> LocalWorkflowConfig:
    """Bind the pinned local model and source for incremental scoring."""
    return LocalWorkflowConfig(
        runtime="databricks",
        engine=config["engine"],
        source=InputSource(
            kind="uc_table",
            table=config["score_source_table"],
            read_mode="incremental",
            max_rows=config["max_rows"],
            max_bytes=config["max_bytes"],
        ),
        model=ModelSelection(
            kind="local_pipeline",
            name=config["model_name"],
            version=config["model_version"],
            tracking_uri=config.get("tracking_uri", "databricks"),
            registry_uri=config.get("registry_uri", "databricks-uc"),
        ),
        sink=OutputSink(kind="uc_delta", table=config["prediction_table"]),
    )


def _training_spec(config: dict[str, Any]) -> LocalTrainingSpec:
    """Keep the evaluation split and source snapshot identical across actions."""
    return LocalTrainingSpec(
        table=config["training_table"],
        version=config["training_version"],
        start=datetime.fromisoformat(config["start"]),
        holdout_start=datetime.fromisoformat(config["holdout_start"]),
        cutoff=datetime.fromisoformat(config["cutoff"]),
        event_column=config["event_column"],
        result_available_at_column=config["result_available_at_column"],
        record_key_columns=tuple(config["record_key_columns"]),
        input_columns=tuple(config["input_columns"]),
        target_column=config["target_column"],
        max_rows=config["max_rows"],
        max_bytes=config["max_bytes"],
        event_time_parsing=training_date_spec(config.get("event_time_parsing", {})),
        result_time_parsing=training_date_spec(config.get("result_time_parsing", {})),
    )


def _monthly_training_spec(spark: Any, config: dict[str, Any], now: datetime) -> LocalTrainingSpec:
    """Pin one Delta version and a UTC calendar window for monthly training."""
    event_parsing = training_date_spec(config.get("event_time_parsing", {}))
    result_parsing = training_date_spec(config.get("result_time_parsing", {}))
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Monthly training needs a timezone-aware run instant.")
    lookback = config.get("monthly_lookback_months")
    if type(lookback) is not int or not 2 <= lookback <= 120:
        raise ValueError("monthly_lookback_months must be an integer from 2 to 120.")
    cutoff = now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def months_before(count: int) -> datetime:
        """Preserve the first-of-month boundary across year rollover."""
        index = cutoff.year * 12 + cutoff.month - 1 - count
        return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)

    table = config["training_table"]
    if not _TABLE_NAME.fullmatch(table):
        raise ValueError("training_table must be a three-part UC name.")
    latest = (
        spark.sql(f"DESCRIBE HISTORY {table}")
        .select("version")
        .orderBy("version", ascending=False)
        .first()
    )
    if latest is None or type(latest["version"]) is not int:
        raise ValueError("Training source has no concrete Delta version.")
    return LocalTrainingSpec(
        table=table,
        version=latest["version"],
        start=months_before(lookback),
        holdout_start=months_before(1),
        cutoff=cutoff,
        event_column=config["event_column"],
        result_available_at_column=config["result_available_at_column"],
        record_key_columns=tuple(config["record_key_columns"]),
        input_columns=tuple(config["input_columns"]),
        target_column=config["target_column"],
        max_rows=config["max_rows"],
        max_bytes=config["max_bytes"],
        event_time_parsing=event_parsing,
        result_time_parsing=result_parsing,
    )


def _monthly_champion_version(config: dict[str, Any]) -> str | None:
    """Resolve the current champion once while allowing first-model training."""
    try:
        champion = resolve_model(
            config["model_name"],
            alias="champion",
            tracking_uri=config.get("tracking_uri", "databricks"),
            registry_uri=config.get("registry_uri", "databricks-uc"),
        )
    except RegistryModelNotFoundError:
        return None
    return champion.version


def _automatic_promotion(
    spark: Any,
    config: dict[str, Any],
    spec: LocalTrainingSpec,
    candidate: Any,
    *,
    promote: bool = True,
) -> AliasChangeReceipt | None:
    """Record contender evidence separately from any optional champion transition."""
    report = candidate.comparison
    frame = read_training_snapshot(spark, spec)
    _, heldout, _ = split_labeled_snapshot(frame, spec)
    native = pl.from_pandas(heldout) if config["engine"] == "polars" else heldout
    options = {
        "target_column": spec.target_column,
        "admission": ExclusiveAliasWriterAdmission(),
        "max_rows": spec.max_rows,
        "max_bytes": spec.max_bytes,
        "tracking_uri": config.get("tracking_uri", "databricks"),
        "registry_uri": config.get("registry_uri", "databricks-uc"),
    }
    stage_challenger(
        report,
        native,
        expected_champion_version=report.champion_version,
        expected_challenger_version=report.candidate_version,
        **options,
    )
    if not promote:
        return None
    if report.champion_version is None:
        value = report.candidate_metrics[report.metric]
        threshold = report.quality_threshold
        if threshold is None or not (
            value <= threshold if report.metric_direction == "minimize" else value >= threshold
        ):
            return None
        return initialize_champion(report, native, **options)
    if not report.eligible:
        return None
    return promote_candidate(
        report,
        native,
        expected_champion_version=report.champion_version,
        **options,
    )


def run_action(
    spark: Any,
    config: dict[str, Any],
    action: str,
    *,
    experiment_name: str | None = None,
    artifact_path: str | Path | None = None,
    now: datetime | None = None,
    candidate_version: str | None = None,
    comparison_sha256: str | None = None,
    expected_champion_version: str | None = None,
    rejection_reason: str = "",
    promotion_receipt: AliasChangeReceipt | None = None,
) -> Any:
    """Delegate training, scoring and explicit lifecycle actions to existing Core services."""
    tracking_uri = config.get("tracking_uri", "databricks")
    registry_uri = config.get("registry_uri", "databricks-uc")
    selection, policy = _workflow_policies(config)
    if action == "rollback":
        if (
            not isinstance(promotion_receipt, AliasChangeReceipt)
            or promotion_receipt.model_name != config["model_name"]
            or not isinstance(expected_champion_version, str)
        ):
            raise ValueError(
                "Rollback needs a receipt for the configured model and expected champion."
            )
        return rollback_promotion(
            promotion_receipt,
            expected_current_version=expected_champion_version,
            admission=ExclusiveAliasWriterAdmission(),
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
    if action == "reject":
        return reject_local_candidate(
            config,
            candidate_version=candidate_version,
            comparison_sha256=comparison_sha256,
            expected_champion_version=expected_champion_version,
            rejection_reason=rejection_reason,
        )
    if action == "approve":
        if policy != "manual_approval" or "promotion_policy" not in config:
            raise ValueError("Approval requires explicit promotion_policy=manual_approval.")
        return approve_local_candidate(
            spark,
            config,
            candidate_version=candidate_version,
            comparison_sha256=comparison_sha256,
            expected_champion_version=expected_champion_version,
        )
    if action in {"train", "train_monthly"}:
        if experiment_name is None or artifact_path is None:
            raise ValueError("Training needs an experiment and temporary artifact path.")
        monthly = action == "train_monthly"
        if policy == "automatic" and config.get("quality_threshold") is None:
            raise ValueError("Automatic promotion requires an absolute quality_threshold.")
        spec = (
            _monthly_training_spec(spark, config, now or datetime.now(UTC))
            if monthly
            else _training_spec(config)
        )
        if policy == "automatic" or "score_model_selection" in config:
            champion_version = controlled_champion_version(
                config["model_name"], tracking_uri=tracking_uri, registry_uri=registry_uri
            )
            expected = config.get("champion_version")
            if (
                "score_model_selection" in config
                and expected is not None
                and str(expected) != champion_version
            ):
                raise ValueError("champion_version does not match the current champion.")
        else:
            champion_version = _monthly_champion_version(config)
            expected = config.get("champion_version")
            if expected is not None and str(expected) != champion_version:
                raise ValueError("champion_version does not match the current champion.")
        lifecycle = ChallengerLifecycle(
            config["model_name"],
            expected_champion_version=champion_version,
            admission=ExclusiveAliasWriterAdmission(),
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        try:
            candidate = train_local_candidate(
                spark,
                spec,
                config["pipeline"],
                model_name=config["model_name"],
                tracking_uri=tracking_uri,
                registry_uri=registry_uri,
                experiment_name=experiment_name,
                run_name="candidate_training",
                artifact_path=artifact_path,
                metric=config["metric"],
                min_improvement=config["min_improvement"],
                engine=config["engine"],
                champion_version=champion_version,
                quality_threshold=config.get("quality_threshold"),
                on_registered=lifecycle.registered,
                risk_category=config.get("risk_category"),
            )
            alias_change = _automatic_promotion(
                spark,
                config,
                spec,
                candidate,
                promote=policy == "automatic",
            )
        except Exception as error:  # noqa: BLE001 - preserve failure and lifecycle evidence
            try:
                lifecycle.failed()
            except Exception as status_error:  # noqa: BLE001 - retain the original failure
                raise error from status_error
            raise
        if policy == "automatic":
            return AutoTrainingOutcome(
                candidate=candidate,
                alias_change=alias_change,
            )
        return candidate
    if action == "score":
        if selection == "champion":
            champion_version = controlled_champion_version(
                config["model_name"], tracking_uri=tracking_uri, registry_uri=registry_uri
            )
            if champion_version is None:
                raise ValueError("Champion scoring requires a committed champion.")
            config = {**config, "model_version": champion_version}
        target = _scoring_target(config)
        if config.get("model_change_mode", "incremental_append") == "full_rebuild":
            _managed_prediction_view_exists(spark, config["prediction_table"])
        score_config = {**config, "prediction_table": target}
        prepared = prepare_local_workflow(_scoring_config(score_config))
        provision_prediction_table(spark, score_config, prepared)
        result = run_incremental_local_batch(
            spark,
            prepared,
            record_key_columns=tuple(config["record_key_columns"]),
            admission=SingleWriterAdmission(),
        )
        if config.get("model_change_mode", "incremental_append") == "full_rebuild":
            _activate_prediction_view(spark, config["prediction_table"], target)
        return result
    raise ValueError(f"Unknown workflow action: {action}.")
