"""Reusable bounded local training and scoring orchestration for Databricks jobs.

The caller serializes all lifecycle writes for each model and output. Importing
this module creates no Spark session, registry connection or cloud resource.
"""

import json
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl

from skyulf.integrations.mlflow.shared._client import make_registry_client, require_mlflow

from ...mlflow.lifecycle.challenger import ChallengerLifecycle
from ...mlflow.lifecycle.promotion import (
    AliasChangeReceipt,
    ExclusiveAliasWriterAdmission,
    controlled_champion_version,
    initialize_champion,
    promote_candidate,
    rollback_promotion,
    stage_challenger,
)
from ...mlflow.lifecycle.validation import quality_gates_pass
from ...mlflow.registration.registry import RegistryModelNotFoundError, resolve_model
from ..data.admission import SingleWriterAdmission
from ..data.training.training_dates import training_date_spec
from ..observability.reports.local_explanations import validate_explanation_config
from ..scoring.incremental.local_incremental import run_incremental_local_batch
from ..scoring.local_sdk import (
    InputSource,
    LocalWorkflowConfig,
    ModelSelection,
    OutputSink,
    prepare_local_workflow,
)
from ..scoring.shared.prediction_output import (
    IDENTIFIER_PATTERN,
    TABLE_NAME_PATTERN,
    activate_prediction_view,
    managed_prediction_view_exists,
    provision_prediction_table,
    scoring_target,
)
from ..shared._contracts import input_budget_bytes
from ..training.fitting.local_retraining import (
    LocalCandidateResult,
    LocalTrainingSpec,
    read_training_snapshot,
    split_labeled_snapshot,
    train_local_candidate,
    validate_cv_holdout_policy,
)
from ..training.shared.local_training_evidence import (
    load_candidate_evidence,
    validate_training_evidence,
)
from ..training.tuning.local_cv import LocalCVSpec
from .local_approval import approve_local_candidate, reject_local_candidate

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


@dataclass(frozen=True, slots=True)
class BundleActionResult:
    """Expose Core output, optional score handoff and copyable operator inputs."""

    action: str
    result: Any
    score_requested: bool
    next_actions: dict[str, dict[str, str]]


def next_actions(result: Any, policy: str) -> dict[str, dict[str, str]]:
    """Expose the exact evidence accepted by existing Core approval and rollback APIs."""
    actions: dict[str, dict[str, str]] = {}
    candidate = result.candidate if isinstance(result, AutoTrainingOutcome) else result
    if isinstance(candidate, LocalCandidateResult) and policy == "manual_approval":
        for action in ("approve", "reject"):
            actions[action] = {
                "lifecycle_action": action,
                "candidate_version": candidate.model_version,
                "expected_champion_version": candidate.comparison.champion_version or "none",
            }
    receipt = result.alias_change if isinstance(result, AutoTrainingOutcome) else result
    if isinstance(receipt, AliasChangeReceipt) and receipt.kind == "promotion":
        actions["rollback"] = {
            "lifecycle_action": "rollback",
            "expected_champion_version": receipt.new_version,
            "promotion_receipt_json": json.dumps(
                asdict(receipt), sort_keys=True, separators=(",", ":")
            ),
        }
    return actions


def build_bundle_result(config: dict[str, Any], action: str, result: Any) -> BundleActionResult:
    """Derive operator inputs and score handoff only from a completed typed outcome."""
    _, policy = workflow_policies(config)
    receipt = result.alias_change if isinstance(result, AutoTrainingOutcome) else result
    score_requested = (
        config["score_handoff"] == "after_alias_change"
        and action in {"train", "approve", "rollback"}
        and isinstance(receipt, AliasChangeReceipt)
        and receipt.kind in {"initial", "promotion", "rollback"}
    )
    return BundleActionResult(action, result, score_requested, next_actions(result, policy))


def _selection_mode(config: dict[str, Any]) -> str:
    """Reject a selection policy that could silently load a wrong model."""
    mode = config.get("model_selection_mode", "pinned_version")
    if mode not in ("pinned_version", "auto_champion"):
        raise ValueError("model_selection_mode must be pinned_version or auto_champion.")
    return mode


def workflow_policies(config: dict[str, Any]) -> tuple[str, str]:
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
        if not IDENTIFIER_PATTERN.fullmatch(bindings.get(name, "")):
            raise ValueError(f"Invalid {name} for a Unity Catalog identifier.")
    suffix = bindings.get("resource_suffix", "")
    if suffix and (not suffix.startswith("_") or not IDENTIFIER_PATTERN.fullmatch(suffix)):
        raise ValueError("Invalid resource_suffix for a Unity Catalog identifier.")
    resolved = config.copy()
    for name in _TABLE_FIELDS:
        resolved[name] = bind_target_name(name, config[name], bindings, suffix)
    return resolved


def bind_target_name(name: str, value: Any, bindings: dict[str, str], suffix: str) -> str:
    """Resolve one UC name and enforce the target's output ownership."""
    if type(value) is not str:
        raise ValueError(f"{name} must be a string.")
    for key, replacement in bindings.items():
        value = value.replace("{" + key + "}", replacement)
    if not TABLE_NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must resolve to a three-part UC name.")
    if name in {"prediction_table", "model_name"}:
        schema = "output_schema" if name == "prediction_table" else "metadata_schema"
        expected = f"{bindings['catalog']}.{bindings[schema]}."
        if not value.startswith(expected):
            raise ValueError(f"{name} must use the active target's {schema}.")
        if suffix and not value.endswith(suffix):
            raise ValueError(f"{name} must include the active target's resource_suffix.")
    return value


def _scoring_config(config: dict[str, Any]) -> LocalWorkflowConfig:
    """Bind the pinned local model and source for incremental scoring."""
    return LocalWorkflowConfig(
        runtime="databricks",
        engine=config["engine"],
        inference_mode=config.get("inference_mode", "local"),
        spark_udf_env_manager=config.get("spark_udf_env_manager", "virtualenv"),
        spark_udf_prediction_batch_rows=config.get("spark_udf_prediction_batch_rows", 10_000),
        source=InputSource(
            kind="uc_table",
            table=config["score_source_table"],
            read_mode="incremental",
            max_rows=config["max_rows"],
            max_bytes=input_budget_bytes(config.get("max_input_mb")),
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


def training_spec(config: dict[str, Any]) -> LocalTrainingSpec:
    """Keep the evaluation split and source snapshot identical across actions."""
    training_window_mode(config)
    version = config.get("training_version")
    if type(version) is not int or version < 0:
        raise ValueError(
            "Set training_version to an explicit nonnegative Delta version before train."
        )
    return LocalTrainingSpec(
        table=config["training_table"],
        version=version,
        split_strategy=config.get("split_strategy", "random"),
        test_size=config.get("test_size", 0.2),
        random_state=config.get("random_state", 42),
        stratify=config.get("stratify", False),
        training_sample_rows=config.get("training_sample_rows"),
        training_sample_seed=config.get("training_sample_seed", 42),
        start=_optional_boundary(config, "start"),
        holdout_start=_optional_boundary(config, "holdout_start"),
        cutoff=_optional_boundary(config, "cutoff"),
        event_column=config.get("event_column"),
        group_column=config.get("cv_group_column"),
        weight_column=config.get("weight_column"),
        reserved_weight_columns=tuple(config.get("reserved_weight_columns", ())),
        weights_python_source=config.get("weights_python_source"),
        weights_python_sha256=config.get("weights_python_sha256"),
        filter_unavailable_results=config.get("filter_unavailable_results", False),
        drop_missing_labels=(
            True
            if config.get("training_layout") == "multi_target"
            else config.get("drop_missing_labels", False)
        ),
        result_available_at_column=config.get("result_available_at_column"),
        result_cutoff=_optional_boundary(config, "result_cutoff"),
        record_key_columns=tuple(config["record_key_columns"]),
        input_columns=tuple(config["input_columns"]),
        target_column=config["target_column"],
        pre_split_steps=tuple(config.get("pre_split_steps", ())),
        max_rows=config["max_rows"],
        max_bytes=input_budget_bytes(config.get("max_input_mb")),
        event_time_parsing=training_date_spec(
            config.get("event_time_parsing") if config.get("event_time_parsing") is not None else {}
        ),
        result_time_parsing=training_date_spec(
            config.get("result_time_parsing")
            if config.get("result_time_parsing") is not None
            else {}
        ),
    )


def _optional_boundary(config: dict[str, Any], field: str) -> datetime | None:
    """Decode an explicitly supplied boundary without inventing an inactive date."""
    value = config.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp with timezone.")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp with timezone.") from exc


def _validate_rolling_window(config: dict[str, Any]) -> None:
    """Validate calendar history, holdout duration and timezone in that order."""
    months = config.get("monthly_lookback_months")
    minimum = 2 if config.get("split_strategy") == "temporal" else 1
    if type(months) is not int or not minimum <= months <= 120:
        raise ValueError(f"monthly_lookback_months must be an integer from {minimum} to 120.")
    if config.get("split_strategy") == "temporal":
        holdout = config.get("holdout_months", 1)
        if type(holdout) is not int or not 1 <= holdout < months:
            raise ValueError(
                "holdout_months must be an integer from 1 to monthly_lookback_months - 1."
            )
    zone = config.get("window_timezone")
    if not isinstance(zone, str) or not zone:
        raise ValueError("Rolling-calendar selection requires window_timezone.")
    try:
        ZoneInfo(zone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("window_timezone must name an IANA timezone.") from exc


def _validate_result_lag(config: dict[str, Any]) -> None:
    """Reject unavailable-result lag settings outside their active policy."""
    if config.get("filter_unavailable_results") is True:
        lag = config.get("result_availability_lag_hours", 0)
        if type(lag) is not int or not 0 <= lag <= 87600:
            raise ValueError("result_availability_lag_hours must be an integer from 0 to 87600.")
    elif config.get("result_availability_lag_hours") is not None:
        raise ValueError("Inactive result filtering requires null result_availability_lag_hours.")


def _validate_window_event(config: dict[str, Any], mode: str) -> None:
    """Require event columns only for time-based source selection."""
    if mode == "full_snapshot":
        if (
            config.get("event_column") is not None
            or config.get("split_strategy", "random") == "temporal"
        ):
            raise ValueError(
                "Full-snapshot selection requires inactive event fields and random splitting."
            )
    elif not config.get("event_column"):
        raise ValueError("Window selection requires an explicit event_column.")


def _validate_calendar_controls(config: dict[str, Any], mode: str) -> None:
    """Keep existing calendar validation separate from elapsed-day settings."""
    if mode == "rolling_calendar":
        _validate_rolling_window(config)
    elif (
        config.get("monthly_lookback_months") is not None
        or config.get("window_timezone") is not None
    ):
        raise ValueError(
            "Non-rolling selection requires null monthly_lookback_months and window_timezone."
        )
    if (mode != "rolling_calendar" or config.get("split_strategy") != "temporal") and config.get(
        "holdout_months"
    ) is not None:
        raise ValueError("holdout_months must be null outside rolling temporal selection.")


def _validate_daily_controls(config: dict[str, Any], mode: str) -> None:
    """Require bounded integer day counts only for their active selection policy."""
    if mode != "rolling_days":
        for field in ("lookback_days", "holdout_days"):
            if config.get(field) is not None:
                raise ValueError(f"{field} must be null outside rolling_days selection.")
        return
    days = config.get("lookback_days")
    if type(days) is not int or not 1 <= days <= 36500:
        raise ValueError("lookback_days must be an integer from 1 to 36500.")
    holdout = config.get("holdout_days")
    if config.get("split_strategy") == "temporal":
        if type(holdout) is not int or not 1 <= holdout < days:
            raise ValueError("holdout_days must be an integer from 1 to lookback_days - 1.")
    elif holdout is not None:
        raise ValueError("holdout_days must be null outside rolling_days temporal selection.")


def training_window_mode(config: dict[str, Any]) -> str:
    """Validate source selection independently of the random/temporal evaluation split."""
    mode = config.get("training_window_mode", "full_snapshot")
    if mode not in ("full_snapshot", "fixed_window", "rolling_calendar", "rolling_days"):
        raise ValueError(
            "training_window_mode must be full_snapshot, fixed_window, "
            "rolling_calendar or rolling_days."
        )
    _validate_window_event(config, mode)
    _validate_calendar_controls(config, mode)
    _validate_daily_controls(config, mode)
    _validate_result_lag(config)
    return mode


def training_settings(config: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Resolve data windows independently of how training was triggered."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Training needs a timezone-aware run instant.")
    settings = dict(config)
    mode = training_window_mode(config)
    if mode == "rolling_calendar":
        lookback = config["monthly_lookback_months"]
        zone = ZoneInfo(config["window_timezone"])
        cutoff = now.astimezone(zone).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0, fold=0
        )

        def months_before(count: int) -> datetime:
            """Preserve the first-of-month boundary across year rollover."""
            index = cutoff.year * 12 + cutoff.month - 1 - count
            return datetime(index // 12, index % 12 + 1, 1, tzinfo=zone)

        settings.update(
            start=months_before(lookback).isoformat(),
            holdout_start=(
                months_before(config.get("holdout_months", 1)).isoformat()
                if config.get("split_strategy") == "temporal"
                else None
            ),
            cutoff=cutoff.isoformat(),
        )
    elif mode == "rolling_days":
        cutoff = now.astimezone(UTC)
        settings.update(
            start=(cutoff - timedelta(days=config["lookback_days"])).isoformat(),
            holdout_start=(
                (cutoff - timedelta(days=config["holdout_days"])).isoformat()
                if config.get("split_strategy") == "temporal"
                else None
            ),
            cutoff=cutoff.isoformat(),
        )
    if config.get("filter_unavailable_results", False) and config.get("result_cutoff") is None:
        settings["result_cutoff"] = (
            now.astimezone(UTC) - timedelta(hours=config.get("result_availability_lag_hours", 0))
        ).isoformat()
    # A placeholder allows validation before resolving an unspecified version.
    if settings.get("training_version") is None:
        settings["training_version"] = 0
    return settings


def resolve_training_spec(spark: Any, config: dict[str, Any], now: datetime) -> LocalTrainingSpec:
    """Pin an explicit or latest snapshot once, using the configured data window."""
    spec = training_spec(training_settings(config, now))
    validate_cv_holdout_policy(spec, LocalCVSpec.from_workflow(config))
    if config.get("training_version") is not None:
        return spec
    table = spec.table
    latest = (
        spark.sql(f"DESCRIBE HISTORY {table}")
        .select("version")
        .orderBy("version", ascending=False)
        .first()
    )
    if latest is None or type(latest["version"]) is not int or latest["version"] < 0:
        raise ValueError("Training source has no concrete Delta version.")
    return replace(spec, version=latest["version"])


def _current_champion_version(config: dict[str, Any]) -> str | None:
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


def prepare_training(
    spark: Any,
    config: dict[str, Any],
    *,
    policy: str,
    now: datetime | None,
) -> tuple[LocalTrainingSpec, LocalCVSpec, str | None]:
    """Validate training policy and pin source/champion for either execution adapter.

    Callers retain their expected-champion compatibility checks and own all
    run creation, fitting and publication. Score model selection stays unchanged.
    """
    if policy == "automatic" and config.get("quality_threshold") is None:
        raise ValueError("Automatic promotion requires an absolute quality_threshold.")
    cv = LocalCVSpec.from_workflow(config)
    validate_explanation_config(config["pipeline"])
    cv.validate_pipeline(
        config["pipeline"],
        target_column=config["target_column"],
        event_column=config.get("event_column"),
    )
    spec = resolve_training_spec(spark, config, now or datetime.now(UTC))
    champion = (
        controlled_champion_version(
            config["model_name"],
            tracking_uri=config.get("tracking_uri", "databricks"),
            registry_uri=config.get("registry_uri", "databricks-uc"),
        )
        if policy == "automatic" or "score_model_selection" in config
        else _current_champion_version(config)
    )
    return spec, cv, champion


def automatic_promotion(
    spark: Any,
    config: dict[str, Any],
    spec: LocalTrainingSpec,
    candidate: Any,
    *,
    promote: bool = True,
) -> AliasChangeReceipt | None:
    """Record contender evidence separately from any optional champion transition."""
    if not isinstance(candidate, LocalCandidateResult):
        raise ValueError("Automatic replay requires a saved candidate result.")
    report = candidate.comparison
    tracking_uri = config.get("tracking_uri", "databricks")
    registry_uri = config.get("registry_uri", "databricks-uc")
    client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
    saved_report, spec, saved_engine, filter_evidence = load_candidate_evidence(
        client,
        candidate.model_name,
        candidate.model_version,
        candidate.comparison_sha256,
        registry_uri=registry_uri,
    )
    if saved_report != report or saved_engine != config["engine"]:
        raise ValueError("Saved candidate evidence differs from automatic comparison.")
    frame = read_training_snapshot(spark, spec)
    _, heldout, _ = split_labeled_snapshot(frame, spec, engine=config["engine"])
    if filter_evidence is not None:
        validate_training_evidence(
            filter_evidence,
            spec,
            project_source_sha256=filter_evidence["project_source_sha256"],
            heldout=heldout,
        )
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
    return _promote_staged_candidate(report, native, options, promote)


def _promote_staged_candidate(
    report: Any, native: Any, options: dict[str, Any], promote: bool
) -> AliasChangeReceipt | None:
    """Apply promotion policy after contender evidence has been staged."""
    if not promote:
        return None
    if report.champion_version is None:
        if report.quality_threshold is None or not quality_gates_pass(report):
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


def _run_training_action(
    spark: Any,
    config: dict[str, Any],
    *,
    policy: str,
    tracking_uri: str,
    registry_uri: str,
    experiment_name: str | None,
    artifact_path: str | Path | None,
    now: datetime | None,
) -> LocalCandidateResult | AutoTrainingOutcome:
    """Train a candidate and preserve failure evidence before applying promotion policy."""
    if experiment_name is None or artifact_path is None:
        raise ValueError("Training needs an experiment and temporary artifact path.")
    spec, cv, champion_version = prepare_training(spark, config, policy=policy, now=now)
    expected = config.get("champion_version")
    # Legacy automatic selection ignored an explicit champion pin. Preserve
    # that SDK compatibility; current policies and durable tasks check it.
    if (
        (policy != "automatic" or "score_model_selection" in config)
        and expected is not None
        and str(expected) != champion_version
    ):
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
            quality_gates=config.get("quality_gates"),
            on_registered=lifecycle.registered,
            risk_category=config.get("risk_category"),
            cv=cv,
            evaluation_charts=config.get("evaluation_charts"),
        )
        alias_change = automatic_promotion(
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


def _run_scoring_action(
    spark: Any,
    config: dict[str, Any],
    *,
    selection: str,
    tracking_uri: str,
    registry_uri: str,
    recovery_request: dict[str, Any] | None = None,
) -> Any:
    """Resolve the scoring pin and activate rebuilt output only after a successful batch."""
    if selection == "champion":
        champion_version = controlled_champion_version(
            config["model_name"], tracking_uri=tracking_uri, registry_uri=registry_uri
        )
        if champion_version is None:
            raise ValueError("Champion scoring requires a committed champion.")
        config = {**config, "model_version": champion_version}
    target = scoring_target(config)
    if config.get("model_change_mode", "incremental_append") == "full_rebuild":
        managed_prediction_view_exists(spark, config["prediction_table"])
    score_config = {**config, "prediction_table": target}
    prepared = prepare_local_workflow(_scoring_config(score_config))
    from ..scoring.batch.spark_scoring import validate_prepared_spark  # noqa: PLC0415

    if config.get("inference_mode", "local") == "spark":
        validate_prepared_spark(prepared)
    if recovery_request is None:
        provision_prediction_table(spark, score_config, prepared)
    result = run_incremental_local_batch(
        spark,
        prepared,
        record_key_columns=tuple(config["record_key_columns"]),
        admission=SingleWriterAdmission(),
        **({"recovery_request": recovery_request} if recovery_request is not None else {}),
    )
    if config.get("model_change_mode", "incremental_append") == "full_rebuild":
        activate_prediction_view(spark, config["prediction_table"], target)
    return result


def _run_rollback_action(
    config: dict[str, Any],
    promotion_receipt: AliasChangeReceipt | None,
    expected_champion_version: str | None,
    tracking_uri: str,
    registry_uri: str,
) -> AliasChangeReceipt:
    """Validate rollback identity before attempting its alias transition."""
    if (
        not isinstance(promotion_receipt, AliasChangeReceipt)
        or promotion_receipt.model_name != config["model_name"]
        or not isinstance(expected_champion_version, str)
    ):
        raise ValueError("Rollback needs a receipt for the configured model and expected champion.")
    return rollback_promotion(
        promotion_receipt,
        expected_current_version=expected_champion_version,
        admission=ExclusiveAliasWriterAdmission(),
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )


def _validate_action_layout(config: dict[str, Any], action: str) -> None:
    """Keep single-call training from silently ignoring competition candidates."""
    if action == "train" and config.get("training_layout") == "model_competition":
        raise ValueError("Model competition training requires the phased lifecycle job.")


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
    recovery_request: dict[str, Any] | None = None,
) -> Any:
    """Delegate training, scoring and explicit lifecycle actions to existing Core services."""
    _validate_action_layout(config, action)
    if recovery_request is not None and action != "score":
        raise ValueError("CDF recovery is only available for scoring.")
    tracking_uri = config.get("tracking_uri", "databricks")
    registry_uri = config.get("registry_uri", "databricks-uc")
    selection, policy = workflow_policies(config)
    if action == "rollback":
        return _run_rollback_action(
            config, promotion_receipt, expected_champion_version, tracking_uri, registry_uri
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
    if action == "train":
        return _run_training_action(
            spark,
            config,
            policy=policy,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            experiment_name=experiment_name,
            artifact_path=artifact_path,
            now=now,
        )
    if action == "score":
        return _run_scoring_action(
            spark,
            config,
            selection=selection,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            recovery_request=recovery_request,
        )
    raise ValueError(f"Unknown workflow action: {action}.")
