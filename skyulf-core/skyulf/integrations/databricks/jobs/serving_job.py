"""Compose opt-in scheduled serving work from verified native integration services."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ...mlflow.registration.registry import load_registered_pipeline, resolve_model
from ..data.admission import SingleWriterAdmission
from ..feature_store.online_publication import (
    OnlinePublicationSpec,
    online_publication_status,
    publish_online_features,
)
from ..observability.monitoring.monitoring_config import MonitorConfig
from ..serving.rollout import (
    RolloutResult,
    advance_rollout,
    reconcile_rollout,
    rollout_plan_from_dict,
)
from ..serving.rollout_evidence import (
    RolloutEvidenceResult,
    RolloutHealthPolicy,
    observe_bootstrap_rollout,
    observe_live_rollout,
)
from ..serving.rollout_promotion import approval_config_digest, promote_completed_rollout
from ..serving.rollout_store import MLflowRolloutStore
from ..shared.json_contracts import finite_json_digest
from ..training.shared.training_evidence import load_candidate_evidence


def _enabled(settings: dict[str, Any]) -> bool:
    """Require an explicit switch and exclusive writer ownership for generated jobs."""
    if not isinstance(settings, dict) or type(settings.get("enabled")) is not bool:
        raise ValueError("Serving job requires an explicit enabled boolean.")
    if not settings["enabled"]:
        return False
    if settings.get("exclusive_writer") is not True:
        raise ValueError("Serving job requires exclusive_writer=true and exclusive writer ACLs.")
    return True


def _approval(settings: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Bind scheduled settings to the original saved automatic promotion request."""
    value = dict(settings["approval"])
    for name, default in (("tracking_uri", "databricks"), ("registry_uri", "databricks-uc")):
        expected = settings.get(name, default)
        if value.get(name, expected) != expected:
            raise ValueError("Serving job registry configuration differs from approval settings.")
        value[name] = expected
    pin = record.get("promotion")
    if not isinstance(pin, dict) or pin.get("auto_promote") is not True:
        raise ValueError("Daily rollout job requires initialized automatic promotion evidence.")
    if settings.get("comparison_sha256") != pin.get("comparison_sha256"):
        raise ValueError("Daily rollout comparison differs from the initialized pin.")
    if approval_config_digest(value) != pin.get("config_sha256"):
        raise ValueError("Daily rollout approval policy differs from the initialized pin.")
    return value


def persist_rollout_observation(store: MLflowRolloutStore, result: RolloutEvidenceResult) -> None:
    """Read back the complete producer record before allowing a traffic decision."""
    payload = {"evidence": asdict(result.evidence), "details": result.details}
    digest = result.details_digest
    path = f"rollout/observations/{digest}.json"
    store.client.log_dict(store.run_id, payload, path)
    with TemporaryDirectory(prefix="skyulf-rollout-observation-") as directory:
        downloaded = store.client.download_artifacts(store.run_id, path, directory)
        actual = json.loads(Path(downloaded).read_text(encoding="utf-8"))
    if finite_json_digest(actual) != digest:
        raise ValueError("Saved rollout observation differs from the producer evidence.")


def _observe(
    spark: Any,
    client: Any,
    registry_client: Any,
    settings: dict[str, Any],
    record: dict[str, Any],
    result: RolloutResult,
    approval: dict[str, Any],
) -> RolloutEvidenceResult:
    """Use native targeted smoke at zero traffic and actual live evidence thereafter."""
    plan = rollout_plan_from_dict(record["plan"])
    now = datetime.now(UTC)
    if result.state.challenger_percentage == 0:
        return observe_bootstrap_rollout(
            client,
            registry_client,
            plan,
            result.state,
            comparison_sha256=settings["comparison_sha256"],
            registry_uri=approval["registry_uri"],
            records=settings.get("smoke_records", []),
            now=now,
        )
    report, spec, _, _ = load_candidate_evidence(
        registry_client,
        result.state.challenger_model_name,
        result.state.challenger_model_version,
        settings["comparison_sha256"],
        registry_uri=approval["registry_uri"],
    )
    if report.quality_threshold is None:
        raise ValueError("Daily live rollout requires a saved absolute quality threshold.")
    endpoints = {key: approval[key] for key in ("tracking_uri", "registry_uri")}
    resolved = resolve_model(
        result.state.challenger_model_name,
        version=result.state.challenger_model_version,
        **endpoints,
    )
    artifact = load_registered_pipeline(resolved, **endpoints)
    return observe_live_rollout(
        spark,
        plan,
        result.state,
        MonitorConfig.from_dict(settings["monitoring"]),
        artifact,
        spec,
        RolloutHealthPolicy(**settings["health"]),
        metric=report.metric,
        quality_threshold=report.quality_threshold,
        quality_gates=report.quality_gates,
        now=now,
    )


def _output(
    client: Any,
    spark: Any,
    approval: dict[str, Any],
    store: MLflowRolloutStore,
    result: RolloutResult,
) -> dict[str, Any]:
    """Automatically request guarded promotion only after durable final completion."""
    output = asdict(result)
    if result.promotion_pending:
        receipt = promote_completed_rollout(
            client, spark, approval, store=store, admission=SingleWriterAdmission()
        )
        output["promotion"] = asdict(receipt)
    return output


def run_daily_rollout(
    spark: Any, settings: dict[str, Any], *, client: Any, registry_client: Any
) -> dict[str, Any]:
    """Reconcile, observe and apply at most one daily step from an existing rollout.

    Generated jobs require exclusive remote writer permissions and serialized
    runs. They never create an endpoint, initialize a rollout, or accept a PASS
    flag from configuration. Initialization and guarded automatic promotion pins
    must already exist. A PREPARED retry only reconciles its previous request.
    """
    if not _enabled(settings):
        return {"status": "DISABLED"}
    store = MLflowRolloutStore(registry_client, settings["run_id"])
    before = store.load()
    result = reconcile_rollout(client, store=store, admission=SingleWriterAdmission())
    record = store.load()
    approval = _approval(settings, record)
    if before["status"] == "PREPARED" or result.status != "COMMITTED":
        return asdict(result)
    if result.state.phase != "ACTIVE":
        return _output(client, spark, approval, store, result)
    observation = _observe(spark, client, registry_client, settings, record, result, approval)
    persist_rollout_observation(store, observation)
    changed = advance_rollout(
        client, store=store, admission=SingleWriterAdmission(), evidence=observation.evidence
    )
    output = _output(client, spark, approval, store, changed)
    output["observation_sha256"] = observation.details_digest
    return output


def run_online_publication(
    spark: Any, settings: dict[str, Any], *, client: Any, feature_client: Any = None
) -> dict[str, Any]:
    """Submit one native sync and report its current status without polling or retries."""
    if not _enabled(settings):
        return {"status": "DISABLED"}
    spec = OnlinePublicationSpec(
        **{
            name: settings[name]
            for name in ("source_table", "online_table", "online_store", "source_table_id")
        }
    )
    receipt = publish_online_features(
        spark, spec, admission=SingleWriterAdmission(), feature_client=feature_client
    )
    return online_publication_status(client, receipt)


def _notebook_settings(dbutils: Any, section: str) -> dict[str, Any]:
    """Read the explicit deployed YAML file and reject malformed selected settings."""
    import yaml  # noqa: PLC0415 - optional integration dependency

    path = dbutils.widgets.get("serving_config_path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("serving_config_path must identify the deployed serving YAML file.")
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or section not in payload:
        raise ValueError(f"Serving configuration requires the {section} section.")
    settings = payload[section]
    _enabled(settings)
    return settings


def run_daily_rollout_notebook(spark: Any, dbutils: Any) -> dict[str, Any]:
    """Run the configured daily job and return its result for notebook presentation."""
    settings = _notebook_settings(dbutils, "rollout")
    if not settings["enabled"]:
        return {"status": "DISABLED"}
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    registry = MlflowClient(
        tracking_uri=settings.get("tracking_uri", "databricks"),
        registry_uri=settings.get("registry_uri", "databricks-uc"),
    )
    return run_daily_rollout(spark, settings, client=WorkspaceClient(), registry_client=registry)


def run_online_publication_notebook(spark: Any, dbutils: Any) -> dict[str, Any]:
    """Run one configured feature publication using the notebook's workspace identity."""
    settings = _notebook_settings(dbutils, "online_publication")
    if not settings["enabled"]:
        return {"status": "DISABLED"}
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    return run_online_publication(spark, settings, client=WorkspaceClient())
