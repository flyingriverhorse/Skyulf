"""Bind monitoring to the exact components of an activated or scored model set."""

from dataclasses import replace
from typing import Any

from .monitoring_config import parse_drift_thresholds
from .monitoring_registration import _enrollment_config, monitoring_destination
from .monitoring_store import enroll_monitor


def validate_set_monitoring(values: dict, settings: dict) -> None:
    """Require retained component predictions before committing a monitored batch."""
    if monitoring_destination(values) is None:
        return
    parse_drift_thresholds(values.get("monitoring_drift_thresholds", "{}"))
    if settings.get("publication", {}).get("mode") == "combined_only":
        raise ValueError("Model-set monitoring requires all or separate_views component outputs.")


def register_set_monitors(
    spark: Any,
    workflow: dict,
    values: dict,
    settings: dict,
    resolved: Any,
    artifact: Any,
    *,
    activation_started_ms: int | None = None,
) -> dict | None:
    """Enroll saved component versions; never infer them from mutable component aliases."""
    namespace = monitoring_destination(values)
    if namespace is None:
        return None
    validate_set_monitoring(values, settings)
    if resolved.name != settings["model_name"]:
        raise ValueError("Monitored model set differs from the project's model set.")
    configs = []
    for component in artifact.manifest.components:
        reference = component.reference
        base = _enrollment_config(
            {**workflow, "model_name": reference.name}, values, reference.version
        )
        configs.append(
            replace(
                base,
                prediction_table=settings["prediction_table"],
                model_set_name=resolved.name,
                model_set_version=resolved.version,
                model_set_branch=component.branch,
            )
        )
    if not configs or len({config.monitor_id for config in configs}) != len(configs):
        raise ValueError("Model-set monitoring requires distinct component model identities.")
    for config in configs:
        enroll_monitor(
            spark,
            namespace,
            config,
            preserve_activation=activation_started_ms is None,
            activation_started_ms=activation_started_ms,
        )
    return {
        "inventory_table": f"{namespace}.model_inventory",
        "configs": [config.payload() for config in configs],
    }


def run_set_monitor_enrollment_notebook(spark: Any, dbutils: Any) -> dict:
    """Enroll activated component versions from the completed, frozen parent invocation."""
    from ..mlflow.model_set import load_registered_model_set  # noqa: PLC0415
    from ..mlflow.registry import resolve_model  # noqa: PLC0415
    from .model_set_project import project_endpoints  # noqa: PLC0415
    from .monitoring_tasks import deployment_store  # noqa: PLC0415

    values = dbutils.widgets.getAll()
    if monitoring_destination(values) is None:
        return {"status": "disabled"}
    store = deployment_store(values, expected_phase="prepare")
    request = store.request
    phase = "model_decision" if request["action"] == "train" else "operator"
    output = store.receipt(phase)["output"]
    receipt = output.get("alias_change" if phase == "model_decision" else "receipt")
    if not receipt or receipt.get("kind") not in {"initial", "promotion", "rollback"}:
        return {"status": "no_activation"}
    settings = request["settings"]
    if receipt["model_name"] != settings["model_name"] or receipt["alias"] != "champion":
        raise ValueError("Enrollment requires the model set's champion transition.")
    endpoints = project_endpoints(request["config"])
    resolved = resolve_model(settings["model_name"], version=receipt["new_version"], **endpoints)
    artifact = load_registered_model_set(resolved, **endpoints)
    registration = register_set_monitors(
        spark,
        request["config"],
        values,
        settings,
        resolved,
        artifact,
        activation_started_ms=store.client.get_run(store.run_id).info.start_time,
    )
    return {"status": "enrolled", **(registration or {})}
