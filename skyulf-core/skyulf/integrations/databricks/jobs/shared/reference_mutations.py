"""Bind generated alias mutations to their frozen shared monitoring destination."""

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from ...lifecycle._lifecycle_state import PhaseStore
from ...observability.monitoring import monitoring_store, reference_admission
from ...observability.monitoring.monitoring_registration import (
    monitoring_destination,
    monitoring_reference_holder,
)


def prepare_reference_mutations(spark: Any, namespace: str | None) -> None:
    """Provision a configured authority once before the lifecycle's first mutation."""
    if namespace is not None:
        monitoring_store.ensure_monitoring_store(spark, *namespace.split("."))


def reference_mutation_scope(
    spark: Any,
    namespace: str | None,
    *,
    job_run_id: int | None = None,
    task_key: str | None = None,
    workspace_host: str | None = None,
    deferred: bool = False,
) -> AbstractContextManager:
    """Leave unconfigured local callers inert; configured writers require the store guard."""
    if namespace is None:
        return nullcontext()
    metadata = (
        {"job_run_id": job_run_id, "task_key": task_key, "workspace_host": workspace_host}
        if workspace_host
        else {}
    )
    operation = (
        reference_admission.defer_reference_mutation
        if deferred
        else reference_admission.hold_reference_mutation
    )
    return operation(spark, namespace, **metadata)


def phase_reference_mutation(spark: Any, store: PhaseStore, phase: str) -> AbstractContextManager:
    """Use saved invocation settings, never editable downstream notebook parameters."""
    values = store.request.get("operator_values", {})
    namespace = (
        store.request["monitoring_namespace"]
        if "monitoring_namespace" in store.request
        else monitoring_destination(values)
    )
    host = store.request.get("workspace_host", values.get("workspace_host"))
    task = phase
    if store.request.get("graph_version") == 3 or values:
        task = {
            "evaluate_register": "register_model",
            "compare": "evaluate_model",
            "decide": "model_decision",
            "operator": "model_decision",
            "finalize": "training_report",
        }.get(phase, phase)
    return reference_mutation_scope(
        spark,
        namespace,
        job_run_id=int(store.context.job_run_id),
        task_key=task,
        workspace_host=host,
        deferred=True,
    )


def enrollment_reference_holder(store: PhaseStore, values: dict, task_key: str) -> dict:
    """Bind enrollment recovery to frozen provenance without locking expensive preparation."""
    namespace = monitoring_destination(values)
    saved = (
        store.request["monitoring_namespace"]
        if "monitoring_namespace" in store.request
        else monitoring_destination(store.request.get("operator_values", values))
    )
    if namespace != saved:
        raise ValueError("Monitoring destination differs from the frozen lifecycle invocation.")
    host = store.request.get(
        "workspace_host", store.request.get("operator_values", {}).get("workspace_host")
    )
    if not host:
        return {}
    return reference_admission.reference_mutation_holder(
        {
            "job_run_id": store.context.job_run_id,
            "task_key": task_key,
            "workspace_host": host,
        }
    )


def notebook_reference_holder(values: dict) -> dict:
    """Preserve legacy manual recovery until all generated owner widgets are available."""
    return monitoring_reference_holder(values)
