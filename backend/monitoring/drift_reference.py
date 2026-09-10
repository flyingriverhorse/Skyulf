"""Select a saved raw-source snapshot for comparisons with uploaded raw data."""

from typing import Any

from backend.exceptions.core import InvalidRequestException
from backend.ml_pipeline._execution.graph_utils import _extract_columns
from backend.ml_pipeline.artifacts.store import ArtifactStore
from backend.ml_pipeline.constants import StepType


def _job_ancestors(job: Any) -> list[dict[str, Any]] | None:
    """Read only the selected model's upstream execution graph, if metadata exists."""
    graph = getattr(job, "graph", None)
    if not isinstance(graph, dict) or not graph.get("nodes"):
        return None

    nodes = {node["node_id"]: node for node in graph["nodes"]}
    pending = [getattr(job, "node_id", None)]
    visited: set[str] = set()
    ancestors = []
    while pending:
        node_id = pending.pop()
        if not isinstance(node_id, str) or node_id not in nodes:
            raise InvalidRequestException(
                "The saved pipeline cannot identify the source data for this model."
            )
        if node_id in visited:
            continue
        visited.add(node_id)
        node = nodes[node_id]
        ancestors.append(node)
        pending.extend(node.get("inputs", []))
    return ancestors


def resolve_drift_reference(
    artifact_store: ArtifactStore, job: Any, legacy_key: str
) -> tuple[str, set[str]]:
    """Select a unique raw loader snapshot and explicit drop-column exclusions.

    The reference contains the rows actually loaded, before preprocessing or
    splitting; sampled loaders remain sampled. It can include validation/test
    rows. Existing jobs can use this snapshot without retraining or rewriting
    their transformed reference.
    Graphless legacy jobs retain their original reference contract. Source columns
    are not inferred from model feature names, which may be encoded or derived.

    Explicit drop-column configs are monitoring exclusions, matching the model
    bundle's convention; they are not a complete model-input dependency analysis.
    """
    ancestors = _job_ancestors(job)
    if ancestors is None:
        return legacy_key, set()

    loaders = [
        node["node_id"]
        for node in ancestors
        if node.get("step_type") in {StepType.DATA_LOADER, "DataLoader"}
    ]
    if len(loaders) != 1:
        raise InvalidRequestException(
            "Drift analysis requires one saved source dataset for the selected model; "
            "the pipeline has no unique source."
        )
    source_key = loaders[0]
    if not artifact_store.exists(source_key):
        raise InvalidRequestException(
            "The original source snapshot is unavailable. Retrain this model to save "
            "a raw reference before checking an uploaded dataset."
        )

    excluded = {
        column
        for node in ancestors
        for column in _extract_columns(node.get("step_type", ""), node.get("params") or {})
    }
    return source_key, excluded
