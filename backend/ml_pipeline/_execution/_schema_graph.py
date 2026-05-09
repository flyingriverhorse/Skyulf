"""Pre-execution schema prediction (C7 Phase B).

Walk a :class:`PipelineConfig` topology once and ask each Calculator's
:meth:`infer_output_schema` what its output columns/dtypes will look like.
The result is a ``Dict[node_id, Optional[SkyulfSchema]]`` — ``None`` for any
node whose schema cannot be predicted from config alone (data-dependent
encoders, unknown step types, missing upstream schema, etc.).

This module is **standalone**: it does not load data, touch the artifact
store, or import the engine. Phase B wires it into ``PipelineEngine.run``;
Phase C reuses the same function for the canvas-side ``GET .../schema-preview``
endpoint; Phase D reuses it for save-time column-reference validation.

Initial schemas (typically one per data-loader node, seeded from the dataset
catalog) are passed in by the caller. When a loader's schema is not seeded,
the loader's prediction is ``None`` and downstream predictions follow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from skyulf.preprocessing import SkyulfSchema
from skyulf.registry import NodeRegistry

from .schemas import NodeConfig, PipelineConfig

logger = logging.getLogger(__name__)


# Step types that do not transform the data frame's column set: the schema
# of the (single) downstream consumer is the schema of the upstream input.
# Training/tuning leaves don't have downstream consumers but we still record
# the input schema so the run record carries something useful.
_PASSTHROUGH_STEP_TYPES = {
    "basic_training",
    "advanced_tuning",
    "data_preview",
}

# Step types that always have an unknown predicted schema (data must be
# loaded to know the columns). The caller can seed these via
# ``initial_schemas`` to get useful downstream predictions.
_OPAQUE_STEP_TYPES = {
    "data_loader",
}


def _step_type_str(node: NodeConfig) -> str:
    """Return the node's step type as a plain string."""
    raw = node.step_type
    return raw.value if hasattr(raw, "value") else str(raw)


def _resolve_input_schema(
    node: NodeConfig,
    predicted: Dict[str, Optional[SkyulfSchema]],
) -> Optional[SkyulfSchema]:
    """Pick the upstream schema to feed into ``node``.

    For single-input nodes this is just the upstream prediction. For
    multi-input nodes (merges, splitters with several feeders) we currently
    take the first known schema — which matches what the runtime merge does
    when columns align. If any expected upstream is missing from
    ``predicted`` (out-of-order topology, malformed config), or its
    prediction is ``None``, the result is ``None``.
    """
    if not node.inputs:
        return None
    for upstream_id in node.inputs:
        if upstream_id not in predicted:
            return None
        schema = predicted[upstream_id]
        if schema is None:
            return None
        return schema
    return None


def _predict_for_node(
    node: NodeConfig,
    input_schema: Optional[SkyulfSchema],
) -> Optional[SkyulfSchema]:
    """Predict a single node's output schema. Returns ``None`` on any failure."""
    step = _step_type_str(node)

    if step in _OPAQUE_STEP_TYPES:
        return None
    if step in _PASSTHROUGH_STEP_TYPES:
        return input_schema
    if step == "feature_engineering":
        # Composite step; its sub-steps are not exposed via the registry,
        # so we cannot predict without expanding them. Treat as opaque.
        return None

    try:
        calculator_cls = NodeRegistry.get_calculator(step)
    except (ValueError, KeyError):
        return None

    if input_schema is None:
        return None

    try:
        calculator = calculator_cls()
        return calculator.infer_output_schema(input_schema, node.params)
    except Exception:  # noqa: BLE001 - best-effort prediction
        logger.debug("infer_output_schema failed for node %s", node.node_id, exc_info=True)
        return None


def predict_schemas(
    config: PipelineConfig,
    initial_schemas: Optional[Dict[str, SkyulfSchema]] = None,
) -> Dict[str, Optional[SkyulfSchema]]:
    """Walk the topology and return a per-node predicted schema map.

    Args:
        config: Pipeline configuration. Nodes are assumed topologically
            ordered (the canvas converter and partitioner both guarantee this).
        initial_schemas: Optional seed map (typically one entry per data
            loader, populated from the dataset catalog).

    Returns:
        ``{node_id: SkyulfSchema | None}`` covering every node in ``config``.
    """
    seeds = dict(initial_schemas or {})
    predicted: Dict[str, Optional[SkyulfSchema]] = {}

    for node in config.nodes:
        if node.node_id in seeds:
            predicted[node.node_id] = seeds[node.node_id]
            continue
        input_schema = _resolve_input_schema(node, predicted)
        predicted[node.node_id] = _predict_for_node(node, input_schema)

    return predicted


def schema_to_dict(schema: Optional[SkyulfSchema]) -> Optional[Dict[str, Any]]:
    """Serialize a schema for transport (JSON / API responses / DB rows)."""
    if schema is None:
        return None
    return {"columns": list(schema.columns), "dtypes": dict(schema.dtypes)}


def schemas_to_dict(
    schemas: Dict[str, Optional[SkyulfSchema]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Serialize an entire predicted-schema map."""
    return {nid: schema_to_dict(s) for nid, s in schemas.items()}


__all__: List[str] = [
    "predict_schemas",
    "schema_to_dict",
    "schemas_to_dict",
]
