"""Static column-reference validation (C7 Phase D).

Walks every node's ``params`` for known column-reference fields
(``columns``, ``column``, ``target``, ``target_column``, ``column_types``
keys) and checks that each referenced name actually exists in the
upstream node's predicted schema.

Phase B's :func:`predict_schemas` does the schema graph; this module
builds on it to surface broken column references (typo, deleted
upstream column, renamed feature, etc.) without running the pipeline.

When the upstream schema is ``None`` (data loader without a catalog
seed, data-dependent encoder upstream, etc.) the node is skipped — we
simply don't know enough to validate. False positives on save would be
worse than missed detections.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from skyulf.preprocessing import SkyulfSchema

from .schemas import NodeConfig, PipelineConfig

# Param keys whose values are column names (str or list of str).
_STRING_REF_KEYS = ("target", "target_column", "column", "label_column")
_LIST_REF_KEYS = (
    "columns",
    "feature_columns",
    "numeric_columns",
    "categorical_columns",
)
# Param keys whose values are dicts keyed by column name.
_DICT_KEY_REF_KEYS = ("column_types", "column_mapping", "rename_map")

# Some step types have column-ref params that are genuinely optional:
# the pipeline runs fine even when the column is absent upstream.
# For these step types we skip validation of those specific param keys
# to avoid false-positive amber indicators on the canvas.
#
# TrainTestSplitter.target_column — only for stratification; optional.
# When a FeatureTargetSplitter is upstream the target has already been
# dropped from the predicted schema so the column will never be found.
#
# basic_training / advanced_tuning: target_column is metadata that tells
# the training strategy which column to use as y. The input to these nodes
# is a SplitDataset where X already has the target separated out, so the
# column is intentionally absent from the predicted feature schema.
_OPTIONAL_PARAM_KEYS: Dict[str, Set[str]] = {
    # TrainTestSplitter: target_column is only for stratification; optional.
    # When FeatureTargetSplitter is upstream the target is already dropped.
    "TrainTestSplitter": {"target_column"},
    "Split": {"target_column"},
    # Training nodes: target_column is metadata (which column to use as y).
    # The X features DataFrame passed to them never contains it.
    "basic_training": {"target_column"},
    "advanced_tuning": {"target_column"},
    # Encoders: target_column is the y series input for supervised encoders
    # (TargetEncoder). X features no longer contain it when
    # FeatureTargetSplitter is upstream.
    "OneHotEncoder": {"target_column"},
    "DummyEncoder": {"target_column"},
    # LabelEncoder / OrdinalEncoder are "target-safe" — they replace values
    # in-place and are explicitly designed to encode the target column (y).
    # When placed after FeatureTargetSplitter the target is absent from the
    # predicted feature schema, but a user may still select it in the column
    # picker (valid use-case). Skip `columns` validation to avoid false-positive
    # amber for these two encoders.
    "OrdinalEncoder": {"target_column", "columns"},
    "LabelEncoder": {"target_column", "columns"},
    "TargetEncoder": {"target_column"},
    "HashEncoder": {"target_column"},
}


def _iter_string_refs(params: Dict[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield references from single-string param fields (e.g. `target`)."""
    for key in _STRING_REF_KEYS:
        val = params.get(key)
        if isinstance(val, str) and val:
            yield key, val


def _iter_list_refs(params: Dict[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield references from list-of-string param fields (e.g. `columns`)."""
    for key in _LIST_REF_KEYS:
        val = params.get(key)
        if not isinstance(val, list):
            continue
        for item in val:
            if isinstance(item, str) and item:
                yield key, item


def _iter_dict_key_refs(params: Dict[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield references from dict-keyed param fields (e.g. `column_types`)."""
    for key in _DICT_KEY_REF_KEYS:
        val = params.get(key)
        if not isinstance(val, dict):
            continue
        for col in val.keys():
            if isinstance(col, str) and col:
                yield key, col


def _iter_referenced_columns(params: Dict[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield ``(field_name, column_name)`` for every column reference in params."""
    yield from _iter_string_refs(params)
    yield from _iter_list_refs(params)
    yield from _iter_dict_key_refs(params)


def _upstream_schema(
    node: NodeConfig,
    predicted: Dict[str, Optional[SkyulfSchema]],
) -> Optional[SkyulfSchema]:
    """Pick the upstream predicted schema the node actually consumes."""
    if not node.inputs:
        return None
    upstream_id = node.inputs[0]
    return predicted.get(upstream_id)


def find_broken_references(
    config: PipelineConfig,
    predicted_schemas: Dict[str, Optional[SkyulfSchema]],
) -> List[Dict[str, Any]]:
    """Return a list of broken column references.

    Each entry: ``{"node_id", "field", "column", "upstream_node_id"}``.

    Args:
        config: Pipeline configuration.
        predicted_schemas: Output of :func:`predict_schemas`.

    Returns:
        List of broken-reference dicts (empty when nothing to flag).
        Nodes whose upstream schema is unknown (``None``) are skipped.
    """
    broken: List[Dict[str, Any]] = []

    for node in config.nodes:
        upstream = _upstream_schema(node, predicted_schemas)
        if upstream is None:
            continue
        upstream_id = node.inputs[0] if node.inputs else None
        step = node.step_type.value if hasattr(node.step_type, "value") else str(node.step_type)
        optional_keys: Set[str] = _OPTIONAL_PARAM_KEYS.get(step, set())
        for field_name, col in _iter_referenced_columns(node.params):
            if field_name in optional_keys:
                continue
            if col not in upstream:
                broken.append(
                    {
                        "node_id": node.node_id,
                        "field": field_name,
                        "column": col,
                        "upstream_node_id": upstream_id,
                    }
                )

    return broken


__all__ = ["find_broken_references"]
