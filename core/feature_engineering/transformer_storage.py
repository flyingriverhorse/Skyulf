"""Transformer Storage Service.

This module provides a service for storing and retrieving fitted transformers
(scalers, encoders, etc.) used during training. These transformers are then
reused when processing validation and test sets to ensure proper ML practices.

Storage is in-memory for now, but can be extended to Redis or database later.
"""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _sanitize_for_storage(value: Any) -> Any:
    """Convert numpy scalars/arrays into JSON-serializable Python types."""

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return [_sanitize_for_storage(item) for item in value.tolist()]

    if isinstance(value, dict):
        return {str(key): _sanitize_for_storage(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_storage(item) for item in value]

    return value


DEPRECATION_MESSAGE = (
    "TransformerStorage is deprecated. Use SklearnPipelineStore from "
    "core.feature_engineering.sklearn_pipeline_store instead."
)


_DEPRECATION_EMITTED = False


class TransformerStorage:
    """In-memory storage for fitted transformers.

    Transformers are stored with keys: pipeline_id:node_id:column_name
    This allows each node to store multiple transformers (one per column).
    """

    def __init__(self):
        global _DEPRECATION_EMITTED
        if not _DEPRECATION_EMITTED:
            warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
            logger.warning(
                "TransformerStorage is deprecated and will be removed in a future release. "
                "Switch to core.feature_engineering.sklearn_pipeline_store.",
            )
            _DEPRECATION_EMITTED = True

        self._storage: Dict[str, Dict[str, Any]] = {}
        # Store metadata about when transformers were created
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def store_transformer(
        self,
        pipeline_id: str,
        node_id: str,
        transformer_name: str,
        transformer: Any,
        column_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a fitted transformer.

        Args:
            pipeline_id: Pipeline identifier
            node_id: Node identifier
            transformer_name: Type of transformer (e.g., 'scaler', 'encoder')
            transformer: The fitted transformer object
            column_name: Optional column name if transformer is column-specific
            metadata: Optional metadata about the transformer
        """
        key = self._build_key(pipeline_id, node_id, transformer_name, column_name)

        try:
            # Store the transformer
            self._storage[key] = {
                'transformer': transformer,
                'created_at': datetime.now(),
                'pipeline_id': pipeline_id,
                'node_id': node_id,
                'transformer_name': transformer_name,
                'column_name': column_name,
                'split_activity': {},
                'updated_at': datetime.now(),
            }

            # Store metadata if provided
            if metadata:
                self._metadata[key] = _sanitize_for_storage(metadata)

            logger.debug(f"Stored transformer: {key}")

        except Exception as e:
            logger.error(f"Failed to store transformer {key}: {e}")
            raise

    def record_split_activity(
        self,
        pipeline_id: str,
        node_id: str,
        transformer_name: str,
        *,
        split_name: str,
        action: str,
        column_name: Optional[str] = None,
        row_count: Optional[int] = None,
    ) -> None:
        """Track how a transformer interacted with a specific split.

        Args:
            pipeline_id: Pipeline identifier
            node_id: Node identifier that owns the transformer
            transformer_name: Logical transformer name (e.g. 'one_hot_encoder')
            split_name: Name of the split (train/test/validation)
            action: Action performed ('fit_transform', 'transform', etc.)
            column_name: Optional column the transformer is associated with
            row_count: Optional number of rows processed
        """

        key = self._build_key(pipeline_id, node_id, transformer_name, column_name)
        entry = self._storage.get(key)
        if not entry:
            logger.debug(
                "Split activity ignored – transformer not registered",
                extra={
                    "pipeline_id": pipeline_id,
                    "node_id": node_id,
                    "transformer_name": transformer_name,
                    "column_name": column_name,
                },
            )
            return

        split_key = str(split_name or "unknown").strip() or "unknown"
        now = datetime.now()
        activity_map: Dict[str, Dict[str, Any]] = entry.setdefault('split_activity', {})

        payload: Dict[str, Any] = {
            'action': str(action or 'not_applied'),
            'row_count': int(row_count) if row_count is not None else None,
            'updated_at': now,
        }
        activity_map[split_key] = _sanitize_for_storage(payload)
        entry['updated_at'] = now

    def list_transformers(
        self,
        *,
        pipeline_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return stored transformer metadata optionally filtered by pipeline/node."""

        results: List[Dict[str, Any]] = []
        for key, entry in self._storage.items():
            if pipeline_id and entry.get('pipeline_id') != pipeline_id:
                continue
            if node_id and entry.get('node_id') != node_id:
                continue

            raw_metadata = self._metadata.get(key)
            if raw_metadata:
                sanitized_metadata = _sanitize_for_storage(raw_metadata)
                self._metadata[key] = sanitized_metadata  # persist sanitized copy
                metadata = sanitized_metadata
            else:
                metadata = {}

            raw_split_activity = entry.get('split_activity') or {}
            split_activity = _sanitize_for_storage(raw_split_activity)
            entry['split_activity'] = split_activity  # persist sanitized copy

            results.append(
                {
                    'storage_key': key,
                    'pipeline_id': entry.get('pipeline_id'),
                    'node_id': entry.get('node_id'),
                    'transformer_name': entry.get('transformer_name'),
                    'column_name': entry.get('column_name'),
                    'created_at': entry.get('created_at'),
                    'updated_at': entry.get('updated_at', entry.get('created_at')),
                    'metadata': deepcopy(metadata) if metadata else {},
                    'split_activity': deepcopy(split_activity),
                }
            )

        return results

    def get_transformer(
        self,
        pipeline_id: str,
        node_id: str,
        transformer_name: str,
        column_name: Optional[str] = None
    ) -> Optional[Any]:
        """Retrieve a fitted transformer.

        Args:
            pipeline_id: Pipeline identifier
            node_id: Node identifier
            transformer_name: Type of transformer
            column_name: Optional column name

        Returns:
            The fitted transformer or None if not found
        """
        key = self._build_key(pipeline_id, node_id, transformer_name, column_name)

        entry = self._storage.get(key)
        if entry:
            logger.debug(f"Retrieved transformer: {key}")
            return entry['transformer']

        logger.debug(f"Transformer not found: {key}")
        return None

    def has_transformer(
        self,
        pipeline_id: str,
        node_id: str,
        transformer_name: str,
        column_name: Optional[str] = None
    ) -> bool:
        """Check if a transformer exists.

        Args:
            pipeline_id: Pipeline identifier
            node_id: Node identifier
            transformer_name: Type of transformer
            column_name: Optional column name

        Returns:
            True if transformer exists, False otherwise
        """
        key = self._build_key(pipeline_id, node_id, transformer_name, column_name)
        return key in self._storage

    def get_metadata(
        self,
        pipeline_id: str,
        node_id: str,
        transformer_name: str,
        column_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a transformer.

        Args:
            pipeline_id: Pipeline identifier
            node_id: Node identifier
            transformer_name: Type of transformer
            column_name: Optional column name

        Returns:
            Metadata dictionary or None if not found
        """
        key = self._build_key(pipeline_id, node_id, transformer_name, column_name)
        return self._metadata.get(key)

    def clear_pipeline(self, pipeline_id: str) -> None:
        """Clear all transformers for a specific pipeline.

        Args:
            pipeline_id: Pipeline identifier
        """
        keys_to_remove = [
            key for key, entry in self._storage.items()
            if entry['pipeline_id'] == pipeline_id
        ]

        for key in keys_to_remove:
            del self._storage[key]
            self._metadata.pop(key, None)

        logger.info(f"Cleared {len(keys_to_remove)} transformers for pipeline {pipeline_id}")

    def clear_node(self, pipeline_id: str, node_id: str) -> None:
        """Clear all transformers for a specific node.

        Args:
            pipeline_id: Pipeline identifier
            node_id: Node identifier
        """
        keys_to_remove = [
            key for key, entry in self._storage.items()
            if entry['pipeline_id'] == pipeline_id and entry['node_id'] == node_id
        ]

        for key in keys_to_remove:
            del self._storage[key]
            self._metadata.pop(key, None)

        logger.info(f"Cleared {len(keys_to_remove)} transformers for node {node_id}")

    def clear_old_transformers(self, max_age_hours: int = 24) -> None:
        """Clear transformers older than specified age.

        Args:
            max_age_hours: Maximum age in hours
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        keys_to_remove = [
            key for key, entry in self._storage.items()
            if entry['created_at'] < cutoff_time
        ]

        for key in keys_to_remove:
            del self._storage[key]
            self._metadata.pop(key, None)

        if keys_to_remove:
            logger.info(f"Cleared {len(keys_to_remove)} old transformers")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored transformers.

        Returns:
            Dictionary with storage statistics
        """
        pipelines = set(entry['pipeline_id'] for entry in self._storage.values())
        nodes = set(entry['node_id'] for entry in self._storage.values())

        return {
            'total_transformers': len(self._storage),
            'unique_pipelines': len(pipelines),
            'unique_nodes': len(nodes),
            'storage_keys': list(self._storage.keys())
        }

    def _build_key(
        self,
        pipeline_id: str,
        node_id: str,
        transformer_name: str,
        column_name: Optional[str] = None
    ) -> str:
        """Build a unique key for storing a transformer.

        Args:
            pipeline_id: Pipeline identifier
            node_id: Node identifier
            transformer_name: Type of transformer
            column_name: Optional column name

        Returns:
            Unique storage key
        """
        if column_name:
            return f"{pipeline_id}:{node_id}:{transformer_name}:{column_name}"
        return f"{pipeline_id}:{node_id}:{transformer_name}"


# Global singleton instance (deprecated)
_transformer_storage: Optional[TransformerStorage] = None


def get_transformer_storage() -> TransformerStorage:
    """Get the global transformer storage instance (deprecated)."""

    global _transformer_storage
    if _transformer_storage is None:
        _transformer_storage = TransformerStorage()
    return _transformer_storage
