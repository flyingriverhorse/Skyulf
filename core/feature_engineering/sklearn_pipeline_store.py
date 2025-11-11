"""Pipeline-oriented storage service for scikit-learn transformers.

This module keeps fitted pipeline objects keyed by pipeline/node identifiers.
It replaces the bespoke transformer registry with a solution that leans on
scikit-learn's Pipeline primitives and still exposes audit-friendly metadata.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone

logger = logging.getLogger(__name__)


def _sanitize_for_storage(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_sanitize_for_storage(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _sanitize_for_storage(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_storage(item) for item in value]
    return value


def _infer_row_count(data: Any) -> Optional[int]:
    try:
        return int(len(data))
    except (TypeError, ValueError):
        return None


class SklearnPipelineStore:
    """Registry for fitted Pipeline-compatible estimators."""

    def __init__(self) -> None:
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register_pipeline(
        self,
        pipeline_id: str,
        node_id: str,
        pipeline_name: str,
        estimator: Any,
        *,
        column_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        clone_pipeline: bool = True,
    ) -> Any:
        key = self._build_key(pipeline_id, node_id, pipeline_name, column_name)
        should_clone = clone_pipeline and isinstance(estimator, BaseEstimator)
        stored_estimator = clone(estimator) if should_clone else estimator
        now = datetime.now()
        self._storage[key] = {
            "estimator": stored_estimator,
            "created_at": now,
            "updated_at": now,
            "pipeline_id": pipeline_id,
            "node_id": node_id,
            "transformer_name": pipeline_name,
            "column_name": column_name,
            "split_activity": {},
        }
        if metadata:
            self._metadata[key] = _sanitize_for_storage(metadata)
        logger.debug("Registered pipeline %s", key)
        return stored_estimator

    def get_pipeline(
        self,
        pipeline_id: str,
        node_id: str,
        pipeline_name: str,
        column_name: Optional[str] = None,
    ) -> Optional[Any]:
        key = self._build_key(pipeline_id, node_id, pipeline_name, column_name)
        entry = self._storage.get(key)
        return entry["estimator"] if entry else None

    def has_pipeline(
        self,
        pipeline_id: str,
        node_id: str,
        pipeline_name: str,
        column_name: Optional[str] = None,
    ) -> bool:
        key = self._build_key(pipeline_id, node_id, pipeline_name, column_name)
        return key in self._storage

    def get_metadata(
        self,
        pipeline_id: str,
        node_id: str,
        pipeline_name: Optional[str] = None,
        *,
        column_name: Optional[str] = None,
        transformer_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        resolved_name = self._resolve_pipeline_name(pipeline_name, transformer_name)
        key = self._build_key(pipeline_id, node_id, resolved_name, column_name)
        raw = self._metadata.get(key)
        return _sanitize_for_storage(raw) if raw else None

    # ------------------------------------------------------------------
    # Compatibility helpers for legacy transformer storage interface
    # ------------------------------------------------------------------
    def store_transformer(
        self,
        pipeline_id: str,
        node_id: str,
        transformer_name: str,
        transformer: Any,
        *,
        column_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Backward-compatible wrapper that stores arbitrary transformers."""

        return self.register_pipeline(
            pipeline_id=pipeline_id,
            node_id=node_id,
            pipeline_name=transformer_name,
            estimator=transformer,
            column_name=column_name,
            metadata=metadata,
            clone_pipeline=False,
        )

    def get_transformer(
        self,
        pipeline_id: str,
        node_id: str,
        transformer_name: str,
        column_name: Optional[str] = None,
    ) -> Optional[Any]:
        return self.get_pipeline(pipeline_id, node_id, transformer_name, column_name)

    def has_transformer(
        self,
        pipeline_id: str,
        node_id: str,
        transformer_name: str,
        column_name: Optional[str] = None,
    ) -> bool:
        return self.has_pipeline(pipeline_id, node_id, transformer_name, column_name)

    def list_transformers(
        self,
        *,
        pipeline_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.list_pipelines(pipeline_id=pipeline_id, node_id=node_id)

    def fit(
        self,
        pipeline_id: str,
        node_id: str,
        pipeline_name: str,
        X: Any,
        y: Optional[Any] = None,
        *,
        column_name: Optional[str] = None,
        split_name: str = "train",
    ) -> BaseEstimator:
        estimator = self._require_estimator(pipeline_id, node_id, pipeline_name, column_name)
        estimator.fit(X, y)
        self.record_split_activity(
            pipeline_id,
            node_id,
            pipeline_name,
            split_name=split_name,
            action="fit",
            column_name=column_name,
            row_count=_infer_row_count(X),
        )
        return estimator

    def transform(
        self,
        pipeline_id: str,
        node_id: str,
        pipeline_name: str,
        X: Any,
        *,
        column_name: Optional[str] = None,
        split_name: str = "validation",
    ) -> Any:
        estimator = self._require_estimator(pipeline_id, node_id, pipeline_name, column_name)
        if not hasattr(estimator, "transform"):
            raise AttributeError("Stored estimator does not implement transform")
        result = estimator.transform(X)
        self.record_split_activity(
            pipeline_id,
            node_id,
            pipeline_name,
            split_name=split_name,
            action="transform",
            column_name=column_name,
            row_count=_infer_row_count(X),
        )
        return result

    def fit_transform(
        self,
        pipeline_id: str,
        node_id: str,
        pipeline_name: str,
        X: Any,
        y: Optional[Any] = None,
        *,
        column_name: Optional[str] = None,
        split_name: str = "train",
    ) -> Any:
        estimator = self._require_estimator(pipeline_id, node_id, pipeline_name, column_name)
        if hasattr(estimator, "fit_transform"):
            result = estimator.fit_transform(X, y)
        else:
            estimator.fit(X, y)
            result = estimator.transform(X)
        self.record_split_activity(
            pipeline_id,
            node_id,
            pipeline_name,
            split_name=split_name,
            action="fit_transform",
            column_name=column_name,
            row_count=_infer_row_count(X),
        )
        return result

    def record_split_activity(
        self,
        pipeline_id: str,
        node_id: str,
        pipeline_name: Optional[str] = None,
        *,
        split_name: str,
        action: str,
        column_name: Optional[str] = None,
        row_count: Optional[int] = None,
        transformer_name: Optional[str] = None,
    ) -> None:
        resolved_name = self._resolve_pipeline_name(pipeline_name, transformer_name)
        key = self._build_key(pipeline_id, node_id, resolved_name, column_name)
        entry = self._storage.get(key)
        if not entry:
            logger.debug("Pipeline activity ignored – estimator not registered", extra={"key": key})
            return
        split_key = str(split_name or "unknown").strip() or "unknown"
        payload = {
            "action": str(action or "not_applied"),
            "row_count": int(row_count) if row_count is not None else None,
            "updated_at": datetime.now(),
        }
        sanitized = _sanitize_for_storage(payload)
        entry.setdefault("split_activity", {})[split_key] = sanitized
        entry["updated_at"] = sanitized["updated_at"]

    def list_pipelines(
        self,
        *,
        pipeline_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for key, entry in self._storage.items():
            if pipeline_id and entry.get("pipeline_id") != pipeline_id:
                continue
            if node_id and entry.get("node_id") != node_id:
                continue
            metadata = self._metadata.get(key) or {}
            sanitized_metadata = _sanitize_for_storage(metadata) if metadata else {}
            split_activity = _sanitize_for_storage(entry.get("split_activity") or {})
            results.append(
                {
                    "storage_key": key,
                    "pipeline_id": entry.get("pipeline_id"),
                    "node_id": entry.get("node_id"),
                    "transformer_name": entry.get("transformer_name"),
                    "column_name": entry.get("column_name"),
                    "created_at": entry.get("created_at"),
                    "updated_at": entry.get("updated_at", entry.get("created_at")),
                    "metadata": sanitized_metadata,
                    "split_activity": split_activity,
                }
            )
        return results

    def clear_pipeline(self, pipeline_id: str) -> None:
        keys_to_remove = [
            key for key, entry in self._storage.items() if entry.get("pipeline_id") == pipeline_id
        ]
        for key in keys_to_remove:
            del self._storage[key]
            self._metadata.pop(key, None)
        if keys_to_remove:
            logger.info("Cleared %s pipeline estimators for %s", len(keys_to_remove), pipeline_id)

    def clear_node(self, pipeline_id: str, node_id: str) -> None:
        keys_to_remove = [
            key
            for key, entry in self._storage.items()
            if entry.get("pipeline_id") == pipeline_id and entry.get("node_id") == node_id
        ]
        for key in keys_to_remove:
            del self._storage[key]
            self._metadata.pop(key, None)
        if keys_to_remove:
            logger.info(
                "Cleared %s pipeline estimators for node %s", len(keys_to_remove), node_id
            )

    def clear_old_pipelines(self, max_age_hours: int = 24) -> None:
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        keys_to_remove: List[str] = []
        for key, entry in self._storage.items():
            created_at = entry.get("created_at")
            if not isinstance(created_at, datetime):
                keys_to_remove.append(key)
                continue
            if created_at < cutoff:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._storage[key]
            self._metadata.pop(key, None)
        if keys_to_remove:
            logger.info("Cleared %s old pipeline estimators", len(keys_to_remove))

    def clear_all(self) -> None:
        self._storage.clear()
        self._metadata.clear()

    def get_stats(self) -> Dict[str, Any]:
        pipelines = {entry.get("pipeline_id") for entry in self._storage.values()}
        nodes = {entry.get("node_id") for entry in self._storage.values()}
        return {
            "total_transformers": len(self._storage),
            "unique_pipelines": len({pid for pid in pipelines if pid is not None}),
            "unique_nodes": len({nid for nid in nodes if nid is not None}),
            "storage_keys": list(self._storage.keys()),
        }

    def _require_estimator(
        self,
        pipeline_id: str,
        node_id: str,
        pipeline_name: Optional[str],
        column_name: Optional[str],
    ) -> BaseEstimator:
        resolved_name = self._resolve_pipeline_name(pipeline_name, None)
        estimator = self.get_pipeline(pipeline_id, node_id, resolved_name, column_name)
        if estimator is None:
            raise KeyError(
                f"Estimator not registered for {pipeline_id}:{node_id}:{resolved_name}:{column_name}"
            )
        return estimator

    @staticmethod
    def _resolve_pipeline_name(
        pipeline_name: Optional[str], transformer_name: Optional[str]
    ) -> str:
        candidate = pipeline_name or transformer_name
        if not candidate:
            raise ValueError("pipeline_name or transformer_name must be provided")
        return str(candidate)

    @staticmethod
    def _build_key(
        pipeline_id: str,
        node_id: str,
        pipeline_name: str,
        column_name: Optional[str] = None,
    ) -> str:
        if column_name:
            return f"{pipeline_id}:{node_id}:{pipeline_name}:{column_name}"
        return f"{pipeline_id}:{node_id}:{pipeline_name}"


def get_pipeline_store() -> SklearnPipelineStore:
    """Delegate to the canonical singleton owner in
    ``core.feature_engineering.pipeline_store_singleton``.

    This keeps the class and instance creation separate: tests or code that
    need the concrete ``SklearnPipelineStore`` class can still import it
    from this module, while runtime callers that need the shared instance
    should call this function which will return the single object owned
    by the canonical module.
    """
    # Local import to avoid circular import at module import time.
    from core.feature_engineering.pipeline_store_singleton import get_pipeline_store as _canonical_get

    return _canonical_get()


__all__ = [
    "SklearnPipelineStore",
    "get_pipeline_store",
]
