"""Canonical pipeline store singleton module.

Provide a single, clear import path for the shared SklearnPipelineStore used
across the feature-engineering codebase. Other modules should import
``get_pipeline_store`` from here to avoid import variations.
"""
from __future__ import annotations

from core.feature_engineering.sklearn_pipeline_store import SklearnPipelineStore

# Centralized, process-scoped singleton owned by this module. By creating
# the store here we provide a single authoritative place for lifecycle and
# potential future persistence/backing-store swaps.
_SINGLETON: SklearnPipelineStore = SklearnPipelineStore()


def get_pipeline_store() -> SklearnPipelineStore:
    """Return the canonical pipeline store singleton for the current process.

    Other modules should import ``get_pipeline_store`` from this module to
    guarantee they receive the same in-memory store instance.
    """
    return _SINGLETON


__all__ = ["get_pipeline_store"]
