import logging
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class NodeRegistry:
    _calculators: dict[str, type] = {}
    _appliers: dict[str, type] = {}
    _metadata: dict[str, dict[str, Any]] = {}
    _lock = Lock()

    @classmethod
    def register(cls, name: str, applier_cls: type, metadata: dict[str, Any] | None = None):
        """
        Decorator to register a Calculator/Applier pair.

        Args:
            name: The unique string identifier for the node (e.g. 'random_forest').
            applier_cls: The class of the Applier (must be passed as we decorate the Calculator).
            metadata: Optional dictionary of UI metadata.
        """

        def wrapper(calculator_cls):
            with cls._lock:
                if name in cls._calculators:
                    logger.warning(
                        f"Node '{name}' is being re-registered. Overwriting previous registration."
                    )

                cls._calculators[name] = calculator_cls
                cls._appliers[name] = applier_cls

                # 1. Use passed metadata if available
                if metadata is not None:
                    cls._metadata[name] = metadata
                # 2. Otherwise check for __node_meta__ (from @node_meta decorator)
                elif hasattr(calculator_cls, "__node_meta__"):
                    meta = calculator_cls.__node_meta__
                    cls._metadata[name] = {
                        "id": meta.id,
                        "name": meta.name,
                        "category": meta.category,
                        "description": meta.description,
                        "params": meta.params,
                        "tags": meta.tags,
                    }

            return calculator_cls

        return wrapper

    @classmethod
    def get_calculator(cls, name: str) -> type:
        if name not in cls._calculators:
            raise ValueError(
                f"Node '{name}' not found in registry. Available nodes: {list(cls._calculators.keys())}"
            )
        return cls._calculators[name]

    @classmethod
    def get_applier(cls, name: str) -> type:
        if name not in cls._appliers:
            raise ValueError(f"Node '{name}' not found in registry.")
        return cls._appliers[name]

    @classmethod
    def get_all_metadata(cls) -> dict[str, dict[str, Any]]:
        """Return a snapshot of all registered node metadata, keyed by node ID."""
        return dict(cls._metadata)
