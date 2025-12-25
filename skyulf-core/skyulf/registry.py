from typing import Any, Dict, Optional, Type
import logging

logger = logging.getLogger(__name__)


class NodeRegistry:
    _calculators: Dict[str, Type] = {}
    _appliers: Dict[str, Type] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls, name: str, applier_cls: Type, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator to register a Calculator/Applier pair.

        Args:
            name: The unique string identifier for the node (e.g. 'random_forest').
            applier_cls: The class of the Applier (must be passed as we decorate the Calculator).
            metadata: Optional dictionary of UI metadata.
        """

        def wrapper(calculator_cls):
            if name in cls._calculators:
                logger.warning(
                    f"Node '{name}' is being re-registered. Overwriting previous registration."
                )

            cls._calculators[name] = calculator_cls
            cls._appliers[name] = applier_cls

            if metadata:
                cls._metadata[name] = metadata

            return calculator_cls

        return wrapper

    @classmethod
    def get_calculator(cls, name: str) -> Type:
        if name not in cls._calculators:
            raise ValueError(
                f"Node '{name}' not found in registry. Available nodes: {list(cls._calculators.keys())}"
            )
        return cls._calculators[name]

    @classmethod
    def get_applier(cls, name: str) -> Type:
        if name not in cls._appliers:
            raise ValueError(f"Node '{name}' not found in registry.")
        return cls._appliers[name]

    @classmethod
    def get_all_metadata(cls) -> Dict[str, Dict[str, Any]]:
        return cls._metadata
