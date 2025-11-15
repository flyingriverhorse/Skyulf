"""Shared hyperparameter configuration primitives."""

from typing import Any, Dict, List, Optional


class HyperparameterField:
    """Describe a single tunable hyperparameter."""

    def __init__(
        self,
        name: str,
        label: str,
        type: str,
        default: Any,
        description: str = "",
        min: Optional[float] = None,
        max: Optional[float] = None,
        step: Optional[float] = None,
        options: Optional[List[Dict[str, Any]]] = None,
        nullable: bool = False,
    ) -> None:
        self.name = name
        self.label = label
        self.type = type
        self.default = default
        self.description = description
        self.min = min
        self.max = max
        self.step = step
        self.options = options or []
        self.nullable = nullable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""

        payload: Dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "description": self.description,
        }
        if self.min is not None:
            payload["min"] = self.min
        if self.max is not None:
            payload["max"] = self.max
        if self.step is not None:
            payload["step"] = self.step
        if self.options:
            payload["options"] = self.options
        if self.nullable:
            payload["nullable"] = self.nullable
        return payload
