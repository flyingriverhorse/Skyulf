"""HyperparameterField dataclass — the building block for all model param specs."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class HyperparameterField:
    """Describe a single tunable hyperparameter."""

    name: str
    label: str
    type: str  # "number", "select", "boolean"
    default: Any
    description: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[dict[str, Any]] | None = (
        None  # For 'select' type: [{"label": "L1", "value": "l1"}]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
