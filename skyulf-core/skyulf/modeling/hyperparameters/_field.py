"""HyperparameterField dataclass — the building block for all model param specs."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class HyperparameterField:
    """Describe a single tunable hyperparameter."""

    name: str
    label: str
    type: str  # "number", "select", "boolean"
    default: Any
    description: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Dict[str, Any]]] = (
        None  # For 'select' type: [{"label": "L1", "value": "l1"}]
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
