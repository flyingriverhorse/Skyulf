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
    depends_on: dict[str, Any] | None = (
        None  # Only relevant/shown when another param equals a given value,
        # e.g. {"param": "penalty", "value": "elasticnet"} for `l1_ratio`.
    )
    exclusive_options: list[Any] | None = (
        None  # For multi-select search-space tuning: values here can't be
        # combined with any other option in the same search space (e.g.
        # `penalty="elasticnet"` mixed with "l1"/"l2" produces invalid
        # per-trial combos elsewhere, so selecting one deselects the rest).
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
