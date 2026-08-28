"""HyperparameterField dataclass — the building block for all model param specs."""

from dataclasses import asdict, dataclass
from typing import Any

from ...types import DEFAULT_RANDOM_STATE


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
    tunable: bool = (
        True  # False = fixed-parameter control only: shown in basic-mode
        # hyperparameters, hidden from the advanced search space. A seed is
        # never a sensible tuning target, so `random_state` fields set False.
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def random_state_field(description: str | None = None) -> HyperparameterField:
    """The shared `random_state` control used by every seeded model.

    Single definition keeps the label/range/default consistent with
    `DEFAULT_RANDOM_STATE` (finding F-21). Marked non-tunable so it shows up
    as a fixed parameter but is never offered as a search-space candidate.
    """
    return HyperparameterField(
        name="random_state",
        label="Random State",
        type="number",
        default=DEFAULT_RANDOM_STATE,
        min=0,
        max=10000,
        step=1,
        tunable=False,
        description=description
        or (
            "Seed controlling the model's randomness (sampling, shuffling, "
            "initialization). Same data + same seed = identical model."
        ),
    )
