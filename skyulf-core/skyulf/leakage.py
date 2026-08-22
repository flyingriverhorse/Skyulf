"""Static diagnostics for preprocessing configurations that risk data leakage.

The data-dependent node list is derived from the registry (every node declares
``learns_from_data`` on its ``@node_meta``), so it cannot drift from the node
implementations. Unknown transformers fail closed: they are treated as
data-dependent until proven otherwise.
"""

from typing import Any, Literal

from .registry import NodeRegistry
from .types import PipelineConfig

OnLeakage = Literal["raise", "warn", "ignore"]
_ON_LEAKAGE_MODES = frozenset({"raise", "warn", "ignore"})

NO_SPLIT_DIAGNOSTIC = (
    "No train/test split is defined in this pipeline, so the leakage guarantee "
    "does not apply: every fit sees the whole dataset. Add a TrainTestSplitter "
    "(or rely on cross-validation) to restore the guarantee."
)


def data_dependent_transformers() -> frozenset[str]:
    """Node IDs whose fit() learns statistics from the data it is given."""
    return frozenset(
        node_id
        for node_id, meta in NodeRegistry.get_all_metadata().items()
        if meta.get("learns_from_data")
    )


def train_test_splitters() -> frozenset[str]:
    """Node IDs that create a train/test boundary (registry-derived)."""
    return frozenset(
        node_id
        for node_id, meta in NodeRegistry.get_all_metadata().items()
        if meta.get("is_splitter")
    )


def validate_leakage_safety(
    pipeline_config: PipelineConfig | dict[str, Any],
    on_leakage: OnLeakage = "raise",
) -> list[str]:
    """Diagnose learned preprocessing configured before a train/test split.

    ``on_leakage`` controls definite violations (a learned fit before the
    split): ``"raise"`` (default) raises ``ValueError``, ``"warn"`` returns
    the warnings, ``"ignore"`` returns an empty list. The no-split verdict is
    advisory: it is reported under ``"raise"``/``"warn"`` but never raises.
    """
    if on_leakage not in _ON_LEAKAGE_MODES:
        raise ValueError(
            f"on_leakage must be one of {sorted(_ON_LEAKAGE_MODES)}, got {on_leakage!r}"
        )

    preprocessing = pipeline_config.get("preprocessing", [])
    splitters = train_test_splitters()
    splitter = next(
        (
            (index, step.get("transformer"))
            for index, step in enumerate(preprocessing)
            if step.get("transformer") in splitters
        ),
        None,
    )
    if splitter is None:
        return [] if on_leakage == "ignore" else [NO_SPLIT_DIAGNOSTIC]

    splitter_index, splitter_name = splitter
    learners = data_dependent_transformers()
    violations = []
    for index, step in enumerate(preprocessing[:splitter_index]):
        transformer = step.get("transformer")
        if transformer in splitters:
            continue
        if transformer in learners:
            reason = "fits its statistics on the full dataset including the test set"
        else:
            # Fail closed: an unregistered transformer cannot be proven safe.
            reason = "is not a known node and is treated as data-dependent until proven otherwise"
        violations.append(
            f"Step {index} ('{transformer}') is configured before the train/test split "
            f"(step {splitter_index}, '{splitter_name}') and {reason} — move it after the splitter."
        )

    if violations and on_leakage == "raise":
        raise ValueError("Data leakage risk:\n" + "\n".join(violations))
    return violations if on_leakage == "warn" else []
