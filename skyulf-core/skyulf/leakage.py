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


def is_explicit_column_drop(step_type: str, params: dict[str, Any]) -> bool:
    """True when a ``DropMissingColumns`` step only drops explicitly named columns.

    With no positive ``missing_threshold`` the node's ``fit()`` just records a
    fixed, user-chosen column list (e.g. "exclude this id column from the
    model") — no statistic is learned from the rows, so running it before the
    train/test split is safe. A positive threshold makes ``fit()`` decide
    WHICH columns to drop from the data it sees; that decision must stay
    after the split, so this returns ``False`` there. Mirrors the node's own
    ``infer_output_schema`` split between its two modes.
    """
    if step_type != "DropMissingColumns":
        return False
    raw = params.get("missing_threshold")
    try:
        threshold = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        threshold = None
    return threshold is None or threshold <= 0


def is_constant_imputation(step_type: str, params: dict[str, Any]) -> bool:
    """True when a ``SimpleImputer`` step fills with a user-fixed constant.

    With ``strategy='constant'`` the fill value comes from the config
    (``fill_value``, defaulting to 0), not from the fitted rows — nothing is
    learned, so running it before the train/test split is safe. The
    ``mean``/``median``/``most_frequent`` strategies compute statistics from
    the data and must stay after the split. Mirrors
    ``imputation._common._compute_polars_fill_values``' constant branch.
    """
    if step_type != "SimpleImputer":
        return False
    return params.get("strategy") == "constant"


def is_explicit_missing_indicator(step_type: str, params: dict[str, Any]) -> bool:
    """True when a ``MissingIndicator`` step flags explicitly named columns.

    With a non-empty ``columns`` list the fit only records that user-chosen
    list — no decision is learned from the rows, so running it before the
    train/test split is safe. With no explicit list the fit discovers WHICH
    columns contain missing values from the data it sees; that decision must
    stay after the split. Mirrors the node's own ``infer_output_schema``
    split between its two modes.
    """
    if step_type != "MissingIndicator":
        return False
    return bool(params.get("columns"))


def is_explicit_hash_encoding(step_type: str, params: dict[str, Any]) -> bool:
    """True when a ``HashEncoder`` step operates on a user-chosen column list.

    Hashing itself is deterministic (fixed ``n_features`` from the config),
    so with an explicit ``columns`` list fit() only records config values —
    nothing is learned from the rows and running before the split is safe.
    An explicit empty list is the UI's "nothing selected" no-op (fit returns
    ``{}``), equally learning nothing. Only when the key is absent does fit
    auto-detect WHICH columns are categorical from the rows it sees; that
    decision must stay after the split. Mirrors the node's own
    ``user_picked_no_columns`` short-circuit in ``fit()``.
    """
    if step_type != "HashEncoder":
        return False
    return isinstance(params.get("columns"), list)


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
        transformer = step.get("transformer") or ""
        if is_explicit_column_drop(transformer, step.get("params") or {}):
            continue
        if is_constant_imputation(transformer, step.get("params") or {}):
            continue
        if is_explicit_missing_indicator(transformer, step.get("params") or {}):
            continue
        if is_explicit_hash_encoding(transformer, step.get("params") or {}):
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
