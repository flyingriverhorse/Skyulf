"""Operation-aware diagnostics for preprocessing data leakage.

Registry metadata describes each node's most data-dependent mode. Explicit
parameter exemptions below distinguish fixed operations from learned ones.
Unknown nodes and unknown operation modes are treated conservatively.
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

_FIXED_TRANSFORMATIONS = frozenset(
    {"log", "sqrt", "square_root", "cube_root", "reciprocal", "square", "exp", "exponential"}
)
_FIXED_FEATURE_OPERATIONS = frozenset({"arithmetic", "ratio", "similarity", "datetime_extract"})
_FEATURE_GENERATORS = frozenset({"FeatureGeneration", "FeatureGenerationNode", "FeatureMath"})
_EMPTY_COLUMN_NOOPS = frozenset(
    {
        "OneHotEncoder",
        "DummyEncoder",
        "TargetEncoder",
        "WOEEncoder",
        "PowerTransformer",
        "StandardScaler",
        "MinMaxScaler",
        "MaxAbsScaler",
        "RobustScaler",
        "SimpleImputer",
        "KNNImputer",
        "IterativeImputer",
        "GeneralBinning",
        "KBinsDiscretizer",
        "CustomBinning",
        "IQR",
        "ZScore",
        "Winsorize",
        "EllipticEnvelope",
    }
)


def data_dependent_transformers() -> frozenset[str]:
    """Return registry IDs with at least one data-dependent fitting mode."""
    return frozenset(
        node_id
        for node_id, meta in NodeRegistry.get_all_metadata().items()
        if meta.get("learns_from_data")
    )


def train_test_splitters() -> frozenset[str]:
    """Return registry IDs that create a train/test row boundary."""
    return frozenset(
        node_id
        for node_id, meta in NodeRegistry.get_all_metadata().items()
        if meta.get("is_splitter")
    )


def is_explicit_column_drop(step_type: str, params: dict[str, Any]) -> bool:
    """Return whether column dropping uses only a fixed configured list."""
    if step_type != "DropMissingColumns":
        return False
    raw = params.get("missing_threshold")
    try:
        threshold = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        threshold = None
    # The implementation only learns a drop list for a positive threshold.
    # NaN, invalid, zero and negative thresholds do not enable that mode.
    return threshold is None or not threshold > 0


def is_constant_imputation(step_type: str, params: dict[str, Any]) -> bool:
    """Return whether imputation fills a configured constant, not a statistic."""
    return step_type == "SimpleImputer" and params.get("strategy") == "constant"


def is_explicit_missing_indicator(step_type: str, params: dict[str, Any]) -> bool:
    """Return whether missingness flags use explicitly named columns."""
    return step_type == "MissingIndicator" and bool(params.get("columns"))


def is_explicit_hash_encoding(step_type: str, params: dict[str, Any]) -> bool:
    """Return whether hashing uses an explicit feature list, including no-op []."""
    return step_type == "HashEncoder" and isinstance(params.get("columns"), list)


def is_target_only_encoding(
    step_type: str,
    params: dict[str, Any],
    target_column: str | None = None,
) -> bool:
    """Return whether an encoder changes only target labels or does nothing.

    LabelEncoder defaults to y. OrdinalEncoder defaults to auto-detected
    feature columns, so omitted or null columns are NOT target-only there.
    The caller supplies authoritative target context; a node cannot grant
    itself an exemption merely by declaring its own target_column.
    """
    if step_type not in {"LabelEncoder", "OrdinalEncoder"}:
        return False
    columns = params.get("columns")
    if step_type == "LabelEncoder" and not columns:
        return True
    if isinstance(columns, list) and not columns:
        return True
    return (
        isinstance(columns, list) and target_column is not None and set(columns) == {target_column}
    )


def leakage_exemption_reason(
    step_type: str,
    params: dict[str, Any],
    *,
    target_column: str | None = None,
) -> str | None:
    """Explain why a configured mode does not fit feature statistics.

    This is shared by core diagnostics and backend admission/fold planning.
    It is not a provenance guarantee: a fixed formula can still reference a
    target, future observation, or feature computed outside this pipeline.
    """
    if is_target_only_encoding(step_type, params, target_column):
        return "target-only encoding or explicit no-op"
    if is_explicit_column_drop(step_type, params):
        return "explicit column drop without a learned missingness threshold"
    if is_constant_imputation(step_type, params):
        return "constant imputation"
    if is_explicit_missing_indicator(step_type, params):
        return "explicit missing-indicator columns"
    if is_explicit_hash_encoding(step_type, params):
        return "hash encoding with explicit columns"
    columns = params.get("columns")
    if step_type in {"count_vectorizer", "tfidf_vectorizer"}:
        if columns is None or columns == []:
            return "text vectorization requires an explicit nonempty column selection"
        if (
            isinstance(columns, list)
            and target_column is not None
            and set(columns) == {target_column}
        ):
            return "target columns are excluded from text vectorization"
    if step_type in _EMPTY_COLUMN_NOOPS and isinstance(columns, list) and not columns:
        return "explicit empty column selection is a no-op"
    if step_type == "GeneralTransformation":
        operations = params.get("transformations", [])
        if isinstance(operations, list) and all(
            isinstance(op, dict) and op.get("method") in _FIXED_TRANSFORMATIONS for op in operations
        ):
            return "fixed row-wise mathematical transformations"
    if step_type in _FEATURE_GENERATORS:
        operations = params.get("operations", [])
        if isinstance(operations, list) and all(
            isinstance(op, dict)
            and op.get("operation_type", "arithmetic") in _FIXED_FEATURE_OPERATIONS
            for op in operations
        ):
            return "fixed row-wise feature operations"
    if step_type == "CustomBinning" and isinstance(columns, list):
        return "fixed bin edges with explicit columns"
    if step_type == "Casting":
        raw_types = params.get("column_types", {})
        if isinstance(raw_types, dict):
            type_map = dict(raw_types)
            target_type = params.get("target_type")
            if isinstance(columns, list) and columns and target_type:
                type_map.update(dict.fromkeys(columns, target_type))
            if all(
                isinstance(dtype, str) and dtype.lower() not in {"category", "categorical"}
                for dtype in type_map.values()
            ):
                return "fixed non-categorical type casts"
    return None


def step_learns_from_data(
    step_type: str,
    params: dict[str, Any],
    *,
    target_column: str | None = None,
) -> bool:
    """Classify one configured step; unknown implementations fail closed."""
    if leakage_exemption_reason(step_type, params, target_column=target_column) is not None:
        return False
    metadata = NodeRegistry.get_all_metadata().get(step_type, {})
    return metadata.get("learns_from_data") is not False


def validate_leakage_safety(
    pipeline_config: PipelineConfig | dict[str, Any],
    on_leakage: OnLeakage = "raise",
    *,
    target_column: str | None = None,
    already_split: bool = False,
) -> list[str]:
    """Diagnose learned preprocessing configured before a train/test split.

    Args:
        pipeline_config: Core linear pipeline configuration.
        on_leakage: Raise for definite violations, return warnings, or ignore.
        target_column: Authoritative target label name, when known.
        already_split: Input is already partitioned into train and held-out rows.

    Returns:
        Warning strings for warn mode, or the no-split advisory in raise mode.

    Raises:
        ValueError: Mode is invalid, or a definite violation uses raise mode.

    No-split configurations receive an advisory rather than an exception.
    An external SplitDataset supplies a boundary before all configured steps.
    Neither warn nor ignore repairs an unsafe ordering.
    """
    if on_leakage not in _ON_LEAKAGE_MODES:
        raise ValueError(
            f"on_leakage must be one of {sorted(_ON_LEAKAGE_MODES)}, got {on_leakage!r}"
        )
    if already_split:
        return []

    preprocessing = pipeline_config.get("preprocessing", [])
    splitters = train_test_splitters()
    if target_column is None:
        for step in preprocessing:
            if step.get("transformer") in splitters | {"feature_target_split"}:
                candidate = (step.get("params") or {}).get("target_column")
                if candidate:
                    target_column = candidate
                    break
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
    metadata = NodeRegistry.get_all_metadata()
    violations = []
    for index, step in enumerate(preprocessing[:splitter_index]):
        transformer = step.get("transformer") or ""
        if not step_learns_from_data(
            transformer, step.get("params") or {}, target_column=target_column
        ):
            continue
        if transformer in metadata:
            reason = "fits its statistics on the full dataset including the test set"
        else:
            reason = "is not a known node and is treated as data-dependent until proven otherwise"
        violations.append(
            f"Step {index} ('{transformer}') is configured before the train/test split "
            f"(step {splitter_index}, '{splitter_name}') and {reason} - move it after the splitter."
        )

    if violations and on_leakage == "raise":
        raise ValueError("Data leakage risk:\n" + "\n".join(violations))
    return violations if on_leakage == "warn" else []
