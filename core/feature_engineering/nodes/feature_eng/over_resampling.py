"""Class oversampling helpers for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import math

import pandas as pd
from pandas.api import types as pd_types

from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, SMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek

from core.feature_engineering.schemas import ClassOversamplingNodeSignal
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

ResamplingSamplingStrategy = Union[str, float]

OVERSAMPLING_METHOD_SMOTE = "smote"
OVERSAMPLING_METHOD_ADASYN = "adasyn"
OVERSAMPLING_METHOD_BORDERLINE_SMOTE = "borderline_smote"
OVERSAMPLING_METHOD_KMEANS_SMOTE = "kmeans_smote"
OVERSAMPLING_METHOD_SVM_SMOTE = "svm_smote"
OVERSAMPLING_METHOD_SMOTE_TOMEK = "smote_tomek"
OVERSAMPLING_METHOD_LABELS: Dict[str, str] = {
    OVERSAMPLING_METHOD_SMOTE: "SMOTE",
    OVERSAMPLING_METHOD_ADASYN: "ADASYN",
    OVERSAMPLING_METHOD_BORDERLINE_SMOTE: "Borderline SMOTE",
    OVERSAMPLING_METHOD_KMEANS_SMOTE: "KMeans SMOTE",
    OVERSAMPLING_METHOD_SVM_SMOTE: "SVM SMOTE",
    OVERSAMPLING_METHOD_SMOTE_TOMEK: "SMOTE Tomek",
}
OVERSAMPLING_SUPPORTED_METHODS = frozenset(OVERSAMPLING_METHOD_LABELS.keys())
OVERSAMPLING_DEFAULT_METHOD = OVERSAMPLING_METHOD_SMOTE
OVERSAMPLING_DEFAULT_SAMPLING_STRATEGY: ResamplingSamplingStrategy = "auto"
OVERSAMPLING_DEFAULT_RANDOM_STATE = 42
OVERSAMPLING_DEFAULT_REPLACEMENT = False
OVERSAMPLING_DEFAULT_K_NEIGHBORS = 5


@dataclass
class NormalizedOversamplingConfig:
    method: str
    target_column: str
    sampling_strategy: ResamplingSamplingStrategy
    random_state: Optional[int]
    replacement: bool
    k_neighbors: int


@dataclass
class _OversamplingPrep:
    features: pd.DataFrame
    labels: pd.Series
    feature_columns: List[str]
    integer_feature_columns: List[str]
    missing_rows: pd.DataFrame
    missing_count: int
    split_value: Optional[Any]
    original_total: int
    before_counts: Dict[Any, int]
    min_class_size: int
    frame_columns: List[str]
    has_split: bool


def _normalize_sampling_strategy(raw_value: Any) -> ResamplingSamplingStrategy:
    if raw_value is None:
        return OVERSAMPLING_DEFAULT_SAMPLING_STRATEGY

    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return OVERSAMPLING_DEFAULT_SAMPLING_STRATEGY

        lowered = text.lower()
        normalized_keyword = lowered.replace("_", " ")
        if normalized_keyword in {"auto", "minority", "not minority", "not majority", "all"}:
            return normalized_keyword

        try:
            numeric = float(text)
        except (TypeError, ValueError):
            return OVERSAMPLING_DEFAULT_SAMPLING_STRATEGY
        if not math.isfinite(numeric) or numeric <= 0:
            return OVERSAMPLING_DEFAULT_SAMPLING_STRATEGY
        return min(numeric, 1.0)

    if isinstance(raw_value, (int, float)):
        numeric = float(raw_value)
        if not math.isfinite(numeric) or numeric <= 0:
            return OVERSAMPLING_DEFAULT_SAMPLING_STRATEGY
        return min(numeric, 1.0)

    return OVERSAMPLING_DEFAULT_SAMPLING_STRATEGY


def _normalize_random_state(raw_value: Any) -> Optional[int]:
    if raw_value is None:
        return None

    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except (TypeError, ValueError):
            return None
    else:
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            return None

    if math.isnan(numeric):
        return None

    try:
        return int(round(numeric))
    except (TypeError, ValueError, OverflowError):  # pragma: no cover - defensive
        return None


def _normalize_k_neighbors(raw_value: Any) -> int:
    if raw_value is None:
        return OVERSAMPLING_DEFAULT_K_NEIGHBORS

    try:
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return OVERSAMPLING_DEFAULT_K_NEIGHBORS
            numeric = float(text)
        else:
            numeric = float(raw_value)
    except (TypeError, ValueError):
        return OVERSAMPLING_DEFAULT_K_NEIGHBORS

    if not math.isfinite(numeric):
        return OVERSAMPLING_DEFAULT_K_NEIGHBORS

    coerced = int(round(numeric))
    return max(1, coerced)


def _normalize_config(config: Any) -> NormalizedOversamplingConfig:
    if not isinstance(config, dict):
        config = {}

    method = str(config.get("method") or OVERSAMPLING_DEFAULT_METHOD).strip().lower()
    if method not in OVERSAMPLING_SUPPORTED_METHODS:
        method = OVERSAMPLING_DEFAULT_METHOD

    target_column = str(config.get("target_column") or config.get("target") or "").strip()
    sampling_strategy = _normalize_sampling_strategy(config.get("sampling_strategy"))
    random_state = _normalize_random_state(config.get("random_state"))
    replacement = bool(config.get("replacement", OVERSAMPLING_DEFAULT_REPLACEMENT))
    k_neighbors = _normalize_k_neighbors(config.get("k_neighbors"))

    return NormalizedOversamplingConfig(
        method=method,
        target_column=target_column,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        replacement=replacement,
        k_neighbors=k_neighbors,
    )


def _build_sampler(config: NormalizedOversamplingConfig, k_neighbors: int):
    common_kwargs = {
        "sampling_strategy": config.sampling_strategy,
        "random_state": config.random_state,
    }

    if config.method == OVERSAMPLING_METHOD_SMOTE:
        return SMOTE(k_neighbors=k_neighbors, **common_kwargs)
    if config.method == OVERSAMPLING_METHOD_ADASYN:
        return ADASYN(n_neighbors=k_neighbors, **common_kwargs)
    if config.method == OVERSAMPLING_METHOD_BORDERLINE_SMOTE:
        return BorderlineSMOTE(k_neighbors=k_neighbors, **common_kwargs)
    if config.method == OVERSAMPLING_METHOD_KMEANS_SMOTE:
        return KMeansSMOTE(k_neighbors=k_neighbors, **common_kwargs)
    if config.method == OVERSAMPLING_METHOD_SVM_SMOTE:
        return SVMSMOTE(k_neighbors=k_neighbors, **common_kwargs)
    if config.method == OVERSAMPLING_METHOD_SMOTE_TOMEK:
        smote = SMOTE(k_neighbors=k_neighbors, **common_kwargs)
        return SMOTETomek(smote=smote)

    raise ValueError(f"Unsupported oversampling method '{config.method}'")


def _format_class_counts(counts: Dict[Any, int]) -> str:
    if not counts:
        return "no classes"

    fragments: List[str] = []
    for label, count in sorted(counts.items(), key=lambda item: str(item[0])):
        label_text = "<NA>" if pd.isna(label) else str(label)
        fragments.append(f"{label_text}: {int(count)}")
    return ", ".join(fragments)


def _normalize_counts_for_signal(counts: Dict[Any, int]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for label, count in counts.items():
        label_text = "<NA>" if pd.isna(label) else str(label)
        normalized[label_text] = int(count)
    return normalized


def _format_sampling_strategy_label(value: ResamplingSamplingStrategy) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if not math.isfinite(numeric):  # pragma: no cover - defensive
        return None
    return f"{numeric:.3f}"


def _resolve_split_value(frame: pd.DataFrame, signal: ClassOversamplingNodeSignal) -> Optional[Any]:
    if SPLIT_TYPE_COLUMN not in frame.columns:
        return None

    split_series = frame[SPLIT_TYPE_COLUMN]
    non_null_split = split_series.dropna()
    split_value: Optional[Any] = None

    if not non_null_split.empty:
        unique_splits = pd.unique(non_null_split)
        split_value = unique_splits[0]
        if unique_splits.size > 1:
            signal.notes.append(
                "Detected multiple split labels; preserving first value for synthetic rows."
            )

    if split_value is None and not split_series.empty:
        split_value = split_series.iloc[0]

    return split_value


def _prepare_oversampling_inputs(
    frame: pd.DataFrame,
    config: NormalizedOversamplingConfig,
    signal: ClassOversamplingNodeSignal,
) -> Tuple[Optional[_OversamplingPrep], Optional[str]]:
    split_value = _resolve_split_value(frame, signal)

    if frame.empty:
        signal.warnings.append("No data available for oversampling.")
        return None, "Class oversampling: no data available"

    if not config.target_column:
        signal.warnings.append("Target column not configured.")
        return None, "Class oversampling: target column not configured"

    if config.target_column not in frame.columns:
        signal.warnings.append(
            f"Target column '{config.target_column}' not found in preview frame."
        )
        return None, f"Class oversampling: target column '{config.target_column}' not found"

    target_series = frame[config.target_column]
    non_null_mask = target_series.notna()
    valid_frame = frame[non_null_mask]
    missing_rows = frame[~non_null_mask]
    missing_count = int(missing_rows.shape[0])
    signal.preserved_missing_rows = missing_count

    if valid_frame.empty:
        signal.warnings.append("Target column contains only missing values.")
        summary = (
            f"Class oversampling: target column '{config.target_column}' has only missing values"
        )
        return None, summary

    unique_classes = pd.unique(valid_frame[config.target_column])
    if unique_classes.size <= 1:
        signal.warnings.append("Target column must contain at least two classes.")
        summary = (
            f"Class oversampling: target column '{config.target_column}' has fewer than two classes"
        )
        return None, summary

    excluded_feature_columns = {config.target_column}
    if SPLIT_TYPE_COLUMN in frame.columns:
        excluded_feature_columns.add(SPLIT_TYPE_COLUMN)

    feature_columns = [column for column in frame.columns if column not in excluded_feature_columns]
    if not feature_columns:
        signal.warnings.append("At least one feature column (besides target) is required.")
        summary = "Class oversampling: requires at least one feature column besides the target"
        return None, summary

    features = valid_frame[feature_columns]
    non_numeric_columns = [column for column in feature_columns if not pd_types.is_numeric_dtype(features[column])]
    if non_numeric_columns:
        column_list = ", ".join(non_numeric_columns[:8])
        if len(non_numeric_columns) > 8:
            column_list = f"{column_list}, ..."
        message = (
            "Class oversampling: all feature columns must be numeric. "
            f"Non-numeric columns detected: {column_list}. "
            "Apply encoding or remove them before oversampling."
        )
        signal.warnings.append("Non-numeric feature columns detected.")
        return None, message

    integer_feature_columns = [
        column for column in feature_columns if pd_types.is_integer_dtype(features[column])
    ]

    if integer_feature_columns:
        features = features.copy()
        for column in integer_feature_columns:
            features[column] = features[column].astype("float64")
        signal.integer_cast_columns = [str(column) for column in integer_feature_columns]

    labels = valid_frame[config.target_column]
    before_counts = labels.value_counts(dropna=False).to_dict()
    class_counts_series = labels.value_counts(dropna=False)
    min_class_size = int(class_counts_series.min()) if not class_counts_series.empty else 0

    prep = _OversamplingPrep(
        features=features,
        labels=labels,
        feature_columns=feature_columns,
        integer_feature_columns=integer_feature_columns,
        missing_rows=missing_rows,
        missing_count=missing_count,
        split_value=split_value,
        original_total=int(frame.shape[0]),
        before_counts=before_counts,
        min_class_size=min_class_size,
        frame_columns=list(frame.columns),
        has_split=SPLIT_TYPE_COLUMN in frame.columns,
    )

    return prep, None


def _compute_effective_k_neighbors(
    prep: _OversamplingPrep,
    config: NormalizedOversamplingConfig,
    signal: ClassOversamplingNodeSignal,
) -> Tuple[Optional[int], Optional[str]]:
    signal.min_class_size = prep.min_class_size

    if prep.min_class_size <= 1:
        signal.warnings.append("Minority class must have at least two samples for oversampling.")
        summary = (
            "Class oversampling: minority class has only "
            f"{prep.min_class_size} sample(s). Need at least two samples."
        )
        return None, summary

    effective_k_neighbors = min(config.k_neighbors, max(1, prep.min_class_size - 1))
    signal.effective_k_neighbors = effective_k_neighbors

    if effective_k_neighbors != config.k_neighbors:
        signal.adjusted_parameters["k_neighbors"] = effective_k_neighbors
        signal.notes.append(
            f"Adjusted k_neighbors from {config.k_neighbors} to {effective_k_neighbors} based on class size."
        )

    return effective_k_neighbors, None


def _resample_dataset(
    prep: _OversamplingPrep,
    config: NormalizedOversamplingConfig,
    signal: ClassOversamplingNodeSignal,
    effective_k_neighbors: int,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        sampler = _build_sampler(config, effective_k_neighbors)
    except ValueError as exc:
        signal.warnings.append(str(exc))
        return None, f"Class oversampling: {exc}"

    try:
        resampled_features, resampled_labels = sampler.fit_resample(prep.features, prep.labels)
    except TypeError as exc:
        lowered_error = str(exc).lower()
        if "cast" in lowered_error and "int" in lowered_error:
            column_list = ", ".join(prep.integer_feature_columns[:8]) if prep.integer_feature_columns else ""
            if prep.integer_feature_columns and len(prep.integer_feature_columns) > 8:
                column_list = f"{column_list}, ..."
            details = f" Columns affected: {column_list}." if column_list else ""
            warning = (
                "Generated synthetic values introduced decimals for integer-constrained columns. "
                "Cast affected columns to float before oversampling."
            )
            signal.warnings.append(warning)
            message = (
                "Class oversampling: generated synthetic values with decimals but some feature columns "
                "enforce integer dtype. Cast those columns to float before oversampling."
                + details
            )
            return None, message
        signal.warnings.append("Type casting error during oversampling.")
        return (
            None,
            "Class oversampling: unable to apply sampling due to type casting error. "
            "Convert integer-restricted columns to float before oversampling.",
        )
    except ValueError as exc:  # pragma: no cover - imblearn validation feedback
        lowered_error = str(exc).lower()
        if "nan" in lowered_error or "missing" in lowered_error:
            signal.warnings.append("Missing values detected in feature columns.")
            message = (
                "Class oversampling: missing values detected in feature columns. "
                "Impute or drop rows with NaNs before running oversampling."
            )
            return None, message
        signal.warnings.append(str(exc))
        return None, f"Class oversampling: unable to apply sampling ({exc})"

    resampled_valid = pd.DataFrame(resampled_features, columns=prep.feature_columns)
    resampled_valid[config.target_column] = resampled_labels

    if prep.has_split:
        resampled_valid[SPLIT_TYPE_COLUMN] = prep.split_value

    ordered_columns = [column for column in prep.frame_columns if column in resampled_valid.columns]
    resampled_valid = resampled_valid.loc[:, ordered_columns]

    return resampled_valid, None


def apply_oversampling(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, ClassOversamplingNodeSignal]:
    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_config(config_payload)

    node_id = node.get("id") if isinstance(node, dict) else None
    method_label = OVERSAMPLING_METHOD_LABELS.get(config.method, config.method.replace("_", " "))

    signal = ClassOversamplingNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        method=config.method,
        method_label=method_label,
        target_column=config.target_column or None,
        sampling_strategy=config.sampling_strategy,
        sampling_strategy_label=_format_sampling_strategy_label(config.sampling_strategy),
        random_state=config.random_state,
        replacement=config.replacement,
        k_neighbors=config.k_neighbors,
    )

    signal.total_rows_before = int(frame.shape[0])
    signal.total_rows_after = signal.total_rows_before

    prep, error_summary = _prepare_oversampling_inputs(frame, config, signal)
    if error_summary:
        return frame, error_summary, signal
    assert prep is not None

    signal.class_counts_before = _normalize_counts_for_signal(prep.before_counts)

    effective_k_neighbors, neighbor_error = _compute_effective_k_neighbors(prep, config, signal)
    if neighbor_error:
        return frame, neighbor_error, signal
    assert effective_k_neighbors is not None

    resampled_valid, resample_error = _resample_dataset(prep, config, signal, effective_k_neighbors)
    if resample_error:
        return frame, resample_error, signal
    assert resampled_valid is not None

    if not prep.missing_rows.empty:
        resampled_frame = pd.concat([resampled_valid, prep.missing_rows.copy()], axis=0, ignore_index=True)
    else:
        resampled_frame = resampled_valid.reset_index(drop=True)

    after_counts = resampled_frame[config.target_column].value_counts(dropna=False).to_dict()
    after_total = int(resampled_frame.shape[0])

    signal.class_counts_after = _normalize_counts_for_signal(after_counts)
    signal.total_rows_after = after_total

    summary_parts = [
        f"{method_label}: rows {prep.original_total}→{after_total}",
        (
            f"{config.target_column} counts "
            f"{_format_class_counts(prep.before_counts)} → {_format_class_counts(after_counts)}"
        ),
        f"k_neighbors={effective_k_neighbors}",
    ]

    if prep.split_value is not None:
        summary_parts.append(f"split={prep.split_value}")

    if prep.split_value is not None and str(prep.split_value).lower() == "train":
        signal.notes.append("Applied oversampling to training split only; test/validation unchanged.")

    if prep.integer_feature_columns:
        preview = ", ".join(prep.integer_feature_columns[:3])
        if len(prep.integer_feature_columns) > 3:
            preview = f"{preview}, ..."
        summary_parts.append(f"auto-cast integer feature columns to float ({preview})")

    if prep.missing_count:
        summary_parts.append(f"preserved {prep.missing_count} row(s) with missing target")

    summary = "; ".join(summary_parts)

    return resampled_frame.reset_index(drop=True), summary, signal


__all__ = [
    "OVERSAMPLING_METHOD_LABELS",
    "OVERSAMPLING_DEFAULT_METHOD",
    "OVERSAMPLING_DEFAULT_SAMPLING_STRATEGY",
    "OVERSAMPLING_DEFAULT_RANDOM_STATE",
    "OVERSAMPLING_DEFAULT_REPLACEMENT",
    "OVERSAMPLING_DEFAULT_K_NEIGHBORS",
    "apply_oversampling",
]
