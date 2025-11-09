"""Class undersampling helpers for feature engineering nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import math

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from .utils import _coerce_config_boolean
from core.feature_engineering.schemas import ClassUndersamplingNodeSignal
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

RESAMPLING_METHOD_RANDOM_UNDER = "random_under_sampling"
RESAMPLING_METHOD_LABELS: Dict[str, str] = {
    RESAMPLING_METHOD_RANDOM_UNDER: "Random under-sampling",
}
RESAMPLING_SUPPORTED_METHODS = set(RESAMPLING_METHOD_LABELS.keys())
RESAMPLING_DEFAULT_METHOD = RESAMPLING_METHOD_RANDOM_UNDER
RESAMPLING_DEFAULT_SAMPLING_STRATEGY = "auto"
RESAMPLING_DEFAULT_RANDOM_STATE = 42
RESAMPLING_DEFAULT_REPLACEMENT = False

ResamplingSamplingStrategy = Union[str, float]


@dataclass
class NormalizedResamplingConfig:
    method: str
    target_column: str
    sampling_strategy: ResamplingSamplingStrategy
    random_state: Optional[int]
    replacement: bool


def _normalize_sampling_strategy(raw_value: Any) -> ResamplingSamplingStrategy:
    if raw_value is None:
        return RESAMPLING_DEFAULT_SAMPLING_STRATEGY

    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return RESAMPLING_DEFAULT_SAMPLING_STRATEGY

        lowered = text.lower()
        normalized_keyword = lowered.replace("_", " ")
        if normalized_keyword in {"auto", "majority", "not minority", "not majority", "all"}:
            return normalized_keyword

        try:
            numeric = float(text)
        except (TypeError, ValueError):
            return RESAMPLING_DEFAULT_SAMPLING_STRATEGY
        if not math.isfinite(numeric) or numeric <= 0:
            return RESAMPLING_DEFAULT_SAMPLING_STRATEGY
        return min(numeric, 1.0)

    if isinstance(raw_value, (int, float)):
        numeric = float(raw_value)
        if not math.isfinite(numeric) or numeric <= 0:
            return RESAMPLING_DEFAULT_SAMPLING_STRATEGY
        return min(numeric, 1.0)

    return RESAMPLING_DEFAULT_SAMPLING_STRATEGY


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


def _normalize_resampling_config(config: Any) -> NormalizedResamplingConfig:
    if not isinstance(config, dict):
        config = {}

    method = str(config.get("method") or RESAMPLING_DEFAULT_METHOD).strip().lower()
    if method not in RESAMPLING_SUPPORTED_METHODS:
        method = RESAMPLING_DEFAULT_METHOD

    target_column = str(config.get("target_column") or config.get("target") or "").strip()

    sampling_strategy = _normalize_sampling_strategy(config.get("sampling_strategy"))
    random_state = _normalize_random_state(config.get("random_state"))
    replacement = _coerce_config_boolean(config.get("replacement"), default=RESAMPLING_DEFAULT_REPLACEMENT)

    return NormalizedResamplingConfig(
        method=method,
        target_column=target_column,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        replacement=replacement,
    )


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


def apply_resampling(
    frame: pd.DataFrame,
    node: Dict[str, Any],
) -> Tuple[pd.DataFrame, str, ClassUndersamplingNodeSignal]:
    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_resampling_config(config_payload)

    node_id = node.get("id") if isinstance(node, dict) else None
    method_label = RESAMPLING_METHOD_LABELS.get(config.method, config.method.replace("_", " "))

    signal = ClassUndersamplingNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
        method=config.method,
        method_label=method_label,
        target_column=config.target_column or None,
        sampling_strategy=config.sampling_strategy,
        sampling_strategy_label=_format_sampling_strategy_label(config.sampling_strategy),
        random_state=config.random_state,
        replacement=config.replacement,
    )

    signal.total_rows_before = int(frame.shape[0])
    signal.total_rows_after = signal.total_rows_before

    split_value: Optional[Any] = None
    if SPLIT_TYPE_COLUMN in frame.columns:
        split_series = frame[SPLIT_TYPE_COLUMN]
        non_null_split = split_series.dropna()
        if not non_null_split.empty:
            unique_splits = non_null_split.unique()
            split_value = unique_splits[0]
            if unique_splits.size > 1:
                signal.notes.append(
                    "Detected multiple split labels; preserving first value for sampled rows."
                )

        if split_value is None and not split_series.empty:
            split_value = split_series.iloc[0]

    if frame.empty:
        signal.warnings.append("No data available for undersampling.")
        return frame, "Class undersampling: no data available", signal

    if not config.target_column:
        signal.warnings.append("Target column not configured.")
        return frame, "Class undersampling: target column not configured", signal

    if config.target_column not in frame.columns:
        signal.warnings.append(f"Target column '{config.target_column}' not found in preview frame.")
        return frame, f"Class undersampling: target column '{config.target_column}' not found", signal

    target_series = frame[config.target_column]
    non_null_mask = target_series.notna()
    valid_frame = frame[non_null_mask]
    missing_rows = frame[~non_null_mask]
    missing_count = int(missing_rows.shape[0])
    signal.preserved_missing_rows = missing_count

    if valid_frame.empty:
        signal.warnings.append("Target column contains only missing values.")
        return frame, f"Class undersampling: target column '{config.target_column}' has only missing values", signal

    unique_classes = pd.unique(valid_frame[config.target_column])
    if unique_classes.size <= 1:
        signal.warnings.append("Target column must contain at least two classes.")
        return frame, f"Class undersampling: target column '{config.target_column}' has fewer than two classes", signal

    excluded_feature_columns = {config.target_column}
    if SPLIT_TYPE_COLUMN in frame.columns:
        excluded_feature_columns.add(SPLIT_TYPE_COLUMN)

    feature_columns = [column for column in frame.columns if column not in excluded_feature_columns]
    if feature_columns:
        features = valid_frame[feature_columns]
    else:
        features = valid_frame[[config.target_column]]

    labels = valid_frame[config.target_column]

    before_counts = labels.value_counts(dropna=False).to_dict()
    signal.class_counts_before = _normalize_counts_for_signal(before_counts)

    original_total = int(frame.shape[0])

    if config.method != RESAMPLING_METHOD_RANDOM_UNDER:
        message = f"Class undersampling: method '{config.method}' is not supported yet"
        signal.warnings.append("Unsupported undersampling method.")
        return frame, message, signal

    sampler = RandomUnderSampler(
        sampling_strategy=config.sampling_strategy,
        random_state=config.random_state,
        replacement=config.replacement,
    )

    try:
        sampler.fit_resample(features, labels)
    except ValueError as exc:  # pragma: no cover - imblearn validation feedback
        signal.warnings.append(str(exc))
        return frame, f"Class undersampling: unable to apply sampling ({exc})", signal

    indices = getattr(sampler, "sample_indices_", None)
    if indices is None:  # pragma: no cover - fallback guard
        resampled_valid = valid_frame.copy()
    else:
        resampled_valid = valid_frame.iloc[indices].copy()

    if not missing_rows.empty:
        resampled_frame = pd.concat(
            [resampled_valid, missing_rows.copy()], axis=0, ignore_index=True
        )
    else:
        resampled_frame = resampled_valid.reset_index(drop=True)

    resampled_frame = resampled_frame.loc[:, frame.columns]
    resampled_target = resampled_frame[config.target_column]
    after_counts = resampled_target.value_counts(dropna=False).to_dict()
    after_total = int(resampled_frame.shape[0])

    signal.class_counts_after = _normalize_counts_for_signal(after_counts)
    signal.total_rows_after = after_total

    method_label = RESAMPLING_METHOD_LABELS.get(config.method, config.method.replace("_", " "))
    summary = (
        f"{method_label}: rows {original_total}→{after_total}; "
        f"{config.target_column} counts {_format_class_counts(before_counts)} → "
        f"{_format_class_counts(after_counts)}"
    )

    if split_value is not None:
        summary = f"{summary}; split={split_value}"

    if split_value is not None and str(split_value).lower() == "train":
        signal.notes.append("Applied undersampling to training split only; test/validation unchanged.")

    if missing_count:
        summary = f"{summary}; preserved {missing_count} row(s) with missing target"

    return resampled_frame.reset_index(drop=True), summary, signal


__all__ = [
    "RESAMPLING_METHOD_RANDOM_UNDER",
    "RESAMPLING_METHOD_LABELS",
    "RESAMPLING_DEFAULT_METHOD",
    "RESAMPLING_DEFAULT_SAMPLING_STRATEGY",
    "RESAMPLING_DEFAULT_RANDOM_STATE",
    "RESAMPLING_DEFAULT_REPLACEMENT",
    "apply_resampling",
]
