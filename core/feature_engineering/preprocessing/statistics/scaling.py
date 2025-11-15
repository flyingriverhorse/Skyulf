"""Scaling helpers for statistics-focused preprocessing."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal, cast

import pandas as pd

from core.feature_engineering.schemas import (
    ScalingAppliedColumnSignal,
    ScalingColumnRecommendation,
    ScalingColumnStats,
    ScalingMethodDetail,
    ScalingMethodName,
    ScalingNodeSignal,
)

try:  # pragma: no cover - optional dependency guard
    from sklearn.preprocessing import (  # type: ignore[import-not-found]
        MaxAbsScaler,
        MinMaxScaler,
        RobustScaler,
        StandardScaler,
    )
except Exception:  # pragma: no cover - defensive guard
    StandardScaler = None  # type: ignore[assignment]
    MinMaxScaler = None  # type: ignore[assignment]
    MaxAbsScaler = None  # type: ignore[assignment]
    RobustScaler = None  # type: ignore[assignment]

from core.feature_engineering.pipeline_store_singleton import get_pipeline_store
from core.feature_engineering.preprocessing.split import SPLIT_TYPE_COLUMN

from ...shared.utils import (
    _coerce_config_boolean,
    _detect_numeric_columns,
    _is_binary_numeric,
)

logger = logging.getLogger(__name__)

SCALING_METHODS: Dict[str, Dict[str, Any]] = {
    "standard": {
        "label": "Standard scaler",
        "description": "Centers features and scales to unit variance.",
        "handles_negative": True,
        "handles_zero": True,
        "handles_outliers": False,
        "strengths": [
            "Well-suited for approximately Gaussian distributions",
            "Works with positive and negative values",
        ],
        "cautions": [
            "Sensitive to heavy tails and extreme outliers",
        ],
    },
    "robust": {
        "label": "Robust scaler",
        "description": "Centers using the median and scales by IQR to reduce outlier impact.",
        "handles_negative": True,
        "handles_zero": True,
        "handles_outliers": True,
        "strengths": [
            "Stable when heavy tails or outliers are present",
        ],
        "cautions": [
            "Output is not bounded; may still require clipping for extreme values",
        ],
    },
    "minmax": {
        "label": "Min-Max scaler",
        "description": "Maps values into the [0, 1] range by default.",
        "handles_negative": False,
        "handles_zero": True,
        "handles_outliers": False,
        "strengths": [
            "Preserves shape of original distribution",
            "Ideal for bounded, non-negative features",
        ],
        "cautions": [
            "Extremely sensitive to minimum and maximum outliers",
        ],
    },
    "maxabs": {
        "label": "MaxAbs scaler",
        "description": "Scales by the maximum absolute value to keep sparse data structure.",
        "handles_negative": True,
        "handles_zero": True,
        "handles_outliers": False,
        "strengths": [
            "Keeps zero entries at zero (sparse friendly)",
            "Works for symmetric data around zero",
        ],
        "cautions": [
            "Large outliers dominate the scaling factor",
        ],
    },
}

SCALING_METHOD_ORDER: Tuple[str, ...] = ("standard", "robust", "minmax", "maxabs")
SCALING_DEFAULT_METHOD = "standard"
ScalingConfidence = Literal["high", "medium", "low"]


@dataclass
class NormalizedScalingConfig:
    columns: List[str]
    default_method: str
    column_methods: Dict[str, str]
    auto_detect: bool
    skipped_columns: List[str]


@dataclass
class ScalingColumnContext:
    column: str
    numeric_series: pd.Series
    valid_mask: pd.Series
    valid_values: pd.Series
    method: ScalingMethodName


@dataclass
class ScalingEnvironment:
    has_splits: bool
    storage: Optional[Any]
    pipeline_id: Optional[str]
    node_id: Optional[str]
    split_counts: Dict[str, int]
    train_mask: Optional[pd.Series]
    train_row_count: int

    @property
    def can_persist(self) -> bool:
        return self.has_splits and self.storage is not None and self.pipeline_id is not None


@dataclass
class ScalingProcessResult:
    transformed_values: Optional[pd.Series] = None
    effective_method: Optional[ScalingMethodName] = None
    skip_reason: Optional[str] = None
    signal_reason: Optional[str] = None


def _normalize_scaling_config(config: Any) -> NormalizedScalingConfig:
    if not isinstance(config, dict):
        config = {}

    raw_columns = config.get("columns")
    if isinstance(raw_columns, str):
        candidate_columns = [segment.strip() for segment in raw_columns.split(",") if segment.strip()]
    elif isinstance(raw_columns, (list, tuple, set)):
        candidate_columns = [str(entry).strip() for entry in raw_columns if str(entry).strip()]
    else:
        candidate_columns = []

    columns: List[str] = []
    seen: set[str] = set()
    for column in candidate_columns:
        if column and column not in seen:
            seen.add(column)
            columns.append(column)

    raw_default_method = str(config.get("default_method") or SCALING_DEFAULT_METHOD).strip().lower()
    default_method = raw_default_method if raw_default_method in SCALING_METHODS else SCALING_DEFAULT_METHOD

    column_methods: Dict[str, str] = {}
    raw_column_methods = config.get("column_methods")
    if isinstance(raw_column_methods, dict):
        for key, value in raw_column_methods.items():
            column = str(key or "").strip()
            method = str(value or "").strip().lower()
            if column and method in SCALING_METHODS:
                column_methods[column] = method

    auto_detect = _coerce_config_boolean(config.get("auto_detect"), default=True)

    raw_skipped = config.get("skipped_columns")
    if isinstance(raw_skipped, str):
        candidate_skipped = [segment.strip() for segment in raw_skipped.split(",") if segment.strip()]
    elif isinstance(raw_skipped, (list, tuple, set)):
        candidate_skipped = [str(entry).strip() for entry in raw_skipped if str(entry).strip()]
    else:
        candidate_skipped = []

    skipped_columns: List[str] = []
    skipped_seen: set[str] = set()
    for column in candidate_skipped:
        if column and column not in skipped_seen:
            skipped_seen.add(column)
            skipped_columns.append(column)

    return NormalizedScalingConfig(
        columns=columns,
        default_method=default_method,
        column_methods=column_methods,
        auto_detect=auto_detect,
        skipped_columns=skipped_columns,
    )


def _safe_float(value: Any, digits: int = 6) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if digits >= 0:
        return float(round(numeric, digits))
    return numeric


def _create_scaler(method: str):  # pragma: no cover - exercised indirectly in integration tests
    if method == "standard":
        return StandardScaler() if StandardScaler is not None else None
    if method == "minmax":
        return MinMaxScaler() if MinMaxScaler is not None else None
    if method == "maxabs":
        return MaxAbsScaler() if MaxAbsScaler is not None else None
    if method == "robust":
        return RobustScaler() if RobustScaler is not None else None
    return None


def _compute_scaling_stats(series: pd.Series) -> Optional[ScalingColumnStats]:
    numeric_series = pd.to_numeric(series, errors="coerce")
    valid = numeric_series.dropna()
    valid_count = int(valid.shape[0])

    if valid_count < 2:
        return None

    if _is_binary_numeric(valid):
        return None

    if valid.nunique(dropna=True) < 2:
        return None

    mean_value = _safe_float(valid.mean())
    median_value = _safe_float(valid.median())
    stddev_value = _safe_float(valid.std(ddof=0))
    minimum_value = _safe_float(valid.min())
    maximum_value = _safe_float(valid.max())

    q1 = _safe_float(valid.quantile(0.25))
    q3 = _safe_float(valid.quantile(0.75))
    if q1 is not None and q3 is not None:
        iqr_value = _safe_float(q3 - q1)
    else:
        iqr_value = None

    skewness_value = _safe_float(valid.skew())

    outlier_ratio = 0.0
    if iqr_value is not None and iqr_value > 0 and q1 is not None and q3 is not None:
        whisker = 1.5 * iqr_value
        lower_bound = q1 - whisker
        upper_bound = q3 + whisker
        mask = (valid < lower_bound) | (valid > upper_bound)
        outlier_count = int(mask.sum())
        if outlier_count and valid_count:
            outlier_ratio = float(round(outlier_count / valid_count, 6))

    return ScalingColumnStats(
        valid_count=valid_count,
        mean=mean_value,
        median=median_value,
        stddev=stddev_value,
        minimum=minimum_value,
        maximum=maximum_value,
        iqr=_safe_float(iqr_value) if iqr_value is not None else None,
        skewness=skewness_value,
        outlier_ratio=outlier_ratio if outlier_ratio > 0 else 0.0,
    )


def _recommend_scaling_method(
    stats: ScalingColumnStats,
) -> Tuple[ScalingMethodName, List[str], List[ScalingMethodName], ScalingConfidence]:
    minimum = stats.minimum
    maximum = stats.maximum
    stddev = stats.stddev or 0.0
    skewness = stats.skewness or 0.0
    outlier_ratio = stats.outlier_ratio or 0.0

    has_negative = minimum is not None and minimum < 0
    has_positive = maximum is not None and maximum > 0
    non_negative = minimum is not None and minimum >= 0

    range_width: Optional[float] = None
    if minimum is not None and maximum is not None:
        range_width = maximum - minimum

    reasons: List[str] = []
    method: ScalingMethodName = cast(ScalingMethodName, SCALING_DEFAULT_METHOD)
    confidence: ScalingConfidence = "medium"

    if outlier_ratio >= 0.1:
        method = cast(ScalingMethodName, "robust")
        confidence = "high" if outlier_ratio >= 0.2 else "medium"
        reasons.append(f"~{outlier_ratio * 100:.1f}% of rows fall outside 1.5xIQR")
    elif abs(skewness) >= 1.5:
        method = cast(ScalingMethodName, "robust")
        confidence = "medium"
    elif non_negative:
        method = cast(ScalingMethodName, "minmax")
        confidence = "medium"
        reasons.append("Feature is non-negative; Min-Max preserves bounds")
        if range_width is not None and maximum is not None and minimum is not None:
            if maximum <= 1 and minimum >= 0:
                confidence = "high"
                reasons.append("Already within [0, 1]; Min-Max maintains interpretability")
    elif has_negative and has_positive:
        near_zero_center = False
        if stddev > 0:
            mean = stats.mean or 0.0
            near_zero_center = abs(mean) <= max(stddev * 0.1, 1e-6)
        abs_max = None
        if minimum is not None and maximum is not None:
            abs_max = max(abs(minimum), abs(maximum))

        if near_zero_center and abs_max is not None and abs_max <= 1.5:
            method = cast(ScalingMethodName, "maxabs")
            confidence = "medium"
            reasons.append("Symmetric around zero with limited magnitude; MaxAbs preserves sparsity")
        else:
            method = cast(ScalingMethodName, "standard")
            confidence = "medium"
            reasons.append("Mixed-sign feature; Standard scaling recenters and normalizes variance")
    else:
        method = cast(ScalingMethodName, "standard")
        if stddev <= 1e-9:
            confidence = "low"
            reasons.append("Very low variance; scaling impact will be minimal")
        else:
            confidence = "medium"
            reasons.append("Default normalization for numeric feature")

    fallback_methods: List[ScalingMethodName] = [
        cast(ScalingMethodName, candidate)
        for candidate in SCALING_METHOD_ORDER
        if candidate != method
    ]

    return method, reasons, fallback_methods, confidence


def _prepare_scaling_context(
    frame: pd.DataFrame,
    column: str,
    config: NormalizedScalingConfig,
) -> Tuple[Optional[ScalingColumnContext], Optional[str]]:
    if column not in frame.columns:
        return None, "missing"

    numeric_series = pd.to_numeric(frame[column], errors="coerce")
    valid_mask = numeric_series.notna()
    if not valid_mask.any():
        return None, "no numeric data"

    valid_values = numeric_series.loc[valid_mask]
    if valid_values.nunique(dropna=True) < 2 or _is_binary_numeric(valid_values):
        return None, "insufficient variance"

    method = cast(ScalingMethodName, config.column_methods.get(column, config.default_method))
    if method not in SCALING_METHODS:
        method = cast(ScalingMethodName, SCALING_DEFAULT_METHOD)

    context = ScalingColumnContext(
        column=column,
        numeric_series=numeric_series,
        valid_mask=valid_mask,
        valid_values=valid_values,
        method=method,
    )
    return context, None


def _determine_fit_mode(env: ScalingEnvironment, stored_scaler: Optional[Any]) -> str:
    if env.train_mask is None:
        return "skip"
    if env.train_row_count > 0 and env.train_mask.any():
        return "fit"
    if stored_scaler is not None:
        return "reuse"
    return "skip"


def _scale_column(
    context: ScalingColumnContext,
    env: ScalingEnvironment,
    stored_scaler: Optional[Any],
    stored_metadata: Optional[Dict[str, Any]],
) -> ScalingProcessResult:
    if not env.can_persist:
        return _scale_without_storage(context)

    fit_mode = _determine_fit_mode(env, stored_scaler)
    if fit_mode == "skip":
        return ScalingProcessResult(skip_reason="no training data", signal_reason="no training data")
    if fit_mode == "fit":
        return _scale_with_fit(context, env)
    return _scale_with_reuse(context, env, stored_scaler, stored_metadata)


def _scale_without_storage(context: ScalingColumnContext) -> ScalingProcessResult:
    scaler = _create_scaler(context.method)
    if scaler is None:
        return ScalingProcessResult(skip_reason="scikit-learn unavailable")

    try:
        reshaped = context.valid_values.to_numpy().reshape(-1, 1)
        scaler.fit(reshaped)
        transformed = scaler.transform(reshaped).reshape(-1)
    except Exception:  # pragma: no cover - defensive guard
        return ScalingProcessResult(skip_reason="scaling failed")

    transformed_series = pd.Series(transformed, index=context.valid_values.index)
    return ScalingProcessResult(transformed_values=transformed_series, effective_method=context.method)


def _scale_with_fit(context: ScalingColumnContext, env: ScalingEnvironment) -> ScalingProcessResult:
    if env.storage is None or env.pipeline_id is None or env.node_id is None:
        return ScalingProcessResult(skip_reason="scaling failed")

    train_mask = env.train_mask
    if train_mask is None or not train_mask.any():
        return ScalingProcessResult(
            skip_reason="no training data to fit scaler",
            signal_reason="no training data",
        )

    train_numeric = context.numeric_series[train_mask]
    train_valid_values = train_numeric.dropna()
    if train_valid_values.nunique(dropna=True) < 2 or _is_binary_numeric(train_valid_values):
        return ScalingProcessResult(
            skip_reason="insufficient variance in training data",
            signal_reason="insufficient variance in training data",
        )

    scaler = _create_scaler(context.method)
    if scaler is None:
        return ScalingProcessResult(skip_reason="scikit-learn unavailable")

    try:
        train_array = train_valid_values.to_numpy().reshape(-1, 1)
        scaler.fit(train_array)
    except Exception as exc:
        truncated = str(exc)[:50]
        return ScalingProcessResult(
            skip_reason=f"failed to fit: {truncated}",
            signal_reason="fit failed",
        )

    metadata = _build_scaler_metadata(scaler, context.method, env.train_row_count, len(train_valid_values))

    env.storage.store_transformer(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name="scaler",
        transformer=scaler,
        column_name=context.column,
        metadata=metadata,
    )

    env.storage.record_split_activity(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name="scaler",
        column_name=context.column,
        split_name="train",
        action="fit_transform",
        row_count=env.train_row_count,
    )

    try:
        all_valid_values = context.numeric_series.loc[context.valid_mask]
        reshaped = all_valid_values.to_numpy().reshape(-1, 1)
        transformed = scaler.transform(reshaped).reshape(-1)
    except Exception as exc:
        truncated = str(exc)[:50]
        return ScalingProcessResult(
            skip_reason=f"failed to transform: {truncated}",
            signal_reason="transform failed",
        )

    for split_name in ("test", "validation"):
        rows_processed = int(env.split_counts.get(split_name, 0))
        env.storage.record_split_activity(
            pipeline_id=env.pipeline_id,
            node_id=env.node_id,
            transformer_name="scaler",
            column_name=context.column,
            split_name=split_name,
            action="transform" if rows_processed > 0 else "not_available",
            row_count=rows_processed,
        )

    transformed_series = pd.Series(transformed, index=context.numeric_series.loc[context.valid_mask].index)
    return ScalingProcessResult(transformed_values=transformed_series, effective_method=context.method)


def _scale_with_reuse(
    context: ScalingColumnContext,
    env: ScalingEnvironment,
    stored_scaler: Optional[Any],
    stored_metadata: Optional[Dict[str, Any]],
) -> ScalingProcessResult:
    if env.storage is None or env.pipeline_id is None or env.node_id is None:
        return ScalingProcessResult(skip_reason="scaling failed")

    if stored_scaler is None or not hasattr(stored_scaler, "transform"):
        return ScalingProcessResult(skip_reason="stored scaler unavailable")

    effective_method: ScalingMethodName = context.method
    if isinstance(stored_metadata, dict):
        stored_method = stored_metadata.get("method")
        if stored_method in SCALING_METHODS:
            effective_method = cast(ScalingMethodName, stored_method)

    try:
        all_valid_values = context.numeric_series.loc[context.valid_mask]
        reshaped = all_valid_values.to_numpy().reshape(-1, 1)
        transformed = stored_scaler.transform(reshaped).reshape(-1)
    except Exception as exc:
        truncated = str(exc)[:50]
        return ScalingProcessResult(
            skip_reason=f"failed to transform: {truncated}",
            signal_reason="transform failed",
        )

    env.storage.record_split_activity(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name="scaler",
        column_name=context.column,
        split_name="train",
        action="not_available",
        row_count=env.train_row_count,
    )

    for split_name in ("test", "validation"):
        rows_processed = int(env.split_counts.get(split_name, 0))
        env.storage.record_split_activity(
            pipeline_id=env.pipeline_id,
            node_id=env.node_id,
            transformer_name="scaler",
            column_name=context.column,
            split_name=split_name,
            action="transform" if rows_processed > 0 else "not_available",
            row_count=rows_processed,
        )

    transformed_series = pd.Series(transformed, index=context.numeric_series.loc[context.valid_mask].index)
    return ScalingProcessResult(transformed_values=transformed_series, effective_method=effective_method)


def _build_scaler_metadata(
    scaler: Any,
    method: ScalingMethodName,
    train_rows: int,
    train_valid_rows: int,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "method": method,
        "method_label": SCALING_METHODS.get(method, {}).get("label", method),
        "train_rows": train_rows,
        "train_valid_rows": train_valid_rows,
    }

    try:
        fitted: Dict[str, Any] = {}
        if method == "standard":
            if hasattr(scaler, "mean_"):
                mean_value = scaler.mean_
                fitted["mean"] = mean_value.tolist() if hasattr(mean_value, "tolist") else float(mean_value)
            if hasattr(scaler, "scale_"):
                std_value = scaler.scale_
                fitted["std"] = std_value.tolist() if hasattr(std_value, "tolist") else float(std_value)
        elif method == "minmax":
            if hasattr(scaler, "data_min_"):
                data_min = scaler.data_min_
                fitted["min"] = data_min.tolist() if hasattr(data_min, "tolist") else float(data_min)
            if hasattr(scaler, "data_max_"):
                data_max = scaler.data_max_
                fitted["max"] = data_max.tolist() if hasattr(data_max, "tolist") else float(data_max)
        elif method == "robust":
            if hasattr(scaler, "center_"):
                center = scaler.center_
                fitted["median"] = center.tolist() if hasattr(center, "tolist") else float(center)
            if hasattr(scaler, "scale_"):
                scale_value = scaler.scale_
                fitted["iqr"] = scale_value.tolist() if hasattr(scale_value, "tolist") else float(scale_value)
        if fitted:
            metadata["fitted_params"] = fitted
    except Exception:
        logger.exception("Failed to extract fitted params into metadata for scaler")

    return metadata


def _append_skip(
    skipped: List[str],
    signal: ScalingNodeSignal,
    column: str,
    summary_reason: str,
    signal_reason: Optional[str] = None,
) -> None:
    summary_message = f"{column} ({summary_reason})"
    skipped.append(summary_message)
    effective_signal_reason = signal_reason or summary_reason
    signal_message = f"{column} ({effective_signal_reason})"
    signal.skipped_columns.append(signal_message)


def _resolve_candidate_columns(
    frame: pd.DataFrame,
    config: NormalizedScalingConfig,
) -> Tuple[List[str], List[str]]:
    skipped_configured_columns = {str(column).strip() for column in config.skipped_columns if str(column).strip()}
    config_skipped: List[str] = []
    candidate_columns: List[str] = []
    seen: set[str] = set()

    for column in config.columns:
        normalized = str(column or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if normalized in skipped_configured_columns:
            config_skipped.append(normalized)
            continue
        candidate_columns.append(normalized)

    if config.auto_detect:
        for column in _detect_numeric_columns(frame):
            if column in seen:
                continue
            seen.add(column)
            if column in skipped_configured_columns:
                config_skipped.append(column)
                continue
            candidate_columns.append(column)

    return candidate_columns, config_skipped


def _build_scaling_environment(
    frame: pd.DataFrame,
    pipeline_id: Optional[str],
    node_id: Optional[str],
) -> ScalingEnvironment:
    has_splits = SPLIT_TYPE_COLUMN in frame.columns
    if not has_splits:
        return ScalingEnvironment(
            has_splits=False,
            storage=None,
            pipeline_id=None,
            node_id=node_id,
            split_counts={},
            train_mask=None,
            train_row_count=0,
        )

    storage = get_pipeline_store() if pipeline_id else None
    split_counts = frame[SPLIT_TYPE_COLUMN].value_counts().to_dict()
    train_mask = frame[SPLIT_TYPE_COLUMN] == "train"
    train_row_count = int(split_counts.get("train", 0))

    return ScalingEnvironment(
        has_splits=True,
        storage=storage,
        pipeline_id=pipeline_id if storage is not None else None,
        node_id=node_id,
        split_counts=split_counts,
        train_mask=train_mask,
        train_row_count=train_row_count,
    )


def _fetch_stored_scaler(
    env: ScalingEnvironment,
    column: str,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if env.storage is None or env.pipeline_id is None or env.node_id is None:
        return None, None

    stored_scaler = env.storage.get_transformer(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name="scaler",
        column_name=column,
    )
    stored_metadata = env.storage.get_metadata(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name="scaler",
        column_name=column,
    )
    return stored_scaler, stored_metadata


def _apply_scale_numeric_features(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, ScalingNodeSignal]:
    """Apply scaling to numeric features.

    Supports train/test/validation split awareness:
    - On training data: fit scaler and store for reuse
    - On test/val data: retrieve stored scaler and transform
    - Without split: fit and transform (no storage)

    Args:
        frame: Input DataFrame
        node: Node configuration dictionary
        pipeline_id: Unique pipeline identifier for transformer storage

    Returns:
        Tuple of (transformed_frame, summary_message, signal_metadata)
    """
    node_id = node.get("id") if isinstance(node, dict) else None
    node_id_str = str(node_id) if node_id is not None else None

    signal = ScalingNodeSignal(node_id=node_id_str)

    if frame.empty:
        return frame, "Scaling: no data available", signal

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_scaling_config(config_payload)

    configured_columns = [str(column).strip() for column in config.columns if str(column).strip()]
    signal.configured_columns = list(dict.fromkeys(configured_columns))
    signal.auto_detect = config.auto_detect
    signal.default_method = cast(ScalingMethodName, config.default_method)
    signal.column_methods = {
        str(column).strip(): cast(ScalingMethodName, method)
        for column, method in config.column_methods.items()
        if str(column).strip()
    }

    candidate_columns, config_skipped = _resolve_candidate_columns(frame, config)

    base_skipped_columns = sorted(set(config_skipped))
    base_skip_messages = [f"{column} (skipped)" for column in base_skipped_columns]

    if not candidate_columns:
        if base_skip_messages:
            signal.skipped_columns.extend(base_skip_messages)
        signal.evaluated_columns = []
        return frame, "Scaling: no numeric columns selected", signal

    signal.evaluated_columns = list(candidate_columns)

    working_frame = frame.copy()
    transformed: List[str] = []
    skipped: List[str] = list(base_skip_messages)
    if base_skip_messages:
        signal.skipped_columns.extend(base_skip_messages)

    env = _build_scaling_environment(working_frame, pipeline_id, node_id_str)

    skip_message_map = {
        "missing": "missing",
        "no numeric data": "no numeric data",
        "insufficient variance": "insufficient variance",
    }

    for column in candidate_columns:
        context, failure_reason = _prepare_scaling_context(working_frame, column, config)
        if context is None:
            suffix = skip_message_map.get(failure_reason or "", failure_reason or "unavailable")
            _append_skip(skipped, signal, column, suffix)
            continue

        stored_scaler, stored_metadata = _fetch_stored_scaler(env, context.column)

        result = _scale_column(context, env, stored_scaler, stored_metadata)
        if result.transformed_values is None:
            summary_suffix = result.skip_reason or "scaling failed"
            signal_suffix = result.signal_reason or summary_suffix
            _append_skip(skipped, signal, column, summary_suffix, signal_suffix)
            continue

        updated_series = context.numeric_series.copy()
        # Ensure integer-backed columns accept floating-point scaling results
        if not pd.api.types.is_float_dtype(updated_series.dtype):
            target_dtype = (
                result.transformed_values.dtype
                if pd.api.types.is_float_dtype(result.transformed_values.dtype)
                else "float64"
            )
            updated_series = updated_series.astype(target_dtype)
        updated_series.loc[result.transformed_values.index] = result.transformed_values
        working_frame[column] = updated_series

        effective_method = result.effective_method or context.method
        label = SCALING_METHODS.get(effective_method, {}).get("label", effective_method)
        transformed.append(f"{column} ({label})")

        transformed_series = result.transformed_values
        total_rows = int(context.numeric_series.shape[0])
        valid_rows = int(context.valid_mask.sum())
        missing_rows = total_rows - valid_rows

        signal.scaled_columns.append(
            ScalingAppliedColumnSignal(
                column=column,
                method=cast(ScalingMethodName, effective_method),
                method_label=str(label) if label else None,
                total_rows=total_rows,
                valid_rows=valid_rows,
                missing_rows=missing_rows,
                original_mean=_safe_float(context.valid_values.mean()),
                original_stddev=_safe_float(context.valid_values.std(ddof=0)),
                original_min=_safe_float(context.valid_values.min()),
                original_max=_safe_float(context.valid_values.max()),
                scaled_mean=_safe_float(transformed_series.mean()),
                scaled_stddev=_safe_float(transformed_series.std(ddof=0)),
                scaled_min=_safe_float(transformed_series.min()),
                scaled_max=_safe_float(transformed_series.max()),
            )
        )

    if not transformed:
        summary = "Scaling: no columns transformed"
    else:
        preview = ", ".join(transformed[:4])
        if len(transformed) > 4:
            preview = f"{preview}, ..."
        summary = f"Scaling: transformed {len(transformed)} column(s) ({preview})"

    if skipped:
        preview = ", ".join(skipped[:3])
        if len(skipped) > 3:
            preview = f"{preview}, ..."
        summary = f"{summary}; skipped {preview}"

    return working_frame, summary, signal


def _build_scaling_recommendations(frame: pd.DataFrame) -> List[ScalingColumnRecommendation]:
    recommendations: List[ScalingColumnRecommendation] = []

    for column in _detect_numeric_columns(frame):
        if column not in frame.columns:
            continue

        series = frame[column]
        stats = _compute_scaling_stats(series)
        if not stats:
            continue

        method, reasons, fallbacks, confidence = _recommend_scaling_method(stats)
        recommendations.append(
            ScalingColumnRecommendation(
                column=column,
                dtype=str(series.dtype),
                recommended_method=method,
                confidence=confidence,
                reasons=reasons,
                fallback_methods=fallbacks,
                stats=stats,
                has_missing=bool(series.isna().any()),
            )
        )

    recommendations.sort(key=lambda item: item.column.lower())
    return recommendations


def _scaling_method_details() -> List[ScalingMethodDetail]:
    details: List[ScalingMethodDetail] = []
    for key in SCALING_METHOD_ORDER:
        meta = SCALING_METHODS.get(key)
        if not meta:
            continue
        method_key = cast(ScalingMethodName, key)
        details.append(
            ScalingMethodDetail(
                key=method_key,
                label=str(meta.get("label") or method_key.title()),
                description=meta.get("description"),
                handles_negative=bool(meta.get("handles_negative", True)),
                handles_zero=bool(meta.get("handles_zero", True)),
                handles_outliers=bool(meta.get("handles_outliers", False)),
                strengths=list(meta.get("strengths") or []),
                cautions=list(meta.get("cautions") or []),
            )
        )
    return details


__all__ = [
    "SCALING_METHODS",
    "SCALING_METHOD_ORDER",
    "SCALING_DEFAULT_METHOD",
    "NormalizedScalingConfig",
    "_apply_scale_numeric_features",
    "_build_scaling_recommendations",
    "_scaling_method_details",
]

