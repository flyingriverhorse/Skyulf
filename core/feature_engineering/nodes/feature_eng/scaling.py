"""Scaling helpers for feature engineering nodes."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

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
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - defensive guard
    StandardScaler = None  # type: ignore[assignment]
    MinMaxScaler = None  # type: ignore[assignment]
    MaxAbsScaler = None  # type: ignore[assignment]
    RobustScaler = None  # type: ignore[assignment]

from core.feature_engineering.transformer_storage import get_transformer_storage
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

from .binning import _detect_numeric_columns, _is_binary_numeric
from .utils import _coerce_config_boolean

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


@dataclass
class NormalizedScalingConfig:
    columns: List[str]
    default_method: str
    column_methods: Dict[str, str]
    auto_detect: bool
    skipped_columns: List[str]


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
) -> Tuple[ScalingMethodName, List[str], List[ScalingMethodName], str]:
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
    confidence: str = "medium"

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
            reasons.append("Default normalization for numeric feature")

    fallback_methods: List[ScalingMethodName] = [
        cast(ScalingMethodName, candidate)
        for candidate in SCALING_METHOD_ORDER
        if candidate != method
    ]

    return method, reasons, fallback_methods, confidence


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

    signal = ScalingNodeSignal(node_id=str(node_id) if node_id is not None else None)

    if frame.empty:
        return frame, "Scaling: no data available", signal

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_scaling_config(config_payload)

    configured_columns = [str(column).strip() for column in config.columns if str(column).strip()]
    signal.configured_columns = list(dict.fromkeys(configured_columns))
    signal.auto_detect = config.auto_detect
    signal.default_method = cast(ScalingMethodName, config.default_method)
    # Ensure column method keys are unique and normalized.
    signal.column_methods = {
        str(column).strip(): cast(ScalingMethodName, method)
        for column, method in config.column_methods.items()
        if str(column).strip()
    }

    skipped_configured_columns = {str(column).strip() for column in config.skipped_columns if str(column).strip()}
    config_skipped: List[str] = []
    candidate_columns: List[str] = []
    seen: set[str] = set()

    for column in config.columns:
        normalized = str(column or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            if normalized in skipped_configured_columns:
                config_skipped.append(normalized)
                continue
            candidate_columns.append(normalized)

    if config.auto_detect:
        for column in _detect_numeric_columns(frame):
            if column in skipped_configured_columns:
                config_skipped.append(column)
                seen.add(column)
                continue
            if column not in seen:
                candidate_columns.append(column)
                seen.add(column)

    if not candidate_columns:
        if config_skipped:
            signal.skipped_columns.extend(f"{column} (skipped)" for column in sorted(set(config_skipped)))
        signal.evaluated_columns = []
        return frame, "Scaling: no numeric columns selected", signal

    signal.evaluated_columns = list(candidate_columns)

    working_frame = frame.copy()
    transformed: List[str] = []
    skipped: List[str] = []
    if config_skipped:
        skipped.extend(f"{column} (skipped)" for column in sorted(set(config_skipped)))
        signal.skipped_columns.extend(f"{column} (skipped)" for column in sorted(set(config_skipped)))

    # Check if we have split column for train/test/validation awareness
    has_splits = SPLIT_TYPE_COLUMN in working_frame.columns
    storage = get_transformer_storage() if pipeline_id and has_splits else None

    split_counts: Dict[str, int] = {}
    train_mask: Optional[pd.Series] = None
    train_row_count = 0
    if has_splits:
        split_counts = working_frame[SPLIT_TYPE_COLUMN].value_counts().to_dict()
        train_row_count = int(split_counts.get("train", 0))
        train_mask = working_frame[SPLIT_TYPE_COLUMN] == "train"

    for column in candidate_columns:
        if column not in working_frame.columns:
            skipped.append(f"{column} (missing)")
            signal.skipped_columns.append(f"{column} (missing)")
            continue

        numeric_series = pd.to_numeric(working_frame[column], errors="coerce")
        valid_mask = numeric_series.notna()
        if not valid_mask.any():
            skipped.append(f"{column} (no numeric data)")
            signal.skipped_columns.append(f"{column} (no numeric data)")
            continue

        valid_values = numeric_series.loc[valid_mask]
        if valid_values.nunique(dropna=True) < 2 or _is_binary_numeric(valid_values):
            skipped.append(f"{column} (insufficient variance)")
            signal.skipped_columns.append(f"{column} (insufficient variance)")
            continue

        method = config.column_methods.get(column, config.default_method)
        if method not in SCALING_METHODS:
            method = SCALING_DEFAULT_METHOD

        # Check for stored transformer
        stored_scaler: Optional[Any] = None
        stored_metadata: Optional[Dict[str, Any]] = None
        if storage is not None:
            stored_scaler = storage.get_transformer(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="scaler",
                column_name=column,
            )
            stored_metadata = storage.get_metadata(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="scaler",
                column_name=column,
            )

        # Determine fit mode
        fit_mode = "fit"
        if storage is not None and train_mask is not None:
            if train_row_count <= 0:
                if stored_scaler is not None:
                    fit_mode = "reuse"
                else:
                    skipped.append(f"{column} (no training data)")
                    signal.skipped_columns.append(f"{column} (no training data)")
                    continue

        scaler: Any
        effective_method = method

        # If we have splits AND pipeline_id (actual execution), process train/test/val with fit/transform
        if has_splits and storage and pipeline_id and fit_mode == "fit":
            logger.debug(
                "Applying split-aware scaling",
                extra={"node_id": node_id, "column": column, "method": method},
            )
            
            if train_mask is None or not train_mask.any():
                skipped.append(f"{column} (no training data to fit scaler)")
                signal.skipped_columns.append(f"{column} (no training data)")
                continue

            # Fit on training data only
            train_numeric = numeric_series[train_mask]
            train_valid_mask = train_numeric.notna()
            train_valid_values = train_numeric.loc[train_valid_mask]

            if train_valid_values.nunique(dropna=True) < 2 or _is_binary_numeric(train_valid_values):
                skipped.append(f"{column} (insufficient variance in training data)")
                signal.skipped_columns.append(f"{column} (insufficient variance in training data)")
                continue

            scaler = _create_scaler(method)
            if scaler is None:
                skipped.append(f"{column} (scikit-learn unavailable)")
                signal.skipped_columns.append(f"{column} (scikit-learn unavailable)")
                continue

            try:
                train_reshaped = train_valid_values.to_numpy().reshape(-1, 1)
                scaler.fit(train_reshaped)
            except Exception as exc:
                skipped.append(f"{column} (failed to fit: {str(exc)[:50]})")
                signal.skipped_columns.append(f"{column} (fit failed)")
                continue

            # Store the fitted scaler
            scaler_metadata = {
                "method": method,
                "method_label": SCALING_METHODS.get(method, {}).get("label", method),
                "train_rows": int(train_mask.sum()),
                "train_valid_rows": len(train_valid_values),
            }

            # Persist fitted parameters into metadata as a fallback so exports
            # can include them even if the transformer object isn't available later.
            try:
                fitted = {}
                if method == "standard":
                    if hasattr(scaler, "mean_"):
                        fitted["mean"] = scaler.mean_.tolist() if hasattr(scaler.mean_, "tolist") else float(scaler.mean_)
                    if hasattr(scaler, "scale_"):
                        fitted["std"] = scaler.scale_.tolist() if hasattr(scaler.scale_, "tolist") else float(scaler.scale_)
                elif method == "minmax":
                    if hasattr(scaler, "data_min_"):
                        fitted["min"] = scaler.data_min_.tolist() if hasattr(scaler.data_min_, "tolist") else float(scaler.data_min_)
                    if hasattr(scaler, "data_max_"):
                        fitted["max"] = scaler.data_max_.tolist() if hasattr(scaler.data_max_, "tolist") else float(scaler.data_max_)
                elif method == "robust":
                    if hasattr(scaler, "center_"):
                        fitted["median"] = scaler.center_.tolist() if hasattr(scaler.center_, "tolist") else float(scaler.center_)
                    if hasattr(scaler, "scale_"):
                        fitted["iqr"] = scaler.scale_.tolist() if hasattr(scaler.scale_, "tolist") else float(scaler.scale_)
                if fitted:
                    scaler_metadata["fitted_params"] = fitted
            except Exception:
                logger.exception("Failed to extract fitted params into metadata for scaler")

            storage.store_transformer(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="scaler",
                transformer=scaler,
                column_name=column,
                metadata=scaler_metadata,
            )

            storage.record_split_activity(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="scaler",
                column_name=column,
                split_name="train",
                action="fit_transform",
                row_count=train_row_count,
            )

            # Transform ALL data (train, test, validation) using fitted scaler
            try:
                all_valid_values = numeric_series.loc[valid_mask]
                reshaped = all_valid_values.to_numpy().reshape(-1, 1)
                transformed_values = scaler.transform(reshaped).reshape(-1)

                for split_name in ("test", "validation"):
                    rows_processed = int(split_counts.get(split_name, 0))
                    storage.record_split_activity(
                        pipeline_id=pipeline_id,
                        node_id=str(node_id),
                        transformer_name="scaler",
                        column_name=column,
                        split_name=split_name,
                        action="transform" if rows_processed > 0 else "not_available",
                        row_count=rows_processed,
                    )
            except Exception as exc:
                skipped.append(f"{column} (failed to transform: {str(exc)[:50]})")
                signal.skipped_columns.append(f"{column} (transform failed)")
                continue

        elif has_splits and storage and pipeline_id and fit_mode == "reuse":
            # Reuse stored scaler (no training data, but test/val data exists)
            logger.debug(
                "Reusing stored scaler",
                extra={"node_id": node_id, "column": column},
            )

            if stored_scaler is None or not hasattr(stored_scaler, "transform"):
                skipped.append(f"{column} (stored scaler unavailable)")
                signal.skipped_columns.append(f"{column} (stored scaler unavailable)")
                continue

            scaler = stored_scaler
            
            # Get method from metadata if available
            if isinstance(stored_metadata, dict):
                stored_method = stored_metadata.get("method")
                if stored_method and stored_method in SCALING_METHODS:
                    effective_method = stored_method

            try:
                all_valid_values = numeric_series.loc[valid_mask]
                reshaped = all_valid_values.to_numpy().reshape(-1, 1)
                transformed_values = scaler.transform(reshaped).reshape(-1)
            except Exception as exc:
                skipped.append(f"{column} (failed to transform: {str(exc)[:50]})")
                signal.skipped_columns.append(f"{column} (transform failed)")
                continue

            storage.record_split_activity(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name="scaler",
                column_name=column,
                split_name="train",
                action="not_available",
                row_count=train_row_count,
            )

            for split_name in ("test", "validation"):
                rows_processed = int(split_counts.get(split_name, 0))
                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="scaler",
                    column_name=column,
                    split_name=split_name,
                    action="transform" if rows_processed > 0 else "not_available",
                    row_count=rows_processed,
                )

        else:
            # NO SPLITS: Standard fit_transform (or splits without pipeline_id for recommendations)
            logger.debug(
                "Applying standard scaling",
                extra={
                    "node_id": node_id,
                    "column": column,
                    "method": method,
                    "has_splits": has_splits,
                    "pipeline_id": pipeline_id,
                },
            )
            
            scaler = _create_scaler(method)
            if scaler is None:
                skipped.append(f"{column} (scikit-learn unavailable)")
                signal.skipped_columns.append(f"{column} (scikit-learn unavailable)")
                continue

            try:
                reshaped = valid_values.to_numpy().reshape(-1, 1)
                scaler.fit(reshaped)
                transformed_values = scaler.transform(reshaped).reshape(-1)
            except Exception:  # pragma: no cover - defensive guard
                skipped.append(f"{column} (scaling failed)")
                signal.skipped_columns.append(f"{column} (scaling failed)")
                continue

        # Apply transformed values back to the dataframe
        updated_series = numeric_series.copy()
        updated_series.loc[valid_mask] = transformed_values
        working_frame[column] = updated_series
        
        label = SCALING_METHODS.get(effective_method, {}).get("label", effective_method)
        transformed.append(f"{column} ({label})")

        transformed_series = pd.Series(transformed_values, index=valid_values.index)
        total_rows = int(numeric_series.shape[0])
        valid_rows = int(valid_mask.sum())
        missing_rows = total_rows - valid_rows

        signal.scaled_columns.append(
            ScalingAppliedColumnSignal(
                column=column,
                method=cast(ScalingMethodName, effective_method),
                method_label=str(label) if label else None,
                total_rows=total_rows,
                valid_rows=valid_rows,
                missing_rows=missing_rows,
                original_mean=_safe_float(valid_values.mean()),
                original_stddev=_safe_float(valid_values.std(ddof=0)),
                original_min=_safe_float(valid_values.min()),
                original_max=_safe_float(valid_values.max()),
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
