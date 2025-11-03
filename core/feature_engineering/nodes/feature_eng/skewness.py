"""Skewness transformation helpers for feature engineering nodes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency guard
    from sklearn.preprocessing import PowerTransformer  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - defensive guard
    PowerTransformer = None  # type: ignore[assignment]

from core.feature_engineering.schemas import (
    SkewnessColumnDistribution,
    SkewnessColumnRecommendation,
    SkewnessMethodDetail,
    SkewnessMethodStatus,
    SkewnessAppliedColumnSignal,
    SkewnessConfiguredTransformation,
    SkewnessNodeSignal,
    SkewnessSkippedColumnSignal,
)

from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

from .binning import _is_binary_numeric

SKEWNESS_METHODS: Dict[str, Dict[str, Any]] = {
    "log": {
        "label": "Logarithmic",
        "description": "Applies natural log (log1p) to compress long right tails.",
        "direction_bias": "right",
        "requires_positive": True,
        "supports_zero": False,
        "supports_negative": False,
    },
    "square_root": {
        "label": "Square Root",
        "description": "Reduces moderate right skew for non-negative values.",
        "direction_bias": "right",
        "requires_positive": False,
        "supports_zero": True,
        "supports_negative": False,
    },
    "cube_root": {
        "label": "Cube Root",
        "description": "Gentle re-scaling that works for both positive and negative values.",
        "direction_bias": "either",
        "requires_positive": False,
        "supports_zero": True,
        "supports_negative": True,
    },
    "reciprocal": {
        "label": "Reciprocal (Inverse)",
        "description": "Flips right-skewed distributions; undefined for zero values.",
        "direction_bias": "right",
        "requires_positive": False,
        "supports_zero": False,
        "supports_negative": True,
    },
    "square": {
        "label": "Square",
        "description": "Expands higher values to correct left skew.",
        "direction_bias": "left",
        "requires_positive": False,
        "supports_zero": True,
        "supports_negative": True,
    },
    "exponential": {
        "label": "Exponential",
        "description": "Strongly stretches differences in higher values to address left skew.",
        "direction_bias": "left",
        "requires_positive": False,
        "supports_zero": True,
        "supports_negative": True,
    },
    "box_cox": {
        "label": "Box-Cox",
        "description": "Power transform for strictly positive data; learns lambda automatically.",
        "direction_bias": "right",
        "requires_positive": True,
        "supports_zero": False,
        "supports_negative": False,
    },
    "yeo_johnson": {
        "label": "Yeo-Johnson",
        "description": "Power transform that supports zero and negative values.",
        "direction_bias": "either",
        "requires_positive": False,
        "supports_zero": True,
        "supports_negative": True,
    },
}

SKEWNESS_METHOD_ORDER: Tuple[str, ...] = (
    "log",
    "square_root",
    "cube_root",
    "reciprocal",
    "square",
    "exponential",
    "box_cox",
    "yeo_johnson",
)

SKEWNESS_THRESHOLD = 0.75


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return float(numeric)


def _build_skewness_distribution(
    values: pd.Series,
    *,
    missing_count: int,
) -> Optional[SkewnessColumnDistribution]:
    valid_values = pd.to_numeric(values, errors="coerce").dropna()
    valid_count = int(valid_values.size)
    if valid_count <= 0:
        return None

    try:
        bin_count = min(30, max(6, int(math.ceil(math.sqrt(valid_count)))))
        histogram_counts, bin_edges = np.histogram(valid_values.to_numpy(), bins=bin_count)
    except Exception:  # pragma: no cover - defensive
        return None

    if not histogram_counts.size or not bin_edges.size:
        return None

    return SkewnessColumnDistribution(
        bin_edges=[float(edge) for edge in bin_edges.tolist()],
        counts=[int(count) for count in histogram_counts.tolist()],
        sample_size=valid_count,
        missing_count=max(missing_count, 0),
        minimum=_finite_or_none(valid_values.min()),
        maximum=_finite_or_none(valid_values.max()),
        mean=_finite_or_none(valid_values.mean()),
        median=_finite_or_none(valid_values.median()),
        stddev=_finite_or_none(valid_values.std(ddof=0)),
    )


def _skewness_direction(skew_value: float) -> str:
    return "right" if skew_value >= 0 else "left"


def _skewness_magnitude(skew_value: float) -> str:
    absolute = abs(skew_value)
    if absolute < 1.0:
        return "moderate"
    if absolute < 2.0:
        return "substantial"
    return "extreme"


def _evaluate_skewness_method_status(series: pd.Series) -> Dict[str, SkewnessMethodStatus]:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return {key: SkewnessMethodStatus(status="unsupported", reason="No numeric data") for key in SKEWNESS_METHODS}

    has_negative = bool((cleaned < 0).any())
    has_zero = bool((cleaned == 0).any())
    has_non_positive = bool((cleaned <= 0).any())
    has_near_zero = bool((cleaned.abs() < 1e-12).any())

    statuses: Dict[str, SkewnessMethodStatus] = {}
    for key, meta in SKEWNESS_METHODS.items():
        requires_positive = bool(meta.get("requires_positive"))
        supports_zero = bool(meta.get("supports_zero", True))
        supports_negative = bool(meta.get("supports_negative", True))

        status = "ready"
        reason: Optional[str] = None

        if requires_positive and has_non_positive:
            status = "unsupported"
            reason = "Requires strictly positive values"
        elif not supports_negative and has_negative:
            status = "unsupported"
            reason = "Not defined for negative values"

        if status == "ready" and not supports_zero and has_zero:
            status = "unsupported"
            reason = "Not defined for zero values"

        if status == "ready" and key == "reciprocal" and (has_zero or has_near_zero):
            status = "unsupported"
            reason = "Reciprocal undefined for zero or near-zero values"

        statuses[key] = SkewnessMethodStatus(status=status, reason=reason)

    return statuses


def _recommended_methods_for_direction(
    direction: str,
    statuses: Dict[str, SkewnessMethodStatus],
) -> List[str]:
    if direction == "right":
        candidate_order = ("log", "square_root", "cube_root", "box_cox", "yeo_johnson", "reciprocal")
    else:
        candidate_order = ("square", "exponential", "yeo_johnson", "cube_root")

    recommended: List[str] = [
        method
        for method in candidate_order
        if statuses.get(method) and statuses[method].status == "ready"
    ]

    if not recommended and statuses.get("yeo_johnson") and statuses["yeo_johnson"].status == "ready":
        recommended = ["yeo_johnson"]

    return recommended[:4]


def _build_skewness_recommendations(
    frame: pd.DataFrame,
    selected_methods: Optional[Dict[str, str]] = None,
    applied_methods: Optional[Dict[str, str]] = None,
) -> List[SkewnessColumnRecommendation]:
    if frame.empty:
        return []

    recommendations: List[SkewnessColumnRecommendation] = []

    numeric_frame = frame.select_dtypes(include=[np.number])

    normalized_selected: Dict[str, str] = {}
    normalized_applied: Dict[str, str] = {}
    if selected_methods:
        for column, method in selected_methods.items():
            column_key = str(column).strip()
            method_key = str(method).strip().lower()
            if column_key and method_key in SKEWNESS_METHODS:
                normalized_selected[column_key] = method_key
    if applied_methods:
        for column, method in applied_methods.items():
            column_key = str(column).strip()
            method_key = str(method).strip().lower()
            if column_key and method_key in SKEWNESS_METHODS:
                normalized_applied[column_key] = method_key

    for column_name in numeric_frame.columns:
        if column_name in frame.columns:
            original_series = frame[column_name]
            if pd.api.types.is_bool_dtype(original_series):
                continue
        numeric_series = pd.to_numeric(numeric_frame[column_name], errors="coerce")
        series = numeric_series.dropna()
        if series.size < 3:
            continue

        if _is_binary_numeric(series):
            continue

        skew_value = series.skew()
        if not math.isfinite(skew_value):
            continue

        if abs(skew_value) < SKEWNESS_THRESHOLD:
            continue

        direction = _skewness_direction(skew_value)
        magnitude = _skewness_magnitude(skew_value)
        statuses = _evaluate_skewness_method_status(series)
        recommended_methods = _recommended_methods_for_direction(direction, statuses)

        direction_label = "Right-skewed" if direction == "right" else "Left-skewed"
        if recommended_methods:
            label_list = ", ".join(SKEWNESS_METHODS[method]["label"] for method in recommended_methods)
            summary = f"{direction_label} ({skew_value:.2f}). Try {label_list}."
        else:
            summary = f"{direction_label} ({skew_value:.2f}). Review before selecting a transform."

        method_status_map: Dict[str, SkewnessMethodStatus] = {}
        for method in SKEWNESS_METHOD_ORDER:
            method_status_map[method] = statuses.get(method, SkewnessMethodStatus(status="unsupported", reason="Unavailable"))

        valid_count = int(series.size)
        total_count = int(numeric_series.size)
        missing_count = max(total_count - valid_count, 0)

        distribution_before = _build_skewness_distribution(series, missing_count=missing_count)

        column_key = str(column_name)
        selected_method = normalized_selected.get(column_key)
        applied_method = normalized_applied.get(column_key)
        transform_method = selected_method or applied_method
        distribution_after: Optional[SkewnessColumnDistribution] = None
        if transform_method:
            try:
                transformed_series, reason = _perform_skewness_transform(series, transform_method)
                if transformed_series is not None and not transformed_series.dropna().empty:
                    distribution_after = _build_skewness_distribution(
                        transformed_series,
                        missing_count=missing_count,
                    )
            except Exception:  # pragma: no cover - defensive
                distribution_after = None

        recommendations.append(
            SkewnessColumnRecommendation(
                column=str(column_name),
                skewness=float(round(skew_value, 6)),
                direction=direction,
                magnitude=magnitude,
                summary=summary,
                recommended_methods=recommended_methods,
                method_status=method_status_map,
                distribution_before=distribution_before,
                distribution_after=distribution_after,
                applied_method=applied_method,
            )
        )

    recommendations.sort(key=lambda item: abs(item.skewness), reverse=True)
    return recommendations


def _skewness_method_details() -> List[SkewnessMethodDetail]:
    details: List[SkewnessMethodDetail] = []
    for key in SKEWNESS_METHOD_ORDER:
        meta = SKEWNESS_METHODS.get(key)
        if not meta:
            continue
        details.append(
            SkewnessMethodDetail(
                key=key,
                label=meta["label"],
                description=meta.get("description"),
                direction_bias=meta.get("direction_bias"),
                requires_positive=bool(meta.get("requires_positive")),
                supports_zero=bool(meta.get("supports_zero", True)),
                supports_negative=bool(meta.get("supports_negative", True)),
            )
        )
    return details


def _perform_skewness_transform(series: pd.Series, method: str) -> Tuple[Optional[pd.Series], Optional[str]]:
    if series.empty:
        return None, "No numeric data"

    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None, "No numeric data"

    if method == "log":
        if (values <= 0).any():
            return None, "Requires positive values"
        return pd.Series(np.log1p(values), index=values.index), None

    if method == "square_root":
        if (values < 0).any():
            return None, "Requires non-negative values"
        return pd.Series(np.sqrt(values), index=values.index), None

    if method == "cube_root":
        return pd.Series(np.cbrt(values), index=values.index), None

    if method == "reciprocal":
        if (values == 0).any() or (values.abs() < 1e-12).any():
            return None, "Contains zero or near-zero values"
        return pd.Series(1.0 / values, index=values.index), None

    if method == "square":
        return pd.Series(np.square(values), index=values.index), None

    if method == "exponential":
        clipped = values.clip(-50, 50)
        return pd.Series(np.exp(clipped), index=values.index), None

    if method == "box_cox":
        if (values <= 0).any():
            return None, "Box-Cox requires positive values"
        if PowerTransformer is None:
            return None, "PowerTransformer unavailable"
        transformer = PowerTransformer(method="box-cox", standardize=False)
        transformed = transformer.fit_transform(values.to_numpy().reshape(-1, 1)).ravel()
        return pd.Series(transformed, index=values.index), None

    if method == "yeo_johnson":
        if PowerTransformer is None:
            return None, "PowerTransformer unavailable"
        transformer = PowerTransformer(method="yeo-johnson", standardize=False)
        transformed = transformer.fit_transform(values.to_numpy().reshape(-1, 1)).ravel()
        return pd.Series(transformed, index=values.index), None

    return None, "Unknown method"


def _fit_power_transformer(values: pd.Series, method: str) -> Tuple[Optional[Any], Optional[str]]:
    """Fit a power transformer for Box-Cox or Yeo-Johnson methods."""

    if PowerTransformer is None:  # pragma: no cover - optional dependency
        return None, "PowerTransformer unavailable"

    sanitized = pd.to_numeric(values, errors="coerce").dropna()
    if sanitized.empty:
        return None, "No numeric data"

    if method == "box_cox" and (sanitized <= 0).any():
        return None, "Box-Cox requires positive values"

    power_method = "box-cox" if method == "box_cox" else "yeo-johnson"
    transformer = PowerTransformer(method=power_method, standardize=False)
    try:
        transformer.fit(sanitized.to_numpy().reshape(-1, 1))
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Failed to fit transformer: {exc}"

    return transformer, None


def _apply_power_transform(transformer: Any, values: pd.Series) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Apply a fitted power transformer to the provided values."""

    if transformer is None:
        return None, "Transformer unavailable"

    sanitized = pd.to_numeric(values, errors="coerce").dropna()
    if sanitized.empty:
        return None, "No numeric data"

    try:
        transformed = transformer.transform(sanitized.to_numpy().reshape(-1, 1)).reshape(-1)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Failed to transform values: {exc}"

    return pd.Series(transformed, index=sanitized.index), None


def _apply_skewness_transformations(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, SkewnessNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None
    signal = SkewnessNodeSignal(node_id=str(node_id) if node_id is not None else None)

    if frame.empty:
        return frame, "Skewness transforms: no data available", signal

    data = node.get("data") or {}
    config = data.get("config") or {}
    raw_transformations = config.get("transformations")

    if not isinstance(raw_transformations, list) or not raw_transformations:
        return frame, "Skewness transforms: no transformations configured", signal

    transformations: List[Tuple[str, str]] = []
    for entry in raw_transformations:
        if not isinstance(entry, dict):
            continue
        column = str(entry.get("column") or "").strip()
        method = str(entry.get("method") or "").strip().lower()
        if column and method in SKEWNESS_METHODS:
            transformations.append((column, method))

    if not transformations:
        return frame, "Skewness transforms: no transformations configured", signal

    working_frame = frame.copy()
    applied: List[str] = []
    skipped: List[str] = []

    method_labels = {key: meta["label"] for key, meta in SKEWNESS_METHODS.items()}
    power_methods = {"box_cox", "yeo_johnson"}
    transformer_name = "skewness_transform"

    has_split_column = SPLIT_TYPE_COLUMN in working_frame.columns
    storage = get_pipeline_store() if pipeline_id and has_split_column else None
    split_counts: Dict[str, int] = (
        working_frame[SPLIT_TYPE_COLUMN].value_counts().to_dict() if has_split_column else {}
    )
    train_mask = (working_frame[SPLIT_TYPE_COLUMN] == "train") if has_split_column else None
    train_row_count = int(split_counts.get("train", 0)) if has_split_column else 0

    for column, method in transformations:
        signal.configured_transformations.append(
            SkewnessConfiguredTransformation(
                column=column,
                method=method,
                method_label=method_labels.get(method),
            )
        )

    for column, method in transformations:
        if column not in working_frame.columns:
            skipped.append(f"{column} missing")
            signal.skipped_columns.append(
                SkewnessSkippedColumnSignal(
                    column=column,
                    reason="missing",
                    method=method,
                    method_label=method_labels.get(method),
                )
            )
            continue

        raw_series = working_frame[column]
        if pd.api.types.is_bool_dtype(raw_series):
            skipped.append(f"{column} (boolean column automatically excluded)")
            signal.skipped_columns.append(
                SkewnessSkippedColumnSignal(
                    column=column,
                    reason="boolean column automatically excluded",
                    method=method,
                    method_label=method_labels.get(method),
                )
            )
            continue

        numeric_series = pd.to_numeric(raw_series, errors="coerce")
        valid_mask = numeric_series.notna()
        if not valid_mask.any():
            skipped.append(f"{column} not numeric")
            signal.skipped_columns.append(
                SkewnessSkippedColumnSignal(
                    column=column,
                    reason="not numeric",
                    method=method,
                    method_label=method_labels.get(method),
                )
            )
            continue

        valid_values = numeric_series.loc[valid_mask]
        if _is_binary_numeric(valid_values):
            skipped.append(f"{column} (binary indicator column excluded)")
            signal.skipped_columns.append(
                SkewnessSkippedColumnSignal(
                    column=column,
                    reason="binary indicator column excluded",
                    method=method,
                    method_label=method_labels.get(method),
                )
            )
            continue

        original_skew = valid_values.skew()
        if original_skew is not None and not math.isfinite(original_skew):
            original_skew = None

        requires_power_transformer = method in power_methods
        transformed_series: Optional[pd.Series] = None
        transformer_status: Optional[str] = None  # "fit" | "reuse"
        skip_reason: Optional[str] = None

        stored_transformer: Optional[Any] = None
        if storage and requires_power_transformer:
            stored_transformer = storage.get_transformer(
                pipeline_id=pipeline_id,  # type: ignore[arg-type]
                node_id=str(node_id),
                transformer_name=transformer_name,
                column_name=column,
            )
            stored_metadata = storage.get_metadata(
                pipeline_id=pipeline_id,  # type: ignore[arg-type]
                node_id=str(node_id),
                transformer_name=transformer_name,
                column_name=column,
            )
            if isinstance(stored_metadata, dict):
                stored_method = str(stored_metadata.get("method") or "").strip().lower()
                if stored_method and stored_method != method:
                    stored_transformer = None
                    stored_metadata = None

        train_valid_values: Optional[pd.Series] = None
        if train_mask is not None:
            train_subset = numeric_series.loc[train_mask & valid_mask]
            train_valid_values = train_subset.dropna()

        if requires_power_transformer:
            if storage and train_mask is not None:
                can_fit = (
                    train_row_count > 0
                    and train_valid_values is not None
                    and not train_valid_values.empty
                )

                if can_fit and train_valid_values is not None:
                    transformer, fit_error = _fit_power_transformer(train_valid_values, method)
                    if transformer is not None:
                        candidate_series, transform_error = _apply_power_transform(transformer, valid_values)
                        if candidate_series is not None and transform_error is None:
                            transformed_series = candidate_series
                            transformer_status = "fit"

                            metadata: Dict[str, Any] = {
                                "method": method,
                                "method_label": method_labels.get(method),
                                "train_rows": train_row_count,
                                "train_valid_rows": int(train_valid_values.shape[0]),
                            }
                            if hasattr(transformer, "lambdas_"):
                                try:
                                    metadata["lambdas"] = getattr(transformer, "lambdas_").tolist()
                                except Exception:  # pragma: no cover - defensive
                                    pass

                            storage.store_transformer(
                                pipeline_id=pipeline_id,  # type: ignore[arg-type]
                                node_id=str(node_id),
                                transformer_name=transformer_name,
                                transformer=transformer,
                                column_name=column,
                                metadata=metadata,
                            )
                        else:
                            skip_reason = transform_error or "Transform failed"
                    else:
                        skip_reason = fit_error or "Failed to fit transformer"

                if transformed_series is None and stored_transformer is not None:
                    candidate_series, reuse_error = _apply_power_transform(stored_transformer, valid_values)
                    if candidate_series is not None and reuse_error is None:
                        transformed_series = candidate_series
                        transformer_status = "reuse"
                        skip_reason = None
                    else:
                        skip_reason = reuse_error or skip_reason

                if transformed_series is None and not can_fit and stored_transformer is None:
                    skip_reason = skip_reason or "Training split unavailable to fit transformer"

            else:
                transformer, fit_error = _fit_power_transformer(valid_values, method)
                if transformer is not None:
                    candidate_series, transform_error = _apply_power_transform(transformer, valid_values)
                    if candidate_series is not None and transform_error is None:
                        transformed_series = candidate_series
                        transformer_used = transformer
                        transformer_status = "fit"
                    else:
                        skip_reason = transform_error or "Transform failed"
                else:
                    skip_reason = fit_error or "Failed to fit transformer"

        else:
            candidate_series, reason = _perform_skewness_transform(valid_values, method)
            if candidate_series is not None:
                transformed_series = candidate_series
            else:
                skip_reason = reason or "unsupported"

        if transformed_series is None:
            detail = skip_reason or "unsupported"
            skipped.append(f"{column} {method_labels.get(method, method)} ({detail})")
            signal.skipped_columns.append(
                SkewnessSkippedColumnSignal(
                    column=column,
                    reason=detail,
                    method=method,
                    method_label=method_labels.get(method),
                )
            )
            continue

        transformed_aligned = transformed_series.reindex(valid_values.index)
        updated_series = numeric_series.copy()
        updated_series.loc[valid_mask] = transformed_aligned
        working_frame[column] = updated_series
        applied.append(f"{column} via {method_labels.get(method, method)}")

        transformed_skew = transformed_aligned.skew()
        if transformed_skew is not None and not math.isfinite(transformed_skew):
            transformed_skew = None

        total_rows = int(numeric_series.shape[0])
        valid_rows = int(valid_mask.sum())
        missing_rows = max(total_rows - valid_rows, 0)

        applied_notes: List[str] = []
        if storage and requires_power_transformer:
            if transformer_status == "fit":
                applied_notes.append("Fitted on training split")
            elif transformer_status == "reuse":
                applied_notes.append("Reused stored transformer")
        elif storage and pipeline_id and has_split_column:
            applied_notes.append("Direct transform tracked in audit")

        signal.applied_columns.append(
            SkewnessAppliedColumnSignal(
                column=column,
                method=method,
                method_label=method_labels.get(method),
                transformed_rows=valid_rows,
                total_rows=total_rows,
                missing_rows=missing_rows,
                original_skewness=float(round(original_skew, 6)) if original_skew is not None else None,
                transformed_skewness=float(round(transformed_skew, 6)) if transformed_skew is not None else None,
                notes=applied_notes,
            )
        )

        if storage and pipeline_id and has_split_column:
            if requires_power_transformer:
                train_action = "not_available"
                if transformer_status == "fit":
                    train_action = "fit_transform" if train_row_count > 0 else "fit"
                elif transformer_status == "reuse":
                    train_action = "transform" if train_row_count > 0 else "not_available"

                storage.record_split_activity(
                    pipeline_id=pipeline_id,  # type: ignore[arg-type]
                    node_id=str(node_id),
                    transformer_name=transformer_name,
                    column_name=column,
                    split_name="train",
                    action=train_action,
                    row_count=train_row_count,
                )

                for split_name in ("test", "validation"):
                    rows_processed = int(split_counts.get(split_name, 0))
                    action = "transform" if rows_processed > 0 else "not_available"
                    storage.record_split_activity(
                        pipeline_id=pipeline_id,  # type: ignore[arg-type]
                        node_id=str(node_id),
                        transformer_name=transformer_name,
                        column_name=column,
                        split_name=split_name,
                        action=action,
                        row_count=rows_processed,
                    )
            else:
                direct_train_valid = (
                    int(train_valid_values.shape[0]) if train_valid_values is not None else valid_rows
                )
                metadata: Dict[str, Any] = {
                    "method": method,
                    "method_label": method_labels.get(method),
                    "train_rows": train_row_count,
                    "train_valid_rows": direct_train_valid,
                    "notes": ["Direct transform; no fitted parameters"],
                }
                storage.store_transformer(
                    pipeline_id=pipeline_id,  # type: ignore[arg-type]
                    node_id=str(node_id),
                    transformer_name=transformer_name,
                    transformer=None,
                    column_name=column,
                    metadata=metadata,
                )

                storage.record_split_activity(
                    pipeline_id=pipeline_id,  # type: ignore[arg-type]
                    node_id=str(node_id),
                    transformer_name=transformer_name,
                    column_name=column,
                    split_name="train",
                    action="transform" if train_row_count > 0 else "not_available",
                    row_count=train_row_count,
                )

                for split_name in ("test", "validation"):
                    rows_processed = int(split_counts.get(split_name, 0))
                    action = "transform" if rows_processed > 0 else "not_available"
                    storage.record_split_activity(
                        pipeline_id=pipeline_id,  # type: ignore[arg-type]
                        node_id=str(node_id),
                        transformer_name=transformer_name,
                        column_name=column,
                        split_name=split_name,
                        action=action,
                        row_count=rows_processed,
                    )

    if not applied:
        summary = "Skewness transforms: no columns transformed"
    else:
        summary = "Skewness transforms: " + ", ".join(applied)

    if skipped:
        summary = f"{summary}; skipped {', '.join(skipped)}"

    if signal.skipped_columns:
        seen = set()
        deduped: List[SkewnessSkippedColumnSignal] = []
        for entry in signal.skipped_columns:
            key = (entry.column, entry.reason, entry.method)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        signal.skipped_columns = deduped

    return working_frame, summary, signal


__all__ = [
    "SKEWNESS_METHODS",
    "SKEWNESS_METHOD_ORDER",
    "SKEWNESS_THRESHOLD",
    "_build_skewness_distribution",
    "_apply_skewness_transformations",
    "_build_skewness_recommendations",
    "_skewness_method_details",
]
