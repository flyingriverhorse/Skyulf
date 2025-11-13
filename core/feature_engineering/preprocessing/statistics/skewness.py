"""Skewness transformation helpers for feature engineering nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

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

from core.feature_engineering.pipeline_store_singleton import get_pipeline_store
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN

from ...shared.utils import _is_binary_numeric

SkewnessDirection = Literal["right", "left"]
SkewnessMagnitude = Literal["moderate", "substantial", "extreme"]
SkewnessStatus = Literal["ready", "unsupported"]


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
SKEWNESS_METHOD_LABELS: Dict[str, str] = {key: meta["label"] for key, meta in SKEWNESS_METHODS.items()}
POWER_METHODS = {"box_cox", "yeo_johnson"}
SKEWNESS_TRANSFORMER_NAME = "skewness_transform"


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


def _skewness_direction(skew_value: float) -> SkewnessDirection:
    return "right" if skew_value >= 0 else "left"


def _skewness_magnitude(skew_value: float) -> SkewnessMagnitude:
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

        status: SkewnessStatus = "ready"
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
    direction: SkewnessDirection,
    statuses: Dict[str, SkewnessMethodStatus],
) -> List[str]:
    candidate_order: Tuple[str, ...]
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


def _normalize_method_lookup(raw_methods: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not isinstance(raw_methods, dict):
        return {}

    normalized: Dict[str, str] = {}
    for column, method in raw_methods.items():
        column_key = str(column).strip()
        method_key = str(method).strip().lower()
        if column_key and method_key in SKEWNESS_METHODS:
            normalized[column_key] = method_key
    return normalized


def _build_method_status_map(statuses: Dict[str, SkewnessMethodStatus]) -> Dict[str, SkewnessMethodStatus]:
    method_status_map: Dict[str, SkewnessMethodStatus] = {}
    for method in SKEWNESS_METHOD_ORDER:
        method_status_map[method] = statuses.get(
            method,
            SkewnessMethodStatus(status="unsupported", reason="Unavailable"),
        )
    return method_status_map


def _summarize_skewness(direction: str, skew_value: float, recommended_methods: List[str]) -> str:
    direction_label = "Right-skewed" if direction == "right" else "Left-skewed"
    if recommended_methods:
        label_list = ", ".join(SKEWNESS_METHODS[method]["label"] for method in recommended_methods)
        return f"{direction_label} ({skew_value:.2f}). Try {label_list}."
    return f"{direction_label} ({skew_value:.2f}). Review before selecting a transform."


def _compute_distribution_after(
    series: pd.Series,
    method: Optional[str],
    missing_count: int,
) -> Optional[SkewnessColumnDistribution]:
    if not method:
        return None
    try:
        transformed_series, _ = _perform_skewness_transform(series, method)
    except Exception:  # pragma: no cover - defensive
        return None
    if transformed_series is None:
        return None
    transformed_non_null = transformed_series.dropna()
    if transformed_non_null.empty:
        return None
    return _build_skewness_distribution(transformed_series, missing_count=missing_count)


def _build_single_skewness_recommendation(
    column_name: str,
    original_series: pd.Series,
    numeric_series: pd.Series,
    selected_methods: Dict[str, str],
    applied_methods: Dict[str, str],
) -> Optional[SkewnessColumnRecommendation]:
    if pd.api.types.is_bool_dtype(original_series):
        return None

    numeric_series = pd.to_numeric(numeric_series, errors="coerce")
    series = numeric_series.dropna()
    if series.size < 3:
        return None

    if _is_binary_numeric(series):
        return None

    skew_value = series.skew()
    if skew_value is None or not math.isfinite(skew_value):
        return None

    if abs(skew_value) < SKEWNESS_THRESHOLD:
        return None

    direction = _skewness_direction(skew_value)
    magnitude = _skewness_magnitude(skew_value)
    statuses = _evaluate_skewness_method_status(series)
    recommended_methods = _recommended_methods_for_direction(direction, statuses)
    summary = _summarize_skewness(direction, skew_value, recommended_methods)
    method_status_map = _build_method_status_map(statuses)

    valid_count = int(series.size)
    total_count = int(numeric_series.size)
    missing_count = max(total_count - valid_count, 0)
    distribution_before = _build_skewness_distribution(series, missing_count=missing_count)

    column_key = str(column_name)
    selected_method = selected_methods.get(column_key)
    applied_method = applied_methods.get(column_key)
    transform_method = selected_method or applied_method
    distribution_after = _compute_distribution_after(series, transform_method, missing_count)

    return SkewnessColumnRecommendation(
        column=column_key,
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


def _build_skewness_recommendations(
    frame: pd.DataFrame,
    selected_methods: Optional[Dict[str, str]] = None,
    applied_methods: Optional[Dict[str, str]] = None,
) -> List[SkewnessColumnRecommendation]:
    if frame.empty:
        return []

    numeric_frame = frame.select_dtypes(include=[np.number])
    if numeric_frame.empty:
        return []

    normalized_selected = _normalize_method_lookup(selected_methods)
    normalized_applied = _normalize_method_lookup(applied_methods)

    recommendations: List[SkewnessColumnRecommendation] = []
    for column_name in numeric_frame.columns:
        original_series = frame[column_name] if column_name in frame.columns else numeric_frame[column_name]
        recommendation = _build_single_skewness_recommendation(
            column_name,
            original_series,
            numeric_frame[column_name],
            normalized_selected,
            normalized_applied,
        )
        if recommendation is not None:
            recommendations.append(recommendation)

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


@dataclass
class SkewnessEnvironment:
    has_splits: bool
    storage: Optional[Any]
    pipeline_id: Optional[str]
    node_id: Optional[str]
    split_counts: Dict[str, int]
    train_mask: Optional[pd.Series]
    train_row_count: int

    @property
    def can_persist(self) -> bool:
        return bool(self.has_splits and self.storage and self.pipeline_id and self.node_id)


@dataclass
class SkewnessTransformContext:
    column: str
    method: str
    method_label: Optional[str]
    numeric_series: pd.Series
    valid_mask: pd.Series
    valid_values: pd.Series
    original_skew: Optional[float]
    is_power_method: bool
    train_valid_values: Optional[pd.Series]
    train_valid_rows: Optional[int]


@dataclass
class SkewnessTransformOutcome:
    transformed_values: Optional[pd.Series] = None
    skip_reason: Optional[str] = None
    transformer_status: Optional[str] = None  # "fit" | "reuse"


@dataclass
class SkewnessSkipInfo:
    summary_message: str
    reason: str


def _normalize_skewness_transformations(raw_transformations: Any) -> List[Tuple[str, str]]:
    if not isinstance(raw_transformations, list):
        return []

    normalized: List[Tuple[str, str]] = []
    for entry in raw_transformations:
        if not isinstance(entry, dict):
            continue
        column = str(entry.get("column") or "").strip()
        method = str(entry.get("method") or "").strip().lower()
        if column and method in SKEWNESS_METHODS:
            normalized.append((column, method))
    return normalized


def _build_skewness_environment(
    frame: pd.DataFrame,
    pipeline_id: Optional[str],
    node_id: Optional[str],
) -> SkewnessEnvironment:
    has_splits = SPLIT_TYPE_COLUMN in frame.columns
    if not has_splits:
        return SkewnessEnvironment(
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

    return SkewnessEnvironment(
        has_splits=True,
        storage=storage,
        pipeline_id=pipeline_id if storage is not None else None,
        node_id=node_id,
        split_counts=split_counts,
        train_mask=train_mask,
        train_row_count=train_row_count,
    )


def _load_stored_transformer(
    env: SkewnessEnvironment,
    column: str,
    method: str,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if not env.can_persist:
        return None, None

    assert env.storage is not None  # for mypy
    stored_transformer = env.storage.get_transformer(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name=SKEWNESS_TRANSFORMER_NAME,
        column_name=column,
    )
    stored_metadata = env.storage.get_metadata(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name=SKEWNESS_TRANSFORMER_NAME,
        column_name=column,
    )
    if isinstance(stored_metadata, dict):
        stored_method = str(stored_metadata.get("method") or "").strip().lower()
        if stored_method and stored_method != method:
            return None, None
    else:
        stored_metadata = None
    return stored_transformer, stored_metadata


def _prepare_skewness_context(
    frame: pd.DataFrame,
    env: SkewnessEnvironment,
    column: str,
    method: str,
) -> Tuple[Optional[SkewnessTransformContext], Optional[SkewnessSkipInfo]]:
    if column not in frame.columns:
        return None, SkewnessSkipInfo(summary_message=f"{column} missing", reason="missing")

    raw_series = frame[column]
    if pd.api.types.is_bool_dtype(raw_series):
        return None, SkewnessSkipInfo(
            summary_message=f"{column} (boolean column automatically excluded)",
            reason="boolean column automatically excluded",
        )

    numeric_series = pd.to_numeric(raw_series, errors="coerce")
    valid_mask = numeric_series.notna()
    if not valid_mask.any():
        return None, SkewnessSkipInfo(summary_message=f"{column} not numeric", reason="not numeric")

    valid_values = numeric_series.loc[valid_mask]
    if _is_binary_numeric(valid_values):
        return None, SkewnessSkipInfo(
            summary_message=f"{column} (binary indicator column excluded)",
            reason="binary indicator column excluded",
        )

    original_skew = valid_values.skew()
    if original_skew is not None and not math.isfinite(original_skew):
        original_skew = None

    train_valid_values: Optional[pd.Series] = None
    train_valid_rows: Optional[int] = None
    if env.train_mask is not None:
        train_subset = numeric_series.loc[env.train_mask & valid_mask]
        train_valid_values = train_subset.dropna()
        train_valid_rows = int(train_valid_values.shape[0]) if not train_valid_values.empty else 0

    context = SkewnessTransformContext(
        column=column,
        method=method,
        method_label=SKEWNESS_METHOD_LABELS.get(method),
        numeric_series=numeric_series,
        valid_mask=valid_mask,
        valid_values=valid_values,
        original_skew=original_skew,
        is_power_method=method in POWER_METHODS,
        train_valid_values=train_valid_values if train_valid_rows else None,
        train_valid_rows=train_valid_rows if train_valid_rows is not None else None,
    )
    return context, None


def _handle_power_transform(
    context: SkewnessTransformContext,
    env: SkewnessEnvironment,
) -> SkewnessTransformOutcome:
    outcome = SkewnessTransformOutcome()

    stored_transformer, _ = _load_stored_transformer(env, context.column, context.method)
    skip_reason: Optional[str] = None

    if env.can_persist and env.train_mask is not None:
        can_fit = (
            env.train_row_count > 0
            and context.train_valid_values is not None
            and not context.train_valid_values.empty
        )

        if can_fit and context.train_valid_values is not None:
            transformer, fit_error = _fit_power_transformer(context.train_valid_values, context.method)
            if transformer is not None:
                candidate_series, transform_error = _apply_power_transform(transformer, context.valid_values)
                if candidate_series is not None and transform_error is None:
                    outcome.transformed_values = candidate_series
                    outcome.transformer_status = "fit"

                    metadata: Dict[str, Any] = {
                        "method": context.method,
                        "method_label": context.method_label,
                        "train_rows": env.train_row_count,
                        "train_valid_rows": int(context.train_valid_rows or 0),
                    }
                    if hasattr(transformer, "lambdas_"):
                        try:
                            metadata["lambdas"] = getattr(transformer, "lambdas_").tolist()
                        except Exception:  # pragma: no cover - defensive
                            pass

                    assert env.storage is not None
                    env.storage.store_transformer(
                        pipeline_id=env.pipeline_id,
                        node_id=env.node_id,
                        transformer_name=SKEWNESS_TRANSFORMER_NAME,
                        transformer=transformer,
                        column_name=context.column,
                        metadata=metadata,
                    )
                    return outcome
                skip_reason = transform_error or "Transform failed"
            else:
                skip_reason = fit_error or "Failed to fit transformer"
        else:
            skip_reason = "Training split unavailable to fit transformer"

        if stored_transformer is not None:
            candidate_series, reuse_error = _apply_power_transform(stored_transformer, context.valid_values)
            if candidate_series is not None and reuse_error is None:
                outcome.transformed_values = candidate_series
                outcome.transformer_status = "reuse"
                return outcome
            skip_reason = reuse_error or skip_reason

        outcome.skip_reason = skip_reason or "Transform failed"
        return outcome

    transformer, fit_error = _fit_power_transformer(context.valid_values, context.method)
    if transformer is None:
        outcome.skip_reason = fit_error or "Failed to fit transformer"
        return outcome

    candidate_series, transform_error = _apply_power_transform(transformer, context.valid_values)
    if candidate_series is None or transform_error is not None:
        outcome.skip_reason = transform_error or "Transform failed"
        return outcome

    outcome.transformed_values = candidate_series
    outcome.transformer_status = "fit"
    return outcome


def _apply_direct_skew_transform(context: SkewnessTransformContext) -> SkewnessTransformOutcome:
    outcome = SkewnessTransformOutcome()
    transformed, reason = _perform_skewness_transform(context.valid_values, context.method)
    if transformed is None:
        outcome.skip_reason = reason or "unsupported"
        return outcome
    outcome.transformed_values = transformed
    return outcome


def _record_skewness_skip(
    skipped: List[str],
    signal: SkewnessNodeSignal,
    column: str,
    method: str,
    summary_message: str,
    reason: str,
) -> None:
    skipped.append(summary_message)
    signal.skipped_columns.append(
        SkewnessSkippedColumnSignal(
            column=column,
            reason=reason,
            method=method,
            method_label=SKEWNESS_METHOD_LABELS.get(method),
        )
    )


def _record_power_transform_activity(
    context: SkewnessTransformContext,
    env: SkewnessEnvironment,
    transformer_status: Optional[str],
) -> None:
    if not env.can_persist:
        return

    assert env.storage is not None
    train_action = "not_available"
    if transformer_status == "fit":
        train_action = "fit_transform" if env.train_row_count > 0 else "fit"
    elif transformer_status == "reuse":
        train_action = "transform" if env.train_row_count > 0 else "not_available"

    env.storage.record_split_activity(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name=SKEWNESS_TRANSFORMER_NAME,
        column_name=context.column,
        split_name="train",
        action=train_action,
        row_count=env.train_row_count,
    )

    for split_name in ("test", "validation"):
        rows_processed = int(env.split_counts.get(split_name, 0))
        action = "transform" if rows_processed > 0 else "not_available"
        env.storage.record_split_activity(
            pipeline_id=env.pipeline_id,
            node_id=env.node_id,
            transformer_name=SKEWNESS_TRANSFORMER_NAME,
            column_name=context.column,
            split_name=split_name,
            action=action,
            row_count=rows_processed,
        )


def _record_direct_transform_activity(
    context: SkewnessTransformContext,
    env: SkewnessEnvironment,
    valid_rows: int,
) -> None:
    if not env.can_persist:
        return

    assert env.storage is not None
    direct_train_valid = context.train_valid_rows if context.train_valid_rows is not None else valid_rows
    metadata: Dict[str, Any] = {
        "method": context.method,
        "method_label": context.method_label,
        "train_rows": env.train_row_count,
        "train_valid_rows": int(direct_train_valid),
        "notes": ["Direct transform; no fitted parameters"],
    }
    env.storage.store_transformer(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name=SKEWNESS_TRANSFORMER_NAME,
        transformer=None,
        column_name=context.column,
        metadata=metadata,
    )

    train_action = "transform" if env.train_row_count > 0 else "not_available"
    env.storage.record_split_activity(
        pipeline_id=env.pipeline_id,
        node_id=env.node_id,
        transformer_name=SKEWNESS_TRANSFORMER_NAME,
        column_name=context.column,
        split_name="train",
        action=train_action,
        row_count=env.train_row_count,
    )

    for split_name in ("test", "validation"):
        rows_processed = int(env.split_counts.get(split_name, 0))
        action = "transform" if rows_processed > 0 else "not_available"
        env.storage.record_split_activity(
            pipeline_id=env.pipeline_id,
            node_id=env.node_id,
            transformer_name=SKEWNESS_TRANSFORMER_NAME,
            column_name=context.column,
            split_name=split_name,
            action=action,
            row_count=rows_processed,
        )


def _record_storage_activity(
    context: SkewnessTransformContext,
    env: SkewnessEnvironment,
    outcome: SkewnessTransformOutcome,
    valid_rows: int,
) -> None:
    if not env.can_persist:
        return

    if context.is_power_method:
        _record_power_transform_activity(context, env, outcome.transformer_status)
    else:
        _record_direct_transform_activity(context, env, valid_rows)


def _apply_skewness_transformations(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, SkewnessNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None
    node_id_str = str(node_id) if node_id is not None else None
    signal = SkewnessNodeSignal(node_id=node_id_str)

    if frame.empty:
        return frame, "Skewness transforms: no data available", signal

    data = node.get("data") or {}
    config = data.get("config") or {}
    transformations = _normalize_skewness_transformations(config.get("transformations"))

    if not transformations:
        return frame, "Skewness transforms: no transformations configured", signal

    working_frame = frame.copy()
    applied: List[str] = []
    skipped: List[str] = []

    env = _build_skewness_environment(working_frame, pipeline_id, node_id_str)

    for column, method in transformations:
        signal.configured_transformations.append(
            SkewnessConfiguredTransformation(
                column=column,
                method=method,
                method_label=SKEWNESS_METHOD_LABELS.get(method),
            )
        )

    for column, method in transformations:
        context, skip_info = _prepare_skewness_context(working_frame, env, column, method)
        if context is None:
            if skip_info is not None:
                _record_skewness_skip(
                    skipped,
                    signal,
                    column,
                    method,
                    skip_info.summary_message,
                    skip_info.reason,
                )
            continue

        if context.is_power_method:
            outcome = _handle_power_transform(context, env)
        else:
            outcome = _apply_direct_skew_transform(context)

        if outcome.transformed_values is None:
            detail = outcome.skip_reason or "unsupported"
            method_label = context.method_label or context.method
            summary_message = f"{context.column} {method_label} ({detail})"
            _record_skewness_skip(skipped, signal, context.column, context.method, summary_message, detail)
            continue

        transformed_series = outcome.transformed_values.reindex(context.valid_values.index)
        updated_series = context.numeric_series.copy()
        updated_series.loc[context.valid_mask] = transformed_series
        working_frame[context.column] = updated_series
        applied.append(f"{context.column} via {context.method_label or context.method}")

        transformed_skew = transformed_series.skew()
        if transformed_skew is not None and not math.isfinite(transformed_skew):
            transformed_skew = None

        total_rows = int(context.numeric_series.shape[0])
        valid_rows = int(context.valid_mask.sum())
        missing_rows = max(total_rows - valid_rows, 0)

        applied_notes: List[str] = []
        if env.can_persist:
            if context.is_power_method:
                if outcome.transformer_status == "fit":
                    applied_notes.append("Fitted on training split")
                elif outcome.transformer_status == "reuse":
                    applied_notes.append("Reused stored transformer")
            else:
                applied_notes.append("Direct transform tracked in audit")

        signal.applied_columns.append(
            SkewnessAppliedColumnSignal(
                column=context.column,
                method=context.method,
                method_label=context.method_label,
                transformed_rows=valid_rows,
                total_rows=total_rows,
                missing_rows=missing_rows,
                original_skewness=float(round(context.original_skew, 6)) if context.original_skew is not None else None,
                transformed_skewness=float(round(transformed_skew, 6)) if transformed_skew is not None else None,
                notes=applied_notes,
            )
        )

        _record_storage_activity(context, env, outcome, valid_rows)

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
