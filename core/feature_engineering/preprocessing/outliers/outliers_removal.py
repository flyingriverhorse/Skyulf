"""Helpers for configuring and applying outlier removal nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd

try:  # pragma: no cover - optional dependency guard
    from sklearn.covariance import EllipticEnvelope  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - defensive guard
    EllipticEnvelope = None  # type: ignore[assignment]

from core.feature_engineering.schemas import (
    OutlierAppliedColumnSignal,
    OutlierColumnInsight,
    OutlierColumnStats,
    OutlierMethodDetail,
    OutlierMethodName,
    OutlierMethodSummary,
    OutlierNodeSignal,
)

from core.feature_engineering.pipeline_store_singleton import get_pipeline_store
from core.feature_engineering.preprocessing.split import SPLIT_TYPE_COLUMN

from ...shared.utils import (
    _coerce_config_boolean,
    _detect_numeric_columns,
    _is_binary_numeric,
)

OUTLIER_METHODS: Dict[OutlierMethodName, Dict[str, Any]] = {
    "iqr": {
        "label": "IQR filter",
        "description": "Drop rows outside Tukey-style whiskers based on the interquartile range.",
        "action": "remove",
        "notes": [
            "Robust to skewed distributions when whisker multiplier is moderate.",
        ],
        "default_parameters": {"multiplier": 1.5},
        "parameter_help": {
            "multiplier": "Whisker multiplier applied to the IQR (1.5 aligns with Tukey boxplots).",
        },
    },
    "zscore": {
        "label": "Z-score filter",
        "description": "Drop rows whose standardized value exceeds a configured z-score threshold.",
        "action": "remove",
        "notes": [
            "Works best when the feature is approximately Gaussian.",
        ],
        "default_parameters": {"threshold": 3.0},
        "parameter_help": {
            "threshold": "Absolute z-score above which rows are removed (common range 2.0–4.0).",
        },
    },
    "elliptic_envelope": {
        "label": "Elliptic Envelope",
        "description": "Robust covariance-based detector that models inliers within an ellipse.",
        "action": "remove",
        "notes": [
            "Assumes features follow an approximately Gaussian distribution.",
            "Requires scikit-learn; falls back gracefully if unavailable.",
        ],
        "default_parameters": {"contamination": 0.01},
        "parameter_help": {
            "contamination": "Expected proportion of outliers (0 < contamination < 0.5).",
        },
    },
    "winsorize": {
        "label": "Winsorize",
        "description": "Cap extreme values at configured percentile bounds instead of dropping rows.",
        "action": "cap",
        "notes": [
            "Keeps row count stable while limiting the influence of heavy tails.",
        ],
        "default_parameters": {"lower_percentile": 5.0, "upper_percentile": 95.0},
        "parameter_help": {
            "lower_percentile": "Lower percentile used for capping (0–100).",
            "upper_percentile": "Upper percentile used for capping (0–100).",
        },
    },
    "manual": {
        "label": "Manual bounds",
        "description": "Use explicit lower and/or upper limits that you supply per column.",
        "action": "manual",
        "notes": [
            "When limits are provided the node removes rows outside the interval.",
        ],
        "default_parameters": {},
        "parameter_help": {
            "lower_bound": "Optional lower bound applied when configured per column.",
            "upper_bound": "Optional upper bound applied when configured per column.",
        },
    },
}

OUTLIER_METHOD_ORDER: Tuple[OutlierMethodName, ...] = (
    "iqr",
    "zscore",
    "elliptic_envelope",
    "winsorize",
    "manual",
)
OUTLIER_DEFAULT_METHOD: OutlierMethodName = "iqr"

OUTLIER_PARAMETER_KEYS: Dict[OutlierMethodName, set[str]] = {
    "zscore": {"threshold"},
    "iqr": {"multiplier"},
    "elliptic_envelope": {"contamination"},
    "winsorize": {"lower_percentile", "upper_percentile"},
    "manual": {"lower_bound", "upper_bound"},
}
ALL_PARAMETER_KEYS: set[str] = set().union(*OUTLIER_PARAMETER_KEYS.values())

DEFAULT_METHOD_PARAMETERS: Dict[OutlierMethodName, Dict[str, float]] = {
    method: dict(metadata.get("default_parameters", {}))
    for method, metadata in OUTLIER_METHODS.items()
}


@dataclass
class NormalizedOutlierConfig:
    columns: List[str]
    default_method: OutlierMethodName
    column_methods: Dict[str, OutlierMethodName]
    auto_detect: bool
    skipped_columns: List[str]
    method_parameters: Dict[OutlierMethodName, Dict[str, float]]
    column_parameters: Dict[str, Dict[str, float]]


@dataclass
class ColumnContext:
    column: str
    numeric_series: pd.Series
    stats: OutlierColumnStats
    method: OutlierMethodName
    params: Dict[str, Any]
    column_params: Dict[str, Any]
    total_rows: int


@dataclass
class ColumnProcessResult:
    applied_signal: OutlierAppliedColumnSignal
    mask_to_remove: Optional[pd.Series] = None
    clipped_series: Optional[pd.Series] = None
    metadata_to_store: Optional[Dict[str, Any]] = None
    transformer_to_store: Optional[Any] = None
    transformer_status: Optional[str] = None
    skip_reason: Optional[str] = None
    applied_notes: List[str] = field(default_factory=list)
    clip_column: bool = False


TRAINING_SKIP_MESSAGES: Dict[str, str] = {
    "zscore": "Training split unavailable to fit z-score thresholds",
    "iqr": "Training split unavailable to fit IQR bounds",
    "winsorize": "Training split unavailable to fit winsor bounds",
    "elliptic_envelope": "Training split unavailable to fit Elliptic Envelope",
}


def _normalize_column_list(value: Any) -> List[str]:
    if isinstance(value, list):
        items = [str(item or "").strip() for item in value]
    elif isinstance(value, str):
        items = [segment.strip() for segment in value.split(",")]
    else:
        items = []
    unique = {
        item
        for item in items
        if item
    }
    return sorted(unique)


def _sanitize_numeric(
    value: Any,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if minimum is not None and numeric < minimum:
        return None
    if maximum is not None and numeric > maximum:
        return None
    return float(numeric)


def _normalize_outlier_method_parameters(raw: Any) -> Dict[OutlierMethodName, Dict[str, float]]:
    parameters: Dict[OutlierMethodName, Dict[str, float]] = {
        method: dict(DEFAULT_METHOD_PARAMETERS.get(method, {}))
        for method in OUTLIER_METHODS
    }

    if isinstance(raw, dict):
        for method_key, payload in raw.items():
            method = str(method_key or "").strip().lower()
            if method not in OUTLIER_METHODS:
                continue
            method_literal = cast(OutlierMethodName, method)
            allowed_keys = OUTLIER_PARAMETER_KEYS[method_literal]
            current = parameters[method_literal]
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                normalized_key = str(key or "").strip().lower()
                if normalized_key not in allowed_keys:
                    continue
                if normalized_key == "threshold":
                    numeric = _sanitize_numeric(value, minimum=0.1)
                elif normalized_key == "multiplier":
                    numeric = _sanitize_numeric(value, minimum=0.1)
                elif normalized_key == "contamination":
                    numeric = _sanitize_numeric(value, minimum=0.0001, maximum=0.49)
                elif normalized_key in {"lower_percentile", "upper_percentile"}:
                    numeric = _sanitize_numeric(value, minimum=0.0, maximum=100.0)
                else:
                    numeric = _sanitize_numeric(value)
                if numeric is not None:
                    current[normalized_key] = numeric

    winsor_params = parameters["winsorize"]
    lower = winsor_params.get("lower_percentile")
    upper = winsor_params.get("upper_percentile")
    if lower is None or upper is None:
        winsor_params["lower_percentile"] = DEFAULT_METHOD_PARAMETERS["winsorize"].get("lower_percentile", 5.0)
        winsor_params["upper_percentile"] = DEFAULT_METHOD_PARAMETERS["winsorize"].get("upper_percentile", 95.0)
    else:
        lower = max(0.0, min(100.0, lower))
        upper = max(0.0, min(100.0, upper))
        if lower >= upper:
            winsor_params["lower_percentile"] = DEFAULT_METHOD_PARAMETERS["winsorize"].get("lower_percentile", 5.0)
            winsor_params["upper_percentile"] = DEFAULT_METHOD_PARAMETERS["winsorize"].get("upper_percentile", 95.0)
        else:
            winsor_params["lower_percentile"] = lower
            winsor_params["upper_percentile"] = upper

    return parameters


def _normalize_outlier_column_parameters(raw: Any) -> Dict[str, Dict[str, float]]:
    if not isinstance(raw, dict):
        return {}

    normalized: Dict[str, Dict[str, float]] = {}

    for column, payload in raw.items():
        column_name = str(column or "").strip()
        if not column_name or not isinstance(payload, dict):
            continue

        column_params: Dict[str, float] = {}
        for key, value in payload.items():
            normalized_key = str(key or "").strip().lower()
            if normalized_key not in ALL_PARAMETER_KEYS:
                continue

            if normalized_key == "threshold":
                numeric = _sanitize_numeric(value, minimum=0.1)
            elif normalized_key == "multiplier":
                numeric = _sanitize_numeric(value, minimum=0.1)
            elif normalized_key == "contamination":
                numeric = _sanitize_numeric(value, minimum=0.0001, maximum=0.49)
            elif normalized_key in {"lower_percentile", "upper_percentile"}:
                numeric = _sanitize_numeric(value, minimum=0.0, maximum=100.0)
            else:
                numeric = _sanitize_numeric(value)

            if numeric is not None:
                column_params[normalized_key] = numeric

        if column_params:
            normalized[column_name] = column_params

    return normalized


def _normalize_outlier_config(config: Any) -> NormalizedOutlierConfig:
    payload = config if isinstance(config, dict) else {}

    columns = _normalize_column_list(payload.get("columns"))

    raw_default_method = str(payload.get("default_method") or "").strip().lower()
    if raw_default_method in OUTLIER_METHODS:
        default_method: OutlierMethodName = cast(OutlierMethodName, raw_default_method)
    else:
        default_method = OUTLIER_DEFAULT_METHOD

    column_methods: Dict[str, OutlierMethodName] = {}
    raw_column_methods = payload.get("column_methods")
    if isinstance(raw_column_methods, dict):
        for column, method in raw_column_methods.items():
            column_name = str(column or "").strip()
            method_key = str(method or "").strip().lower()
            if column_name and method_key in OUTLIER_METHODS:
                column_methods[column_name] = cast(OutlierMethodName, method_key)

    auto_detect = _coerce_config_boolean(payload.get("auto_detect"), default=True)

    skipped_columns = _normalize_column_list(payload.get("skipped_columns"))

    method_parameters = _normalize_outlier_method_parameters(payload.get("method_parameters"))
    column_parameters = _normalize_outlier_column_parameters(payload.get("column_parameters"))

    return NormalizedOutlierConfig(
        columns=columns,
        default_method=default_method,
        column_methods=column_methods,
        auto_detect=auto_detect,
        skipped_columns=skipped_columns,
        method_parameters=method_parameters,
        column_parameters=column_parameters,
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return float(round(numeric, 6))


def _prepare_numeric_series(series: pd.Series) -> Optional[Tuple[pd.Series, OutlierColumnStats]]:
    numeric_series = pd.to_numeric(series, errors="coerce")
    valid = numeric_series.dropna()
    valid_count = int(valid.shape[0])

    if valid_count < 3:
        return None

    if _is_binary_numeric(valid):
        return None

    if valid.nunique(dropna=True) < 3:
        return None

    q1 = _safe_float(valid.quantile(0.25))
    q3 = _safe_float(valid.quantile(0.75))
    iqr: Optional[float] = None
    if q1 is not None and q3 is not None:
        iqr = _safe_float(q3 - q1)

    stats = OutlierColumnStats(
        valid_count=valid_count,
        mean=_safe_float(valid.mean()),
        median=_safe_float(valid.median()),
        stddev=_safe_float(valid.std(ddof=0)),
        minimum=_safe_float(valid.min()),
        maximum=_safe_float(valid.max()),
        q1=q1,
        q3=q3,
        iqr=iqr,
    )

    return numeric_series, stats


def _summarize_zscore(
    series: pd.Series,
    stats: OutlierColumnStats,
    threshold: float,
) -> OutlierMethodSummary:
    mean = stats.mean
    stddev = stats.stddev
    total_rows = int(series.dropna().shape[0])
    summary = OutlierMethodSummary(method="zscore", action="remove")

    if total_rows == 0 or mean is None or stddev is None or stddev <= 0 or threshold <= 0:
        summary.notes.append("Z-score not applicable (insufficient variance).")
        return summary

    lower_bound = mean - threshold * stddev
    upper_bound = mean + threshold * stddev
    mask = (series < lower_bound) | (series > upper_bound)
    affected_rows = int(mask.sum())

    summary.lower_bound = _safe_float(lower_bound)
    summary.upper_bound = _safe_float(upper_bound)
    summary.affected_rows = affected_rows
    summary.affected_ratio = float(round(affected_rows / total_rows, 6)) if total_rows else 0.0

    if affected_rows == 0:
        summary.notes.append("No rows exceed the z-score threshold.")

    return summary


def _summarize_iqr(
    series: pd.Series,
    stats: OutlierColumnStats,
    multiplier: float,
) -> OutlierMethodSummary:
    q1 = stats.q1
    q3 = stats.q3
    iqr = stats.iqr
    total_rows = int(series.dropna().shape[0])
    summary = OutlierMethodSummary(method="iqr", action="remove")

    if total_rows == 0 or q1 is None or q3 is None or iqr is None or iqr <= 0 or multiplier <= 0:
        summary.notes.append("IQR filter not applicable (insufficient spread).")
        return summary

    whisker = multiplier * iqr
    lower_bound = q1 - whisker
    upper_bound = q3 + whisker
    mask = (series < lower_bound) | (series > upper_bound)
    affected_rows = int(mask.sum())

    summary.lower_bound = _safe_float(lower_bound)
    summary.upper_bound = _safe_float(upper_bound)
    summary.affected_rows = affected_rows
    summary.affected_ratio = float(round(affected_rows / total_rows, 6)) if total_rows else 0.0

    if affected_rows == 0:
        summary.notes.append("No rows fall outside the IQR whiskers.")

    return summary


def _summarize_winsorize(
    series: pd.Series,
    lower_percentile: float,
    upper_percentile: float,
) -> OutlierMethodSummary:
    valid = series.dropna()
    total_rows = int(valid.shape[0])
    summary = OutlierMethodSummary(method="winsorize", action="cap")

    if total_rows == 0:
        summary.notes.append("Winsorize not applicable (no numeric rows).")
        return summary

    lower_frac = max(0.0, min(1.0, lower_percentile / 100.0))
    upper_frac = max(0.0, min(1.0, upper_percentile / 100.0))

    if lower_frac >= upper_frac:
        lower_frac = 0.05
        upper_frac = 0.95

    lower_bound = valid.quantile(lower_frac)
    upper_bound = valid.quantile(upper_frac)

    mask_low = valid < lower_bound
    mask_high = valid > upper_bound
    affected_rows = int(mask_low.sum() + mask_high.sum())

    summary.lower_bound = _safe_float(lower_bound)
    summary.upper_bound = _safe_float(upper_bound)
    summary.affected_rows = affected_rows
    summary.affected_ratio = float(round(affected_rows / total_rows, 6)) if total_rows else 0.0

    if affected_rows == 0:
        summary.notes.append("All rows already fall within the winsorized range.")

    return summary


def _compute_zscore_bounds(values: pd.Series, threshold: float) -> Optional[Tuple[float, float, float, float]]:
    sanitized = pd.to_numeric(values, errors="coerce").dropna()
    if sanitized.shape[0] < 3:
        return None

    stddev = float(sanitized.std(ddof=0))
    if not math.isfinite(stddev) or stddev <= 0:
        return None

    mean = float(sanitized.mean())
    if not math.isfinite(mean):
        return None

    lower = mean - threshold * stddev
    upper = mean + threshold * stddev
    return mean, stddev, lower, upper


def _compute_iqr_bounds(values: pd.Series, multiplier: float) -> Optional[Tuple[float, float, float, float, float]]:
    sanitized = pd.to_numeric(values, errors="coerce").dropna()
    if sanitized.shape[0] < 3:
        return None

    q1 = float(sanitized.quantile(0.25))
    q3 = float(sanitized.quantile(0.75))
    if not (math.isfinite(q1) and math.isfinite(q3)):
        return None

    iqr = q3 - q1
    if not math.isfinite(iqr) or iqr <= 0:
        return None

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return q1, q3, iqr, lower, upper


def _compute_winsor_bounds(
    values: pd.Series,
    lower_percentile: float,
    upper_percentile: float,
) -> Optional[Tuple[float, float]]:
    sanitized = pd.to_numeric(values, errors="coerce").dropna()
    if sanitized.empty:
        return None

    lower_frac = max(0.0, min(1.0, lower_percentile / 100.0))
    upper_frac = max(0.0, min(1.0, upper_percentile / 100.0))
    if lower_frac >= upper_frac:
        return None

    lower_value = float(sanitized.quantile(lower_frac))
    upper_value = float(sanitized.quantile(upper_frac))
    if not (math.isfinite(lower_value) and math.isfinite(upper_value)):
        return None

    return lower_value, upper_value


def _format_bound(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{float(value):.4f}"


def _fit_elliptic_envelope_model(
    values: pd.Series,
    contamination: float,
) -> Tuple[Optional[Any], Optional[str]]:
    if EllipticEnvelope is None:  # pragma: no cover - optional dependency
        return None, "EllipticEnvelope unavailable (scikit-learn not installed)."

    sanitized = pd.to_numeric(values, errors="coerce").dropna()
    if sanitized.shape[0] < 5:
        return None, "Requires at least 5 numeric rows to fit Elliptic Envelope."

    contamination_value = float(contamination or 0.01)
    if not math.isfinite(contamination_value):
        contamination_value = 0.01
    contamination_value = max(0.0001, min(0.49, contamination_value))

    try:
        model = EllipticEnvelope(contamination=contamination_value)
        model.fit(sanitized.to_numpy().reshape(-1, 1))
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Elliptic Envelope fit failed: {exc}"

    return model, None


def _apply_elliptic_envelope_mask(transformer: Any, values: pd.Series) -> Tuple[Optional[pd.Series], Optional[str]]:
    if transformer is None:  # pragma: no cover - defensive guard
        return None, "Transformer unavailable"

    sanitized_full = pd.to_numeric(values, errors="coerce")
    valid = sanitized_full.dropna()
    if valid.empty:
        return None, "No numeric data"

    try:
        predictions = transformer.predict(valid.to_numpy().reshape(-1, 1))
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Elliptic Envelope predict failed: {exc}"

    mask = pd.Series(predictions == -1, index=valid.index)
    return mask, None


def _evaluate_elliptic_envelope(
    series: pd.Series,
    contamination: float,
) -> Tuple[OutlierMethodSummary, Optional[pd.Series]]:
    summary = OutlierMethodSummary(method="elliptic_envelope", action="remove")

    model, error = _fit_elliptic_envelope_model(series, contamination)
    if model is None:
        summary.notes.append(error or "Elliptic Envelope unavailable.")
        return summary, None

    mask, apply_error = _apply_elliptic_envelope_mask(model, series)
    if mask is None:
        summary.notes.append(apply_error or "Unable to evaluate Elliptic Envelope predictions.")
        return summary, None

    total_rows = int(series.dropna().shape[0])
    affected_rows = int(mask.sum())
    summary.affected_rows = affected_rows
    summary.affected_ratio = float(round(affected_rows / total_rows, 6)) if total_rows else 0.0

    if affected_rows == 0:
        summary.notes.append("No rows flagged as outliers by Elliptic Envelope.")
    else:
        summary.notes.append(
            f"Flagged {affected_rows} of {total_rows} rows "
            f"(~{summary.affected_ratio * 100:.2f}%)."
        )

    return summary, mask


def _summarize_manual(
    series: pd.Series,
    lower_bound: Optional[float],
    upper_bound: Optional[float],
) -> OutlierMethodSummary:
    valid = series.dropna()
    total_rows = int(valid.shape[0])
    summary = OutlierMethodSummary(method="manual", action="remove")

    if total_rows == 0:
        summary.notes.append("Manual bounds not applicable (no numeric rows).")
        return summary

    lb = lower_bound
    ub = upper_bound

    if lb is not None and ub is not None and lb > ub:
        lb, ub = ub, lb

    summary.lower_bound = _safe_float(lb)
    summary.upper_bound = _safe_float(ub)

    if lb is None and ub is None:
        summary.notes.append("Provide at least one bound to remove rows.")
        return summary

    mask = pd.Series(False, index=valid.index)
    if lb is not None:
        mask |= valid < lb
    if ub is not None:
        mask |= valid > ub
    affected_rows = int(mask.sum())

    summary.affected_rows = affected_rows
    summary.affected_ratio = float(round(affected_rows / total_rows, 6)) if total_rows else 0.0

    if affected_rows == 0:
        summary.notes.append("No rows violate the provided bounds.")

    return summary


def _build_outlier_recommendations(frame: pd.DataFrame) -> List[OutlierColumnInsight]:
    recommendations: List[OutlierColumnInsight] = []

    if frame.empty:
        return recommendations

    numeric_columns = _detect_numeric_columns(frame)

    for column in numeric_columns:
        if column not in frame.columns:
            continue

        prepared = _prepare_numeric_series(frame[column])
        if prepared is None:
            continue

        numeric_series, stats = prepared
        method_summaries: List[OutlierMethodSummary] = []

        params = DEFAULT_METHOD_PARAMETERS

        zscore_threshold = params["zscore"].get("threshold", 3.0)
        method_summaries.append(_summarize_zscore(numeric_series, stats, zscore_threshold))

        iqr_multiplier = params["iqr"].get("multiplier", 1.5)
        method_summaries.append(_summarize_iqr(numeric_series, stats, iqr_multiplier))

        elliptic_params = params["elliptic_envelope"]
        elliptic_summary, _ = _evaluate_elliptic_envelope(
            numeric_series,
            elliptic_params.get("contamination", 0.01),
        )
        method_summaries.append(elliptic_summary)

        winsor_params = params["winsorize"]
        method_summaries.append(
            _summarize_winsorize(
                numeric_series,
                winsor_params.get("lower_percentile", 5.0),
                winsor_params.get("upper_percentile", 95.0),
            )
        )

        method_summaries.append(_summarize_manual(numeric_series, None, None))

        recommended_method: Optional[OutlierMethodName] = None
        recommended_reason: Optional[str] = None

        candidate = [
            summary
            for summary in method_summaries
            if summary.method != "manual" and summary.affected_ratio > 0
        ]

        if candidate:
            candidate.sort(key=lambda entry: entry.affected_ratio)
            choice = candidate[0]
            recommended_method = choice.method
            recommended_reason = f"~{choice.affected_ratio * 100:.1f}% rows outside suggested range"
        else:
            recommended_reason = "No potential outliers detected with default parameters."

        recommendations.append(
            OutlierColumnInsight(
                column=column,
                dtype=str(frame[column].dtype),
                stats=stats,
                method_summaries=method_summaries,
                recommended_method=recommended_method,
                recommended_reason=recommended_reason,
                has_missing=bool(frame[column].isna().any()),
            )
        )

    return recommendations


def _outlier_method_details() -> List[OutlierMethodDetail]:
    details: List[OutlierMethodDetail] = []
    for method in OUTLIER_METHOD_ORDER:
        metadata = OUTLIER_METHODS[method]
        details.append(
            OutlierMethodDetail(
                key=method,
                label=metadata.get("label", method.title()),
                description=metadata.get("description"),
                action=metadata.get("action", "remove"),
                notes=list(metadata.get("notes", [])),
                default_parameters=dict(metadata.get("default_parameters", {})),
                parameter_help=dict(metadata.get("parameter_help", {})),
            )
        )
    return details


def _process_column_without_storage(context: ColumnContext, working_index: pd.Index) -> ColumnProcessResult:
    applied_signal = OutlierAppliedColumnSignal(
        column=context.column,
        method=context.method,
        action="remove",
        total_rows=context.total_rows,
    )
    result = ColumnProcessResult(applied_signal=applied_signal)

    numeric_series = context.numeric_series
    stats = context.stats
    method = context.method
    params = context.params
    column_params = context.column_params

    if method == "zscore":
        threshold = params.get("threshold", DEFAULT_METHOD_PARAMETERS["zscore"].get("threshold", 3.0))
        summary = _summarize_zscore(numeric_series, stats, threshold)
        applied_signal.action = "remove"
        applied_signal.lower_bound = summary.lower_bound
        applied_signal.upper_bound = summary.upper_bound
        applied_signal.affected_rows = summary.affected_rows
        applied_signal.notes = summary.notes
        if summary.affected_rows and summary.lower_bound is not None and summary.upper_bound is not None:
            mask = (numeric_series < summary.lower_bound) | (numeric_series > summary.upper_bound)
            result.mask_to_remove = mask.reindex(working_index, fill_value=False)

    elif method == "iqr":
        multiplier = params.get("multiplier", DEFAULT_METHOD_PARAMETERS["iqr"].get("multiplier", 1.5))
        summary = _summarize_iqr(numeric_series, stats, multiplier)
        applied_signal.action = "remove"
        applied_signal.lower_bound = summary.lower_bound
        applied_signal.upper_bound = summary.upper_bound
        applied_signal.affected_rows = summary.affected_rows
        applied_signal.notes = summary.notes
        if summary.affected_rows and summary.lower_bound is not None and summary.upper_bound is not None:
            mask = (numeric_series < summary.lower_bound) | (numeric_series > summary.upper_bound)
            result.mask_to_remove = mask.reindex(working_index, fill_value=False)

    elif method == "elliptic_envelope":
        contamination = params.get(
            "contamination",
            DEFAULT_METHOD_PARAMETERS["elliptic_envelope"].get("contamination", 0.01),
        )
        summary, mask = _evaluate_elliptic_envelope(numeric_series, contamination)
        applied_signal.action = "remove"
        applied_signal.lower_bound = None
        applied_signal.upper_bound = None
        applied_signal.affected_rows = summary.affected_rows
        applied_signal.notes = summary.notes
        if mask is not None and summary.affected_rows:
            result.mask_to_remove = mask.reindex(working_index, fill_value=False)

    elif method == "winsorize":
        lower_pct = params.get(
            "lower_percentile",
            DEFAULT_METHOD_PARAMETERS["winsorize"].get("lower_percentile", 5.0),
        )
        upper_pct = params.get(
            "upper_percentile",
            DEFAULT_METHOD_PARAMETERS["winsorize"].get("upper_percentile", 95.0),
        )
        summary = _summarize_winsorize(numeric_series, lower_pct, upper_pct)
        applied_signal.action = "cap"
        applied_signal.lower_bound = summary.lower_bound
        applied_signal.upper_bound = summary.upper_bound
        applied_signal.affected_rows = summary.affected_rows
        applied_signal.notes = summary.notes
        if summary.lower_bound is not None and summary.upper_bound is not None:
            clipped = numeric_series.clip(lower=summary.lower_bound, upper=summary.upper_bound)
            result.clipped_series = clipped
            result.clip_column = bool(summary.affected_rows)

    else:  # manual
        lower_bound = column_params.get("lower_bound", params.get("lower_bound"))
        upper_bound = column_params.get("upper_bound", params.get("upper_bound"))
        summary = _summarize_manual(numeric_series, lower_bound, upper_bound)
        applied_signal.action = "remove"
        applied_signal.lower_bound = summary.lower_bound
        applied_signal.upper_bound = summary.upper_bound
        applied_signal.affected_rows = summary.affected_rows
        applied_signal.notes = summary.notes
        if summary.affected_rows and (summary.lower_bound is not None or summary.upper_bound is not None):
            mask = pd.Series(False, index=numeric_series.index)
            if summary.lower_bound is not None:
                mask |= numeric_series < summary.lower_bound
            if summary.upper_bound is not None:
                mask |= numeric_series > summary.upper_bound
            result.mask_to_remove = mask.reindex(working_index, fill_value=False)

    return result


def _process_column_with_storage(
    context: ColumnContext,
    working_index: pd.Index,
    train_valid_values: Optional[pd.Series],
    stored_metadata: Optional[Dict[str, Any]],
    stored_transformer: Optional[Any],
    train_row_count: int,
) -> ColumnProcessResult:
    applied_signal = OutlierAppliedColumnSignal(
        column=context.column,
        method=context.method,
        action="remove",
        total_rows=context.total_rows,
    )
    result = ColumnProcessResult(applied_signal=applied_signal)

    method = context.method
    requires_training = method in {"zscore", "iqr", "winsorize", "elliptic_envelope"}

    if method == "zscore":
        _apply_storage_zscore(result, context, working_index, train_valid_values, stored_metadata, train_row_count)
    elif method == "iqr":
        _apply_storage_iqr(result, context, working_index, train_valid_values, stored_metadata, train_row_count)
    elif method == "winsorize":
        _apply_storage_winsorize(result, context, working_index, train_valid_values, stored_metadata, train_row_count)
    elif method == "elliptic_envelope":
        _apply_storage_elliptic(
            result,
            context,
            working_index,
            train_valid_values,
            stored_metadata,
            stored_transformer,
            train_row_count,
        )
    else:
        _apply_storage_manual(result, context, working_index, train_row_count)

    if result.transformer_status is None and requires_training and result.skip_reason is None:
        result.skip_reason = TRAINING_SKIP_MESSAGES.get(method, "Training split unavailable to fit parameters")

    if result.transformer_status is None and not result.skip_reason:
        result.skip_reason = "Unable to derive parameters"

    return result


def _apply_storage_zscore(
    result: ColumnProcessResult,
    context: ColumnContext,
    working_index: pd.Index,
    train_valid_values: Optional[pd.Series],
    stored_metadata: Optional[Dict[str, Any]],
    train_row_count: int,
) -> None:
    threshold = context.params.get("threshold", DEFAULT_METHOD_PARAMETERS["zscore"].get("threshold", 3.0))
    numeric_series = context.numeric_series

    mean = stddev = lower_bound = upper_bound = None

    if train_valid_values is not None and not train_valid_values.empty:
        bounds = _compute_zscore_bounds(train_valid_values, threshold)
        if bounds is not None:
            mean, stddev, lower_bound, upper_bound = bounds
            mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
            result.mask_to_remove = mask.reindex(working_index, fill_value=False)
            result.transformer_status = "fit"
            result.metadata_to_store = {
                "method": context.method,
                "threshold": threshold,
                "mean": mean,
                "stddev": stddev,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "train_rows": train_row_count,
                "train_valid_rows": int(train_valid_values.shape[0]),
            }
            result.skip_reason = None
            result.applied_notes.append("Fitted thresholds on training split")
        else:
            result.skip_reason = "Unable to compute z-score thresholds from training data"

    if result.transformer_status != "fit" and stored_metadata is not None:
        stored_lower = stored_metadata.get("lower_bound")
        stored_upper = stored_metadata.get("upper_bound")
        if stored_lower is not None and stored_upper is not None:
            lower_bound = float(stored_lower)
            upper_bound = float(stored_upper)
            mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
            result.mask_to_remove = mask.reindex(working_index, fill_value=False)
            result.transformer_status = "reuse"
            result.skip_reason = None
            result.applied_notes.append("Reused stored thresholds")
            threshold = float(stored_metadata.get("threshold", threshold))
            mean = stored_metadata.get("mean", mean)
            stddev = stored_metadata.get("stddev", stddev)

    if result.transformer_status is not None and result.mask_to_remove is not None:
        affected_rows = int(result.mask_to_remove.sum())
        result.applied_signal.action = "remove"
        result.applied_signal.lower_bound = lower_bound
        result.applied_signal.upper_bound = upper_bound
        result.applied_signal.affected_rows = affected_rows
        result.applied_notes.append(
            f"Threshold={float(threshold):.4f}; bounds {_format_bound(lower_bound)} to {_format_bound(upper_bound)}"
        )


def _apply_storage_iqr(
    result: ColumnProcessResult,
    context: ColumnContext,
    working_index: pd.Index,
    train_valid_values: Optional[pd.Series],
    stored_metadata: Optional[Dict[str, Any]],
    train_row_count: int,
) -> None:
    multiplier = context.params.get("multiplier", DEFAULT_METHOD_PARAMETERS["iqr"].get("multiplier", 1.5))
    numeric_series = context.numeric_series

    q1 = q3 = iqr = lower_bound = upper_bound = None

    if train_valid_values is not None and not train_valid_values.empty:
        bounds = _compute_iqr_bounds(train_valid_values, multiplier)
        if bounds is not None:
            q1, q3, iqr, lower_bound, upper_bound = bounds
            mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
            result.mask_to_remove = mask.reindex(working_index, fill_value=False)
            result.transformer_status = "fit"
            result.metadata_to_store = {
                "method": context.method,
                "multiplier": multiplier,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "train_rows": train_row_count,
                "train_valid_rows": int(train_valid_values.shape[0]),
            }
            result.skip_reason = None
            result.applied_notes.append("Fitted IQR bounds on training split")
        else:
            result.skip_reason = "Unable to compute IQR bounds from training data"

    if result.transformer_status != "fit" and stored_metadata is not None:
        stored_lower = stored_metadata.get("lower_bound")
        stored_upper = stored_metadata.get("upper_bound")
        if stored_lower is not None and stored_upper is not None:
            lower_bound = float(stored_lower)
            upper_bound = float(stored_upper)
            mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
            result.mask_to_remove = mask.reindex(working_index, fill_value=False)
            result.transformer_status = "reuse"
            result.skip_reason = None
            result.applied_notes.append("Reused stored IQR bounds")
            multiplier = float(stored_metadata.get("multiplier", multiplier))

    if result.transformer_status is not None and result.mask_to_remove is not None:
        affected_rows = int(result.mask_to_remove.sum())
        result.applied_signal.action = "remove"
        result.applied_signal.lower_bound = lower_bound
        result.applied_signal.upper_bound = upper_bound
        result.applied_signal.affected_rows = affected_rows
        result.applied_notes.append(
            f"Multiplier={float(multiplier):.4f}; bounds {_format_bound(lower_bound)} to {_format_bound(upper_bound)}"
        )


def _apply_storage_winsorize(
    result: ColumnProcessResult,
    context: ColumnContext,
    working_index: pd.Index,
    train_valid_values: Optional[pd.Series],
    stored_metadata: Optional[Dict[str, Any]],
    train_row_count: int,
) -> None:
    lower_pct = context.params.get(
        "lower_percentile",
        DEFAULT_METHOD_PARAMETERS["winsorize"].get("lower_percentile", 5.0),
    )
    upper_pct = context.params.get(
        "upper_percentile",
        DEFAULT_METHOD_PARAMETERS["winsorize"].get("upper_percentile", 95.0),
    )
    numeric_series = context.numeric_series

    lower_bound = upper_bound = None

    if train_valid_values is not None and not train_valid_values.empty:
        bounds = _compute_winsor_bounds(train_valid_values, lower_pct, upper_pct)
        if bounds is not None:
            lower_bound, upper_bound = bounds
            clipped = numeric_series.clip(lower=lower_bound, upper=upper_bound)
            result.clipped_series = clipped
            result.transformer_status = "fit"
            result.metadata_to_store = {
                "method": context.method,
                "lower_percentile": lower_pct,
                "upper_percentile": upper_pct,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "train_rows": train_row_count,
                "train_valid_rows": int(train_valid_values.shape[0]),
            }
            result.skip_reason = None
            result.applied_notes.append("Fitted winsor bounds on training split")
        else:
            result.skip_reason = "Unable to compute winsor bounds from training data"

    if result.transformer_status != "fit" and stored_metadata is not None:
        stored_lower = stored_metadata.get("lower_bound")
        stored_upper = stored_metadata.get("upper_bound")
        if stored_lower is not None and stored_upper is not None:
            lower_bound = float(stored_lower)
            upper_bound = float(stored_upper)
            clipped = numeric_series.clip(lower=lower_bound, upper=upper_bound)
            result.clipped_series = clipped
            result.transformer_status = "reuse"
            result.skip_reason = None
            result.applied_notes.append("Reused stored winsor bounds")
            lower_pct = float(stored_metadata.get("lower_percentile", lower_pct))
            upper_pct = float(stored_metadata.get("upper_percentile", upper_pct))

    if result.transformer_status is not None and result.clipped_series is not None:
        if lower_bound is not None and upper_bound is not None:
            change_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
        elif lower_bound is not None:
            change_mask = numeric_series < lower_bound
        elif upper_bound is not None:
            change_mask = numeric_series > upper_bound
        else:
            change_mask = pd.Series(False, index=numeric_series.index)
        affected_rows = int(change_mask.sum())
        result.applied_signal.action = "cap"
        result.applied_signal.lower_bound = lower_bound
        result.applied_signal.upper_bound = upper_bound
        result.applied_signal.affected_rows = affected_rows
        result.applied_notes.append(
            f"Winsorized to [{_format_bound(lower_bound)}, {_format_bound(upper_bound)}]"
        )
        result.clip_column = bool(affected_rows)


def _apply_storage_elliptic(
    result: ColumnProcessResult,
    context: ColumnContext,
    working_index: pd.Index,
    train_valid_values: Optional[pd.Series],
    stored_metadata: Optional[Dict[str, Any]],
    stored_transformer: Optional[Any],
    train_row_count: int,
) -> None:
    contamination = context.params.get(
        "contamination",
        DEFAULT_METHOD_PARAMETERS["elliptic_envelope"].get("contamination", 0.01),
    )
    numeric_series = context.numeric_series

    if train_valid_values is not None and not train_valid_values.empty:
        transformer, fit_error = _fit_elliptic_envelope_model(train_valid_values, contamination)
        if transformer is not None:
            mask, predict_error = _apply_elliptic_envelope_mask(transformer, numeric_series)
            if mask is not None:
                result.mask_to_remove = mask.reindex(working_index, fill_value=False)
                result.transformer_status = "fit"
                result.skip_reason = None
                result.transformer_to_store = transformer
                result.metadata_to_store = {
                    "method": context.method,
                    "contamination": float(contamination),
                    "train_rows": train_row_count,
                    "train_valid_rows": int(train_valid_values.shape[0]),
                }
                result.applied_notes.append("Fitted Elliptic Envelope on training split")
            else:
                result.skip_reason = predict_error or "Failed to score rows with Elliptic Envelope"
        else:
            result.skip_reason = fit_error or "Unable to fit Elliptic Envelope"

    if result.transformer_status != "fit" and stored_transformer is not None:
        mask, predict_error = _apply_elliptic_envelope_mask(stored_transformer, numeric_series)
        if mask is not None:
            result.mask_to_remove = mask.reindex(working_index, fill_value=False)
            result.transformer_status = "reuse"
            result.skip_reason = None
            result.applied_notes.append("Reused stored Elliptic Envelope model")
            contamination = float(
                stored_metadata.get("contamination", contamination) if stored_metadata else contamination
            )
        else:
            result.skip_reason = predict_error or "Stored Elliptic Envelope failed to score rows"

    if result.transformer_status is not None and result.mask_to_remove is not None:
        affected_rows = int(result.mask_to_remove.sum())
        result.applied_signal.action = "remove"
        result.applied_signal.lower_bound = None
        result.applied_signal.upper_bound = None
        result.applied_signal.affected_rows = affected_rows
        result.applied_notes.append(f"Contamination={float(contamination):.4f}")


def _apply_storage_manual(
    result: ColumnProcessResult,
    context: ColumnContext,
    working_index: pd.Index,
    train_row_count: int,
) -> None:
    numeric_series = context.numeric_series
    params = context.params
    column_params = context.column_params

    lower_bound = column_params.get("lower_bound", params.get("lower_bound"))
    upper_bound = column_params.get("upper_bound", params.get("upper_bound"))
    summary = _summarize_manual(numeric_series, lower_bound, upper_bound)

    if summary.lower_bound is None and summary.upper_bound is None:
        result.skip_reason = "Manual bounds not configured"
        return

    mask = pd.Series(False, index=numeric_series.index)
    if summary.lower_bound is not None:
        mask |= numeric_series < summary.lower_bound
    if summary.upper_bound is not None:
        mask |= numeric_series > summary.upper_bound

    result.mask_to_remove = mask.reindex(working_index, fill_value=False)
    result.transformer_status = "direct"
    result.skip_reason = None
    result.applied_signal.action = "remove"
    result.applied_signal.lower_bound = summary.lower_bound
    result.applied_signal.upper_bound = summary.upper_bound
    result.applied_signal.affected_rows = int(result.mask_to_remove.sum())
    result.applied_notes.extend(summary.notes)
    result.metadata_to_store = {
        "method": context.method,
        "lower_bound": summary.lower_bound,
        "upper_bound": summary.upper_bound,
        "train_rows": train_row_count,
    }


def _collect_outlier_candidate_columns(
    frame: pd.DataFrame,
    config: NormalizedOutlierConfig,
) -> List[str]:
    seen: set[str] = set()
    candidates: List[str] = []

    for column in config.columns:
        if column not in seen:
            seen.add(column)
            candidates.append(column)

    if config.auto_detect:
        for column in _detect_numeric_columns(frame):
            if column not in seen:
                seen.add(column)
                candidates.append(column)

    return candidates


def _prepare_outlier_context(
    column: str,
    working_frame: pd.DataFrame,
    config: NormalizedOutlierConfig,
) -> Tuple[Optional[ColumnContext], Optional[str]]:
    if column not in working_frame.columns:
        return None, f"{column} (missing)"

    prepared = _prepare_numeric_series(working_frame[column])
    if prepared is None:
        return None, f"{column} (insufficient numeric data)"

    numeric_series, stats = prepared
    total_rows = int(numeric_series.dropna().shape[0])
    if total_rows == 0:
        return None, f"{column} (no numeric rows)"

    method: OutlierMethodName = config.column_methods.get(column, config.default_method)
    if method not in OUTLIER_METHODS:
        method = OUTLIER_DEFAULT_METHOD

    params = dict(config.method_parameters.get(method, {}))
    column_params = config.column_parameters.get(column, {})
    for key, value in column_params.items():
        if key in OUTLIER_PARAMETER_KEYS[method]:
            params[key] = value

    context = ColumnContext(
        column=column,
        numeric_series=numeric_series,
        stats=stats,
        method=method,
        params=params,
        column_params=column_params,
        total_rows=total_rows,
    )

    return context, None


def _resolve_column_storage_artifacts(
    storage: Any,
    pipeline_id: Optional[str],
    node_id: Optional[Any],
    transformer_name: str,
    column: str,
    method: OutlierMethodName,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if storage is None:
        return None, None

    stored_transformer = storage.get_transformer(  # type: ignore[call-arg]
        pipeline_id=pipeline_id,
        node_id=str(node_id) if node_id is not None else None,
        transformer_name=transformer_name,
        column_name=column,
    )
    stored_metadata = storage.get_metadata(  # type: ignore[call-arg]
        pipeline_id=pipeline_id,
        node_id=str(node_id) if node_id is not None else None,
        transformer_name=transformer_name,
        column_name=column,
    )

    if not isinstance(stored_metadata, dict):
        stored_metadata = None

    if stored_metadata:
        stored_method = str(stored_metadata.get("method") or "").strip().lower()
        if stored_method and stored_method != method:
            return None, None

    return stored_transformer, stored_metadata


def _extract_train_valid_values(
    numeric_series: pd.Series,
    train_mask: Optional[pd.Series],
) -> Optional[pd.Series]:
    if train_mask is None:
        return None

    train_numeric = numeric_series.loc[train_mask]
    return train_numeric.dropna()


def _apply_column_effects(
    result: ColumnProcessResult,
    column: str,
    working_frame: pd.DataFrame,
    combined_remove_mask: pd.Series,
    applied_columns: List[OutlierAppliedColumnSignal],
    clipped_columns: List[str],
) -> None:
    if result.mask_to_remove is not None and result.applied_signal.action == "remove":
        combined_remove_mask |= result.mask_to_remove
        result.applied_signal.affected_rows = int(result.mask_to_remove.sum())

    if result.clipped_series is not None:
        working_frame.loc[result.clipped_series.index, column] = result.clipped_series

    if result.applied_notes:
        if result.applied_signal.notes:
            result.applied_signal.notes.extend(result.applied_notes)
        else:
            result.applied_signal.notes = list(result.applied_notes)

    applied_columns.append(result.applied_signal)

    if result.clip_column:
        clipped_columns.append(column)


def _record_outlier_storage_activity(
    storage: Any,
    pipeline_id: Optional[str],
    node_id: Optional[Any],
    transformer_name: str,
    column: str,
    result: ColumnProcessResult,
    train_row_count: int,
    split_counts: Dict[str, int],
) -> None:
    if storage is None or result.transformer_status is None:
        return

    if result.metadata_to_store is not None or result.transformer_to_store is not None:
        storage.store_transformer(  # type: ignore[call-arg]
            pipeline_id=pipeline_id,
            node_id=str(node_id) if node_id is not None else None,
            transformer_name=transformer_name,
            transformer=result.transformer_to_store,
            column_name=column,
            metadata=result.metadata_to_store,
        )

    train_action = "not_available"
    if result.transformer_status == "fit":
        train_action = "fit_transform" if train_row_count > 0 else "fit"
    elif result.transformer_status in {"reuse", "direct"}:
        train_action = "transform" if train_row_count > 0 else "not_available"

    storage.record_split_activity(  # type: ignore[call-arg]
        pipeline_id=pipeline_id,
        node_id=str(node_id) if node_id is not None else None,
        transformer_name=transformer_name,
        column_name=column,
        split_name="train",
        action=train_action,
        row_count=train_row_count,
    )

    for split_name in ("test", "validation"):
        rows_processed = int(split_counts.get(split_name, 0))
        action = "transform" if rows_processed > 0 else "not_available"
        storage.record_split_activity(  # type: ignore[call-arg]
            pipeline_id=pipeline_id,
            node_id=str(node_id) if node_id is not None else None,
            transformer_name=transformer_name,
            column_name=column,
            split_name=split_name,
            action=action,
            row_count=rows_processed,
        )


def _apply_outlier_removal(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, OutlierNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None
    signal = OutlierNodeSignal(node_id=str(node_id) if node_id is not None else None)

    if frame.empty:
        return frame, "Outlier removal: no data available", signal

    data = node.get("data") if isinstance(node, dict) else None
    config_payload = data.get("config") if isinstance(data, dict) else {}
    config = _normalize_outlier_config(config_payload)

    signal.configured_columns = list(config.columns)
    signal.default_method = config.default_method
    signal.column_methods = dict(config.column_methods)
    signal.auto_detect = config.auto_detect

    skipped_set = set(config.skipped_columns)
    if skipped_set:
        signal.skipped_columns.extend(f"{column} (skipped)" for column in sorted(skipped_set))

    candidate_columns = _collect_outlier_candidate_columns(frame, config)
    signal.evaluated_columns = list(candidate_columns)

    if not candidate_columns:
        return frame, "Outlier removal: no numeric columns selected", signal

    working_frame = frame.copy()
    combined_remove_mask = pd.Series(False, index=working_frame.index, dtype=bool)
    applied_columns: List[OutlierAppliedColumnSignal] = []
    clipped_columns: List[str] = []

    has_splits = SPLIT_TYPE_COLUMN in working_frame.columns
    storage = get_pipeline_store() if pipeline_id and has_splits else None
    split_counts: Dict[str, int] = (
        working_frame[SPLIT_TYPE_COLUMN].value_counts().to_dict() if has_splits else {}
    )
    train_mask = (working_frame[SPLIT_TYPE_COLUMN] == "train") if has_splits else None
    train_row_count = int(split_counts.get("train", 0)) if has_splits else 0
    transformer_name = "outlier_removal"

    for column in candidate_columns:
        if column in skipped_set:
            continue

        context, skip_message = _prepare_outlier_context(column, working_frame, config)
        if skip_message:
            signal.skipped_columns.append(skip_message)
            continue
        assert context is not None  # appease type checkers

        if storage is None:
            result = _process_column_without_storage(context, working_frame.index)
        else:
            stored_transformer, stored_metadata = _resolve_column_storage_artifacts(
                storage,
                pipeline_id,
                node_id,
                transformer_name,
                column,
                context.method,
            )
            train_valid_values = _extract_train_valid_values(context.numeric_series, train_mask)

            result = _process_column_with_storage(
                context,
                working_frame.index,
                train_valid_values,
                stored_metadata,
                stored_transformer,
                train_row_count,
            )

        if result.skip_reason:
            signal.skipped_columns.append(f"{column} ({result.skip_reason})")
            continue

        _apply_column_effects(
            result,
            column,
            working_frame,
            combined_remove_mask,
            applied_columns,
            clipped_columns,
        )

        if storage is not None:
            _record_outlier_storage_activity(
                storage,
                pipeline_id,
                node_id,
                transformer_name,
                column,
                result,
                train_row_count,
                split_counts,
            )

    removed_rows = int(combined_remove_mask.sum())
    if removed_rows:
        working_frame = working_frame.loc[~combined_remove_mask].copy()

    signal.applied_columns = applied_columns
    signal.removed_rows = removed_rows
    signal.clipped_columns = sorted(set(clipped_columns))

    summary_parts: List[str] = []
    if removed_rows:
        summary_parts.append(f"removed {removed_rows} row(s)")
    if clipped_columns:
        summary_parts.append(f"winsorized {len(set(clipped_columns))} column(s)")
    if not summary_parts:
        summary_parts.append("no changes applied")

    summary = "Outlier removal: " + "; ".join(summary_parts)

    return working_frame, summary, signal


__all__ = [
    "OUTLIER_METHODS",
    "OUTLIER_METHOD_ORDER",
    "OUTLIER_DEFAULT_METHOD",
    "DEFAULT_METHOD_PARAMETERS",
    "_outlier_method_details",
    "_build_outlier_recommendations",
    "_apply_outlier_removal",
    "_normalize_outlier_config",
]

