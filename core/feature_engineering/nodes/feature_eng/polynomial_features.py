"""Polynomial feature expansion node helpers."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN
from core.feature_engineering.schemas import (
    PolynomialFeaturesNodeSignal,
    PolynomialGeneratedFeature,
)
from core.feature_engineering.pipeline_store_singleton import get_pipeline_store

from ...shared.utils import (
    _auto_detect_numeric_columns,
    _coerce_config_boolean,
    _coerce_string_list,
)

try:  # pragma: no cover - optional dependency guard
    from sklearn.preprocessing import PolynomialFeatures  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - defensive guard
    PolynomialFeatures = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from sklearn.preprocessing import PolynomialFeatures as PolynomialFeaturesClass
else:  # pragma: no cover - runtime fallback when sklearn missing
    PolynomialFeaturesClass = Any

logger = logging.getLogger(__name__)

POLYNOMIAL_MIN_DEGREE = 2
POLYNOMIAL_MAX_DEGREE = 5
POLYNOMIAL_DEFAULT_DEGREE = 2
POLYNOMIAL_DEFAULT_PREFIX = "poly"
TRANSFORMER_NAME = "polynomial_features"


@dataclass
class NormalizedPolynomialConfig:
    columns: List[str]
    auto_detect: bool
    degree: int
    include_bias: bool
    interaction_only: bool
    include_input_features: bool
    output_prefix: str


@dataclass
class _NumericPreparation:
    working_frame: pd.DataFrame
    numeric_frame: pd.DataFrame
    valid_columns: List[str]
    column_summary: str
    method_label: str
    filled_columns: Dict[str, int]


@dataclass
class _TransformerResolution:
    transformer: Any
    metadata: Dict[str, Any]
    transform_mode: str
    storage: Optional[Any]
    has_splits: bool
    split_counts: Dict[str, int]


def _normalize_polynomial_config(config: Any) -> NormalizedPolynomialConfig:
    if not isinstance(config, dict):
        config = {}

    columns = _coerce_string_list(config.get("columns"))
    auto_detect = _coerce_config_boolean(config.get("auto_detect"), default=False)

    raw_degree = config.get("degree")
    try:
        numeric = float(raw_degree)
        if math.isnan(numeric):
            raise ValueError
        degree = int(round(numeric))
    except (TypeError, ValueError):
        degree = POLYNOMIAL_DEFAULT_DEGREE
    degree = max(POLYNOMIAL_MIN_DEGREE, min(POLYNOMIAL_MAX_DEGREE, degree))

    include_bias = _coerce_config_boolean(config.get("include_bias"), default=False)
    interaction_only = _coerce_config_boolean(config.get("interaction_only"), default=False)
    include_input_features = _coerce_config_boolean(config.get("include_input_features"), default=False)

    raw_prefix = str(config.get("output_prefix") or POLYNOMIAL_DEFAULT_PREFIX).strip()
    sanitized_prefix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw_prefix)
    if not sanitized_prefix:
        sanitized_prefix = POLYNOMIAL_DEFAULT_PREFIX

    unique_columns: List[str] = []
    seen: set[str] = set()
    for column in columns:
        normalized = column.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_columns.append(normalized)

    return NormalizedPolynomialConfig(
        columns=unique_columns,
        auto_detect=auto_detect,
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only,
        include_input_features=include_input_features,
        output_prefix=sanitized_prefix,
    )


def _build_method_label(config: NormalizedPolynomialConfig) -> str:
    detail_flags: List[str] = []
    if config.interaction_only:
        detail_flags.append("interactions only")
    if config.include_input_features:
        detail_flags.append("keeps originals")
    if config.include_bias:
        detail_flags.append("bias term")

    if detail_flags:
        return f"PolynomialFeatures (degree={config.degree}; {'; '.join(detail_flags)})"
    return f"PolynomialFeatures (degree={config.degree})"


def _format_terms(powers: Sequence[int], input_columns: Sequence[str]) -> List[str]:
    terms: List[str] = []
    for exponent, column in zip(powers, input_columns):
        if int(exponent) <= 0:
            continue
        exponent_int = int(exponent)
        if exponent_int == 1:
            terms.append(str(column))
        else:
            terms.append(f"{column}^{exponent_int}")
    return terms


def _sanitize_feature_name(
    raw_name: str,
    *,
    degree: int,
    prefix: str,
    occupied: set[str],
) -> str:
    if degree == 0:
        base = "bias"
    else:
        base = (
            raw_name.replace(" ", "_x_")
            .replace("*", "_x_")
            .replace("^", "_pow_")
            .replace("__", "_")
            .strip("_")
        )
        if not base:
            base = "feature"

    candidate_base = f"{prefix}__{base}" if prefix else base
    candidate = candidate_base
    suffix = 2
    while candidate in occupied:
        candidate = f"{candidate_base}_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def _prepare_feature_plan(
    transformer: PolynomialFeaturesClass,
    *,
    input_columns: List[str],
    config: NormalizedPolynomialConfig,
    occupied: Iterable[str],
) -> Tuple[List[int], List[str], List[Dict[str, Any]]]:
    try:
        powers = getattr(transformer, "powers_")
        feature_names = transformer.get_feature_names_out(input_columns)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("PolynomialFeatures transformer missing feature metadata", exc_info=exc)
        return [], [], []

    occupied_set = set(str(name) for name in occupied)
    selected_indices: List[int] = []
    output_columns: List[str] = []
    summaries: List[Dict[str, Any]] = []

    for index, (raw_name, power_row) in enumerate(zip(feature_names, powers)):
        degree = int(np.sum(power_row))
        if degree == 0 and not config.include_bias:
            continue
        if degree == 1 and not config.include_input_features:
            continue

        terms = _format_terms(power_row, input_columns)
        expression = " * ".join(terms) if terms else "1"
        column_name = _sanitize_feature_name(
            raw_name,
            degree=degree,
            prefix=config.output_prefix,
            occupied=occupied_set,
        )

        selected_indices.append(int(index))
        output_columns.append(column_name)
        summaries.append(
            {
                "column": column_name,
                "degree": degree,
                "terms": terms,
                "expression": expression,
                "raw_feature": str(raw_name),
            }
        )

    return selected_indices, output_columns, summaries


def _extract_plan_from_metadata(
    metadata: Optional[Dict[str, Any]]
) -> Tuple[List[int], List[str], List[Dict[str, Any]]]:
    if not isinstance(metadata, dict):
        return [], [], []

    raw_indices = metadata.get("selected_indices")
    raw_outputs = metadata.get("output_columns")
    raw_summaries = metadata.get("feature_summaries")

    try:
        indices = [int(idx) for idx in raw_indices] if isinstance(raw_indices, Iterable) else []
        outputs = [str(name) for name in raw_outputs] if isinstance(raw_outputs, Iterable) else []
    except Exception:  # pragma: no cover - defensive guard
        indices, outputs = [], []

    summaries: List[Dict[str, Any]] = []
    if isinstance(raw_summaries, Iterable):
        for entry in raw_summaries:
            if isinstance(entry, dict):
                summaries.append({
                    "column": str(entry.get("column", "")),
                    "degree": int(entry.get("degree", 0)),
                    "terms": [str(term) for term in entry.get("terms", []) if str(term).strip()],
                    "expression": str(entry.get("expression", "")),
                    "raw_feature": str(entry.get("raw_feature", "")),
                })

    if indices and outputs and len(indices) == len(outputs):
        return indices, outputs, summaries

    return [], [], []


def _collect_candidate_columns(
    frame: pd.DataFrame,
    config: NormalizedPolynomialConfig,
    signal: PolynomialFeaturesNodeSignal,
) -> Tuple[Optional[List[str]], Optional[str]]:
    candidate_columns: List[str] = []
    seen: set[str] = set()

    for column in config.columns:
        normalized = column.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            candidate_columns.append(normalized)

    if config.auto_detect:
        for column in _auto_detect_numeric_columns(frame):
            if column not in seen:
                seen.add(column)
                candidate_columns.append(column)

    signal.evaluated_columns = list(candidate_columns)

    if candidate_columns:
        return candidate_columns, None

    message = "Polynomial features: no numeric columns selected"
    signal.notes.append(message)
    return None, message


def _prepare_numeric_inputs(
    frame: pd.DataFrame,
    candidate_columns: List[str],
    config: NormalizedPolynomialConfig,
    signal: PolynomialFeaturesNodeSignal,
) -> Tuple[Optional[_NumericPreparation], Optional[str]]:
    missing_reasons: List[str] = []
    valid_columns: List[str] = []

    for column in candidate_columns:
        if column not in frame.columns:
            missing_reasons.append(f"{column} (missing)")
            continue
        numeric_series = pd.to_numeric(frame[column], errors="coerce")
        if numeric_series.dropna().empty:
            missing_reasons.append(f"{column} (no numeric values)")
            continue
        valid_columns.append(column)

    if missing_reasons:
        signal.skipped_columns.extend(missing_reasons)

    if not valid_columns:
        message = "Polynomial features: no numeric data available"
        signal.notes.append(message)
        return None, message

    working_frame = frame.copy()
    numeric_frame = working_frame[valid_columns].apply(pd.to_numeric, errors="coerce")

    filled_columns: Dict[str, int] = {}
    for column in valid_columns:
        missing_count = int(numeric_frame[column].isna().sum())
        if missing_count:
            filled_columns[column] = missing_count

    if filled_columns:
        numeric_frame = numeric_frame.fillna(0.0)
        signal.filled_columns = filled_columns

    signal.applied_columns = list(valid_columns)

    prep = _NumericPreparation(
        working_frame=working_frame,
        numeric_frame=numeric_frame,
        valid_columns=list(valid_columns),
        column_summary=", ".join(valid_columns),
        method_label=_build_method_label(config),
        filled_columns=filled_columns,
    )

    return prep, None


def _resolve_transformer(
    prep: _NumericPreparation,
    config: NormalizedPolynomialConfig,
    signal: PolynomialFeaturesNodeSignal,
    *,
    pipeline_id: Optional[str],
    node_id: Optional[Any],
) -> Tuple[Optional[_TransformerResolution], Optional[str]]:
    has_splits = SPLIT_TYPE_COLUMN in prep.working_frame.columns
    storage = get_pipeline_store() if pipeline_id and has_splits else None
    split_counts: Dict[str, int] = {}
    transformer: Optional[Any] = None
    metadata: Dict[str, Any] = {}
    transform_mode = "stateless"

    if has_splits and storage is not None and pipeline_id is not None:
        node_identifier = str(node_id) if node_id is not None else None
        if node_identifier is None:
            warning = "Polynomial features skipped: missing node identifier for stored transformer reuse"
            signal.notes.append(warning)
            logger.info(warning)
            return None, warning

        split_counts = prep.working_frame[SPLIT_TYPE_COLUMN].value_counts().to_dict()
        train_mask = prep.working_frame[SPLIT_TYPE_COLUMN] == "train"
        train_rows = int(split_counts.get("train", 0))
        train_numeric = prep.numeric_frame.loc[train_mask]

        stored_transformer = storage.get_transformer(
            pipeline_id=pipeline_id,
            node_id=node_identifier,
            transformer_name=TRANSFORMER_NAME,
        )
        stored_metadata = storage.get_metadata(
            pipeline_id=pipeline_id,
            node_id=node_identifier,
            transformer_name=TRANSFORMER_NAME,
        )

        stored_indices, stored_outputs, stored_summaries = _extract_plan_from_metadata(stored_metadata)

        if train_rows > 0 and not train_numeric.empty:
            transformer = PolynomialFeatures(
                degree=config.degree,
                interaction_only=config.interaction_only,
                include_bias=config.include_bias,
                order="C",
            )
            transformer.fit(train_numeric.values)
            transform_mode = "fit"

            selected_indices, output_columns, feature_summaries = _prepare_feature_plan(
                transformer,
                input_columns=prep.valid_columns,
                config=config,
                occupied=prep.working_frame.columns,
            )

            metadata = {
                "input_columns": list(prep.valid_columns),
                "selected_indices": selected_indices,
                "output_columns": output_columns,
                "feature_summaries": feature_summaries,
                "degree": config.degree,
                "include_bias": config.include_bias,
                "interaction_only": config.interaction_only,
                "include_input_features": config.include_input_features,
                "output_prefix": config.output_prefix,
                "column_summary": prep.column_summary,
                "method_label": prep.method_label,
                "method": "PolynomialFeatures",
            }

            storage.store_transformer(
                pipeline_id=pipeline_id,
                node_id=node_identifier,
                transformer_name=TRANSFORMER_NAME,
                transformer=transformer,
                metadata=metadata,
            )
        elif stored_transformer is not None:
            transformer = stored_transformer
            metadata = dict(stored_metadata or {})
            transform_mode = "reuse"

            if not stored_indices or not stored_outputs:
                selected_indices, output_columns, feature_summaries = _prepare_feature_plan(
                    transformer,
                    input_columns=prep.valid_columns,
                    config=config,
                    occupied=prep.working_frame.columns,
                )
                metadata.update(
                    {
                        "input_columns": list(prep.valid_columns),
                        "selected_indices": selected_indices,
                        "output_columns": output_columns,
                        "feature_summaries": feature_summaries,
                    }
                )
                metadata.setdefault("column_summary", prep.column_summary)
                metadata.setdefault("method_label", prep.method_label)
                metadata.setdefault("method", "PolynomialFeatures")

                storage.store_transformer(
                    pipeline_id=pipeline_id,
                    node_id=node_identifier,
                    transformer_name=TRANSFORMER_NAME,
                    transformer=transformer,
                    metadata=metadata,
                )
        else:
            warning = "Polynomial features skipped: no training rows available to fit transformer"
            signal.notes.append(warning)
            logger.info(warning)
            return None, warning
    else:
        transformer = PolynomialFeatures(
            degree=config.degree,
            interaction_only=config.interaction_only,
            include_bias=config.include_bias,
            order="C",
        )
        transformer.fit(prep.numeric_frame.values)
        transform_mode = "fit"
        selected_indices, output_columns, feature_summaries = _prepare_feature_plan(
            transformer,
            input_columns=prep.valid_columns,
            config=config,
            occupied=prep.working_frame.columns,
        )
        metadata = {
            "input_columns": list(prep.valid_columns),
            "selected_indices": selected_indices,
            "output_columns": output_columns,
            "feature_summaries": feature_summaries,
            "degree": config.degree,
            "include_bias": config.include_bias,
            "interaction_only": config.interaction_only,
            "include_input_features": config.include_input_features,
            "output_prefix": config.output_prefix,
            "column_summary": prep.column_summary,
            "method_label": prep.method_label,
            "method": "PolynomialFeatures",
        }

    if transformer is None:
        message = "Polynomial features: transformer unavailable"
        signal.notes.append(message)
        return None, message

    resolution = _TransformerResolution(
        transformer=transformer,
        metadata=metadata,
        transform_mode=transform_mode,
        storage=storage,
        has_splits=has_splits,
        split_counts=split_counts,
    )

    return resolution, None


def apply_polynomial_features(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, PolynomialFeaturesNodeSignal]:
    """Expand numeric columns using sklearn's PolynomialFeatures transformer."""

    node_id = node.get("id") if isinstance(node, dict) else None

    signal = PolynomialFeaturesNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
    )

    if frame.empty:
        signal.notes.append("Polynomial features: no data available")
        return frame, "Polynomial features: no data available", signal

    if PolynomialFeatures is None:  # pragma: no cover - optional dependency
        warning = "Polynomial features skipped: scikit-learn is not installed"
        logger.warning(warning)
        signal.notes.append(warning)
        return frame, "Polynomial features: scikit-learn unavailable", signal

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_polynomial_config(config_payload)

    signal.configured_columns = list(config.columns)
    signal.auto_detect = config.auto_detect
    signal.degree = config.degree
    signal.include_bias = config.include_bias
    signal.interaction_only = config.interaction_only
    signal.include_input_features = config.include_input_features
    signal.output_prefix = config.output_prefix

    candidate_columns, error_message = _collect_candidate_columns(frame, config, signal)
    if error_message:
        return frame, error_message, signal
    assert candidate_columns is not None

    prep, error_message = _prepare_numeric_inputs(frame, candidate_columns, config, signal)
    if error_message:
        return frame, error_message, signal
    assert prep is not None

    resolution, error_message = _resolve_transformer(
        prep,
        config,
        signal,
        pipeline_id=pipeline_id,
        node_id=node_id,
    )
    if error_message:
        return frame, error_message, signal
    assert resolution is not None

    transformer = resolution.transformer
    metadata = resolution.metadata

    selected_indices, output_columns, feature_summaries = _extract_plan_from_metadata(metadata)
    if not selected_indices or not output_columns:
        selected_indices, output_columns, feature_summaries = _prepare_feature_plan(
            transformer,
            input_columns=prep.valid_columns,
            config=config,
            occupied=prep.working_frame.columns,
        )
        if not selected_indices:
            message = "Polynomial features: no derived features generated"
            signal.notes.append(message)
            return frame, message, signal

        metadata.update(
            {
                "input_columns": list(prep.valid_columns),
                "selected_indices": selected_indices,
                "output_columns": output_columns,
                "feature_summaries": feature_summaries,
            }
        )
        if resolution.storage is not None:
            resolution.storage.store_transformer(
                pipeline_id=pipeline_id,
                node_id=str(node_id) if node_id is not None else None,
                transformer_name=TRANSFORMER_NAME,
                transformer=transformer,
                metadata=metadata,
            )

    transformed_matrix = transformer.transform(prep.numeric_frame.values)
    selected_matrix = transformed_matrix[:, selected_indices]

    feature_frame = pd.DataFrame(
        selected_matrix,
        index=prep.working_frame.index,
        columns=output_columns,
    )
    working_frame = pd.concat([prep.working_frame, feature_frame], axis=1)

    signal.generated_columns = list(output_columns)
    signal.generated_features = [PolynomialGeneratedFeature(**summary) for summary in feature_summaries]
    signal.feature_count = len(output_columns)
    signal.transform_mode = resolution.transform_mode

    if resolution.storage is not None and resolution.has_splits:
        train_rows = int(resolution.split_counts.get("train", 0))
        train_action = (
            "fit_transform" if resolution.transform_mode == "fit" and train_rows > 0 else "transform"
        )
        resolution.storage.record_split_activity(
            pipeline_id=pipeline_id,
            node_id=str(node_id),
            transformer_name=TRANSFORMER_NAME,
            split_name="train",
            action=train_action if train_rows > 0 else "not_available",
            row_count=train_rows if train_rows > 0 else None,
        )
        for split_name in ("validation", "test"):
            rows = int(resolution.split_counts.get(split_name, 0))
            resolution.storage.record_split_activity(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name=TRANSFORMER_NAME,
                split_name=split_name,
                action="transform" if rows > 0 else "not_available",
                row_count=rows if rows > 0 else None,
            )

    summary = (
        f"Polynomial features: added {len(output_columns)} column"
        f"{'s' if len(output_columns) != 1 else ''} "
        f"(degree={config.degree}, columns={', '.join(prep.valid_columns)})"
    )
    signal.notes.append(summary)

    return working_frame, summary, signal
