"""Binning helpers for feature engineering nodes."""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, cast, Literal

import pandas as pd
from pandas.api import types as pd_types
from sklearn.preprocessing import KBinsDiscretizer

from core.feature_engineering.schemas import (
    BinnedColumnBin,
    BinnedColumnDistribution,
    BinningAppliedColumnSignal,
    BinningColumnRecommendation,
    BinningColumnStats,
    BinningExcludedColumn,
    BinningNodeSignal,
)
from core.feature_engineering.pipeline_store_singleton import get_pipeline_store

from ...shared.utils import (
    _coerce_config_boolean,
    _detect_numeric_columns,
    _format_interval_value,
    _is_binary_numeric,
)

logger = logging.getLogger(__name__)

# Import split detection constant
SPLIT_TYPE_COLUMN = "__split_type__"

BINNING_STRATEGIES: Set[str] = {"equal_width", "equal_frequency", "custom", "kbins"}
BINNING_DUPLICATE_MODES: Set[str] = {"raise", "drop"}
BINNING_LABEL_FORMATS: Set[str] = {"range", "bin_index", "ordinal", "column_suffix"}
BINNING_MISSING_STRATEGIES: Set[str] = {"keep", "label"}
KBINS_STRATEGIES: Set[str] = {"uniform", "quantile", "kmeans"}
KBINS_ENCODE_TYPES: Set[str] = {"ordinal", "onehot", "onehot-dense"}

BINNING_DEFAULT_EQUAL_WIDTH_BINS = 5
BINNING_DEFAULT_EQUAL_FREQUENCY_BINS = 4
BINNING_DEFAULT_PRECISION = 3
BINNING_DEFAULT_SUFFIX = "_binned"
BINNING_DEFAULT_MISSING_LABEL = "Missing"
KBINS_DEFAULT_N_BINS = 5
KBINS_DEFAULT_ENCODE = "ordinal"
KBINS_DEFAULT_STRATEGY = "quantile"


@dataclass
class ColumnBinningOverride:
    strategy: Optional[str] = None
    equal_width_bins: Optional[int] = None
    equal_frequency_bins: Optional[int] = None
    kbins_n_bins: Optional[int] = None
    kbins_encode: Optional[str] = None
    kbins_strategy: Optional[str] = None
    custom_bins: Optional[List[float]] = None
    custom_labels: Optional[List[str]] = None


@dataclass
class NormalizedBinningConfig:
    strategy: str
    columns: List[str]
    equal_width_bins: int
    equal_frequency_bins: int
    include_lowest: bool
    precision: int
    duplicates: str
    output_suffix: str
    drop_original: bool
    label_format: str
    missing_strategy: str
    missing_label: Optional[str]
    custom_bins: Dict[str, List[float]]
    custom_labels: Dict[str, List[str]]
    column_strategies: Dict[str, ColumnBinningOverride]
    # KBinsDiscretizer specific
    kbins_n_bins: int
    kbins_encode: str
    kbins_strategy: str


def _resolve_binning_strategy(value: Any) -> str:
    raw_strategy = str(value or "").strip().lower()
    return raw_strategy if raw_strategy in BINNING_STRATEGIES else "equal_width"


def _normalize_columns(raw_columns: Any) -> List[str]:
    if isinstance(raw_columns, list):
        seen: Set[str] = set()
        columns: List[str] = []
        for entry in raw_columns:
            column = str(entry or "").strip()
            if column and column not in seen:
                seen.add(column)
                columns.append(column)
        return columns
    if isinstance(raw_columns, str):
        columns = []
        for segment in raw_columns.split(","):
            column = segment.strip()
            if column and column not in columns:
                columns.append(column)
        return columns
    return []


def _clamp_integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if not math.isfinite(numeric):
        numeric = default
    integer = int(round(numeric))
    if integer < minimum:
        integer = minimum
    if integer > maximum:
        integer = maximum
    return integer


def _sanitize_number_list(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)):
        return []
    sanitized: List[float] = []
    for entry in value:
        try:
            numeric = float(entry)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        sanitized.append(numeric)
    if len(sanitized) < 2:
        return []
    ordered = sorted(set(sanitized))
    if len(ordered) < 2:
        return []
    return ordered


def _parse_custom_bins(raw_custom_bins: Any) -> Dict[str, List[float]]:
    custom_bins: Dict[str, List[float]] = {}
    if not isinstance(raw_custom_bins, dict):
        return custom_bins
    for key, value in raw_custom_bins.items():
        column = str(key or "").strip()
        if not column:
            continue
        bins = _sanitize_number_list(value)
        if bins:
            custom_bins[column] = bins
    return custom_bins


def _parse_label_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(entry).strip() for entry in value if str(entry).strip()]
    if isinstance(value, str):
        return [segment.strip() for segment in value.split(",") if segment.strip()]
    return []


def _parse_custom_labels(raw_custom_labels: Any) -> Dict[str, List[str]]:
    custom_labels: Dict[str, List[str]] = {}
    if not isinstance(raw_custom_labels, dict):
        return custom_labels
    for key, value in raw_custom_labels.items():
        column = str(key or "").strip()
        if not column:
            continue
        labels = _parse_label_list(value)
        if labels:
            custom_labels[column] = labels
    return custom_labels


def _coerce_optional_int(value: Any, minimum: int, maximum: int) -> Optional[int]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    integer = int(round(numeric))
    if integer < minimum:
        integer = minimum
    if integer > maximum:
        integer = maximum
    return integer


def _normalize_column_override(value: Any) -> Optional[ColumnBinningOverride]:
    if not isinstance(value, dict):
        return None

    override = ColumnBinningOverride()

    raw_strategy = str(value.get("strategy") or "").strip().lower()
    if raw_strategy in BINNING_STRATEGIES:
        override.strategy = raw_strategy

    if "equal_width_bins" in value:
        coerced = _coerce_optional_int(value.get("equal_width_bins"), 2, 200)
        if coerced is not None:
            override.equal_width_bins = coerced

    if "equal_frequency_bins" in value:
        coerced = _coerce_optional_int(value.get("equal_frequency_bins"), 2, 200)
        if coerced is not None:
            override.equal_frequency_bins = coerced

    if "kbins_n_bins" in value:
        coerced = _coerce_optional_int(value.get("kbins_n_bins"), 2, 200)
        if coerced is not None:
            override.kbins_n_bins = coerced

    raw_kbins_encode = str(value.get("kbins_encode") or "").strip().lower()
    if raw_kbins_encode in KBINS_ENCODE_TYPES:
        override.kbins_encode = raw_kbins_encode

    raw_kbins_strategy = str(value.get("kbins_strategy") or "").strip().lower()
    if raw_kbins_strategy in KBINS_STRATEGIES:
        override.kbins_strategy = raw_kbins_strategy

    override.custom_bins = _sanitize_number_list(value.get("custom_bins")) or None
    labels = _parse_label_list(value.get("custom_labels"))
    if labels:
        override.custom_labels = labels

    if (
        override.strategy is None
        and override.equal_width_bins is None
        and override.equal_frequency_bins is None
        and override.kbins_n_bins is None
        and override.kbins_encode is None
        and override.kbins_strategy is None
        and not override.custom_bins
        and not override.custom_labels
    ):
        return None

    return override


def _parse_column_overrides(raw_overrides: Any) -> Dict[str, ColumnBinningOverride]:
    column_strategies: Dict[str, ColumnBinningOverride] = {}
    if not isinstance(raw_overrides, dict):
        return column_strategies
    for key, value in raw_overrides.items():
        column = str(key or "").strip()
        if not column:
            continue
        override = _normalize_column_override(value)
        if override is not None:
            column_strategies[column] = override
    return column_strategies


def _normalize_binning_config(config: Any) -> NormalizedBinningConfig:
    if not isinstance(config, dict):
        config = {}

    strategy = _resolve_binning_strategy(config.get("strategy"))
    columns = _normalize_columns(config.get("columns"))

    equal_width_bins = _clamp_integer(
        config.get("equal_width_bins"),
        BINNING_DEFAULT_EQUAL_WIDTH_BINS,
        2,
        200,
    )
    equal_frequency_bins = _clamp_integer(
        config.get("equal_frequency_bins"),
        BINNING_DEFAULT_EQUAL_FREQUENCY_BINS,
        2,
        200,
    )
    precision = _clamp_integer(config.get("precision"), BINNING_DEFAULT_PRECISION, 0, 8)

    duplicates = str(config.get("duplicates") or "").strip().lower()
    if duplicates not in BINNING_DUPLICATE_MODES:
        duplicates = "raise"

    include_lowest = _coerce_config_boolean(config.get("include_lowest"), default=True)
    drop_original = _coerce_config_boolean(config.get("drop_original"), default=False)

    output_suffix = str(config.get("output_suffix") or BINNING_DEFAULT_SUFFIX).strip()
    if not output_suffix:
        output_suffix = BINNING_DEFAULT_SUFFIX

    label_format = str(config.get("label_format") or "").strip().lower()
    if label_format not in BINNING_LABEL_FORMATS:
        label_format = "range"

    missing_strategy = str(config.get("missing_strategy") or "").strip().lower()
    if missing_strategy not in BINNING_MISSING_STRATEGIES:
        missing_strategy = "keep"

    missing_label: Optional[str] = None
    if missing_strategy == "label":
        raw_label = str(config.get("missing_label") or "").strip()
        missing_label = raw_label or BINNING_DEFAULT_MISSING_LABEL

    # KBinsDiscretizer parameters
    kbins_n_bins = _clamp_integer(
        config.get("kbins_n_bins"),
        KBINS_DEFAULT_N_BINS,
        2,
        200,
    )

    kbins_encode = str(config.get("kbins_encode") or "").strip().lower()
    if kbins_encode not in KBINS_ENCODE_TYPES:
        kbins_encode = KBINS_DEFAULT_ENCODE

    kbins_strategy = str(config.get("kbins_strategy") or "").strip().lower()
    if kbins_strategy not in KBINS_STRATEGIES:
        kbins_strategy = KBINS_DEFAULT_STRATEGY

    custom_bins = _parse_custom_bins(config.get("custom_bins"))
    custom_labels = _parse_custom_labels(config.get("custom_labels"))
    raw_column_strategies = config.get("column_strategies") or config.get("column_overrides")
    column_strategies = _parse_column_overrides(raw_column_strategies)

    return NormalizedBinningConfig(
        strategy=strategy,
        columns=columns,
        equal_width_bins=equal_width_bins,
        equal_frequency_bins=equal_frequency_bins,
        include_lowest=include_lowest,
        precision=precision,
        duplicates=duplicates,
        output_suffix=output_suffix,
        drop_original=drop_original,
        label_format=label_format,
        missing_strategy=missing_strategy,
        missing_label=missing_label,
        custom_bins=custom_bins,
        custom_labels=custom_labels,
        column_strategies=column_strategies,
        kbins_n_bins=kbins_n_bins,
        kbins_encode=kbins_encode,
        kbins_strategy=kbins_strategy,
    )


def _build_binning_labels(
    column: str,
    config: NormalizedBinningConfig,
    bin_count: int,
) -> Optional[List[str]]:
    if bin_count <= 0:
        return None
    label_format = config.label_format
    if label_format == "range":
        return None
    if label_format == "bin_index":
        return [f"bin_{index}" for index in range(bin_count)]
    if label_format == "ordinal":
        return [f"Bin {index + 1}" for index in range(bin_count)]
    if label_format == "column_suffix":
        base = config.output_suffix or BINNING_DEFAULT_SUFFIX
        return [f"{column}{base}_{index + 1}" for index in range(bin_count)]
    return None


@dataclass
class _SplitInfo:
    has_splits: bool
    train_mask: Optional[pd.Series]
    test_mask: Optional[pd.Series]
    validation_mask: Optional[pd.Series]


@dataclass
class _ColumnProcessingOutcome:
    frame: pd.DataFrame
    detail: Optional[str]
    skipped_reason: Optional[str]


@dataclass
class _StrategyContext:
    column: str
    new_column: str
    config: NormalizedBinningConfig
    numeric_series: pd.Series
    valid_values: pd.Series
    override: Optional[ColumnBinningOverride]
    working_frame: pd.DataFrame
    signal: BinningNodeSignal
    split_info: _SplitInfo
    resolved_custom_bins: Dict[str, List[float]]
    resolved_custom_labels: Dict[str, List[str]]
    pipeline_id: Optional[str]
    node_id: Optional[str]


def _compute_split_info(frame: pd.DataFrame) -> _SplitInfo:
    has_splits = SPLIT_TYPE_COLUMN in frame.columns
    if not has_splits:
        return _SplitInfo(False, None, None, None)
    train_mask = frame[SPLIT_TYPE_COLUMN] == "train"
    test_mask = frame[SPLIT_TYPE_COLUMN] == "test"
    validation_mask = frame[SPLIT_TYPE_COLUMN] == "validation"
    return _SplitInfo(True, train_mask, test_mask, validation_mask)


def _prepare_custom_assets(
    config: NormalizedBinningConfig,
    signal: BinningNodeSignal,
) -> Tuple[Dict[str, List[float]], Dict[str, List[str]]]:
    resolved_custom_bins: Dict[str, List[float]] = {
        key: list(values) for key, values in config.custom_bins.items()
    }
    resolved_custom_labels: Dict[str, List[str]] = {
        key: list(values) for key, values in config.custom_labels.items()
    }

    if config.column_strategies:
        for column, override in config.column_strategies.items():
            if override.custom_bins:
                resolved_custom_bins[column] = list(override.custom_bins)
            if override.custom_labels:
                resolved_custom_labels[column] = list(override.custom_labels)

        overrides_payload: Dict[str, Dict[str, Any]] = {}
        for column, override in config.column_strategies.items():
            payload: Dict[str, Any] = {}
            if override.strategy:
                payload["strategy"] = override.strategy
            if override.equal_width_bins is not None:
                payload["equal_width_bins"] = override.equal_width_bins
            if override.equal_frequency_bins is not None:
                payload["equal_frequency_bins"] = override.equal_frequency_bins
            if override.kbins_n_bins is not None:
                payload["kbins_n_bins"] = override.kbins_n_bins
            if override.kbins_encode is not None:
                payload["kbins_encode"] = override.kbins_encode
            if override.kbins_strategy is not None:
                payload["kbins_strategy"] = override.kbins_strategy
            if override.custom_bins:
                payload["custom_bins"] = list(override.custom_bins)
            if override.custom_labels:
                payload["custom_labels"] = list(override.custom_labels)
            if payload:
                overrides_payload[column] = payload
        if overrides_payload:
            signal.column_overrides = overrides_payload

    if resolved_custom_bins:
        signal.custom_bins = resolved_custom_bins
    if resolved_custom_labels:
        signal.custom_labels = resolved_custom_labels

    return resolved_custom_bins, resolved_custom_labels


def _process_binning_column(
    column: str,
    working_frame: pd.DataFrame,
    config: NormalizedBinningConfig,
    signal: BinningNodeSignal,
    split_info: _SplitInfo,
    resolved_custom_bins: Dict[str, List[float]],
    resolved_custom_labels: Dict[str, List[str]],
    pipeline_id: Optional[str],
    node_id: Optional[str],
) -> _ColumnProcessingOutcome:
    signal.evaluated_columns.append(column)

    if column not in working_frame.columns:
        message = f"{column} (missing)"
        signal.skipped_columns.append(message)
        return _ColumnProcessingOutcome(working_frame, None, message)

    raw_series = working_frame[column]
    if pd_types.is_bool_dtype(raw_series):
        message = f"{column} (boolean column automatically excluded)"
        signal.skipped_columns.append(message)
        return _ColumnProcessingOutcome(working_frame, None, message)

    numeric_series = pd.to_numeric(raw_series, errors="coerce")

    if split_info.has_splits and split_info.train_mask is not None:
        train_numeric = numeric_series[split_info.train_mask]
        valid_values = train_numeric.dropna()
    else:
        valid_values = numeric_series.dropna()

    if int(valid_values.size) < 2:
        message = f"{column} (insufficient numeric values)"
        signal.skipped_columns.append(message)
        return _ColumnProcessingOutcome(working_frame, None, message)

    if _is_binary_numeric(valid_values):
        message = f"{column} (binary indicator column excluded)"
        signal.skipped_columns.append(message)
        return _ColumnProcessingOutcome(working_frame, None, message)

    override = config.column_strategies.get(column)
    column_strategy = config.strategy
    if override and override.strategy in BINNING_STRATEGIES:
        column_strategy = override.strategy

    context = _StrategyContext(
        column=column,
        new_column=f"{column}{config.output_suffix}",
        config=config,
        numeric_series=numeric_series,
        valid_values=valid_values,
        override=override,
        working_frame=working_frame,
        signal=signal,
        split_info=split_info,
        resolved_custom_bins=resolved_custom_bins,
        resolved_custom_labels=resolved_custom_labels,
        pipeline_id=pipeline_id,
        node_id=node_id,
    )

    try:
        updated_frame, detail = _apply_strategy_to_column(column_strategy, context)
    except ValueError as error:
        message = f"{column} ({error})"
        signal.skipped_columns.append(message)
        return _ColumnProcessingOutcome(working_frame, None, message)
    except Exception as error:  # pragma: no cover - log unexpected issues
        logger.error("[BINNING] Exception processing column %s: %s", column, error, exc_info=True)
        message = f"{column} ({error})"
        signal.skipped_columns.append(message)
        return _ColumnProcessingOutcome(working_frame, None, message)

    if context.config.drop_original and column in updated_frame.columns:
        updated_frame = updated_frame.drop(columns=[column])

    return _ColumnProcessingOutcome(updated_frame, detail, None)


def _apply_strategy_to_column(
    column_strategy: str,
    context: _StrategyContext,
) -> Tuple[pd.DataFrame, Optional[str]]:
    if column_strategy == "equal_width":
        return _apply_equal_width_strategy(context)
    if column_strategy == "equal_frequency":
        return _apply_equal_frequency_strategy(context)
    if column_strategy == "custom":
        return _apply_custom_strategy(context)
    return _apply_kbins_strategy(context)


def _get_split_row_counts(split_info: _SplitInfo) -> Tuple[int, int, int]:
    if not split_info.has_splits:
        return 0, 0, 0
    train_rows = int(split_info.train_mask.sum()) if split_info.train_mask is not None else 0
    test_rows = int(split_info.test_mask.sum()) if split_info.test_mask is not None else 0
    validation_rows = (
        int(split_info.validation_mask.sum()) if split_info.validation_mask is not None else 0
    )
    return train_rows, test_rows, validation_rows


def _record_split_activity(
    storage,
    pipeline_id: str,
    node_id: str,
    transformer_name: str,
    column: str,
    split_info: _SplitInfo,
    train_rows: int,
    test_rows: int,
    validation_rows: int,
    *,
    train_action: str,
) -> None:
    storage.record_split_activity(
        pipeline_id=pipeline_id,
        node_id=node_id,
        transformer_name=transformer_name,
        column_name=column,
        split_name="train",
        action=train_action,
        row_count=train_rows,
    )
    if test_rows > 0:
        storage.record_split_activity(
            pipeline_id=pipeline_id,
            node_id=node_id,
            transformer_name=transformer_name,
            column_name=column,
            split_name="test",
            action="transform",
            row_count=test_rows,
        )
    if validation_rows > 0:
        storage.record_split_activity(
            pipeline_id=pipeline_id,
            node_id=node_id,
            transformer_name=transformer_name,
            column_name=column,
            split_name="validation",
            action="transform",
            row_count=validation_rows,
        )


def _check_kbins_warnings(captured):
    for warning in captured:
        message = str(warning.message)
        if "Bins whose width are too small" in message:
            raise ValueError("kbins degenerate bins; reduce requested bins")


def _fit_kbins_with_warning_check(discretizer, values):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        discretizer.fit(values)
    _check_kbins_warnings(caught)


def _transform_kbins_with_warning_check(discretizer, values):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        transformed = discretizer.transform(values)
    _check_kbins_warnings(caught)
    return transformed


def _apply_equal_width_strategy(context: _StrategyContext) -> Tuple[pd.DataFrame, Optional[str]]:
    config = context.config
    override = context.override

    requested_bins = max(
        2,
        override.equal_width_bins if override and override.equal_width_bins else config.equal_width_bins,
    )
    min_val = context.valid_values.min()
    max_val = context.valid_values.max()
    bin_intervals = pd.interval_range(
        start=min_val,
        end=max_val,
        periods=requested_bins,
        closed="right",
    )
    bin_edges = [min_val] + [interval.right for interval in bin_intervals]

    binned_series = pd.cut(
        context.numeric_series,
        bins=bin_edges,
        labels=None,
        include_lowest=config.include_lowest,
        precision=config.precision,
        duplicates=config.duplicates,
    )

    original_categories = list(binned_series.cat.categories)
    if not original_categories:
        raise ValueError("unable to derive bins")

    label_override = _build_binning_labels(context.column, config, len(original_categories))
    if label_override is not None and len(label_override) == len(original_categories):
        binned_series = binned_series.cat.rename_categories(label_override)

    actual_bins = len(binned_series.cat.categories)
    reduced_bins = bool(requested_bins and actual_bins < requested_bins)

    converted = binned_series.astype("object")
    if config.missing_strategy == "label":
        fill_value = config.missing_label or BINNING_DEFAULT_MISSING_LABEL
        converted = converted.fillna(fill_value)

    def _coerce_value(value: Any) -> Optional[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if isinstance(value, pd.Interval):
            return _format_interval_value(value, config.precision)
        return str(value)

    formatted_series = converted.map(_coerce_value)
    context.working_frame[context.new_column] = formatted_series

    distribution = _build_binned_distribution(
        context.new_column,
        context.working_frame[context.new_column],
        source_column=context.column,
        missing_label=config.missing_label if config.missing_strategy == "label" else None,
    )

    context.signal.applied_columns.append(
        BinningAppliedColumnSignal(
            source_column=context.column,
            output_column=context.new_column,
            strategy="equal_width",
            requested_bins=requested_bins,
            actual_bins=actual_bins,
            reduced_bins=reduced_bins,
            drop_original=config.drop_original,
            include_lowest=config.include_lowest,
            precision=config.precision,
            duplicates=config.duplicates,
            label_format=config.label_format,
            missing_strategy=config.missing_strategy,
            missing_label=config.missing_label if config.missing_strategy == "label" else None,
            custom_labels_applied=bool(label_override),
            sample_bins=list(distribution.bins[:5]) if distribution else [],
        )
    )

    detail = (
        f"{context.column}→{context.new_column} (equal-width, {actual_bins} bin"
        f"{'s' if actual_bins != 1 else ''})"
    )
    if reduced_bins:
        detail = f"{detail} (requested {requested_bins})"

    if (
        context.pipeline_id
        and context.node_id
        and context.split_info.has_splits
    ):
        storage = get_pipeline_store()
        train_rows, test_rows, validation_rows = _get_split_row_counts(context.split_info)
        bin_edges_payload = (
            [interval.left for interval in original_categories] + [original_categories[-1].right]
            if original_categories
            else []
        )
        storage.store_transformer(
            pipeline_id=context.pipeline_id,
            node_id=context.node_id,
            transformer_name="binning_equal_width",
            transformer={
                "type": "pandas_binning",
                "strategy": "equal_width",
                "bin_edges": bin_edges_payload,
                "categories": [str(cat) for cat in original_categories],
            },
            column_name=context.column,
            metadata={
                "method": "equal_width",
                "method_label": f"Equal-width ({actual_bins} bins)",
                "n_bins": actual_bins,
                "requested_bins": requested_bins,
                "train_rows": train_rows,
            },
        )
        _record_split_activity(
            storage,
            context.pipeline_id,
            context.node_id,
            "binning_equal_width",
            context.column,
            context.split_info,
            train_rows,
            test_rows,
            validation_rows,
            train_action="transform",
        )

    return context.working_frame, detail


def _apply_equal_frequency_strategy(context: _StrategyContext) -> Tuple[pd.DataFrame, Optional[str]]:
    config = context.config
    override = context.override

    requested_bins = max(
        2,
        override.equal_frequency_bins if override and override.equal_frequency_bins else config.equal_frequency_bins,
    )
    quantiles = [i / requested_bins for i in range(requested_bins + 1)]
    bin_edges = context.valid_values.quantile(quantiles).unique().tolist()
    if len(bin_edges) < 2:
        raise ValueError("unable to create bins")

    binned_series = pd.cut(
        context.numeric_series,
        bins=bin_edges,
        labels=None,
        include_lowest=True,
        precision=config.precision,
        duplicates=config.duplicates,
    )

    original_categories = list(binned_series.cat.categories)
    if not original_categories:
        raise ValueError("unable to derive bins")

    label_override = _build_binning_labels(context.column, config, len(original_categories))
    if label_override is not None and len(label_override) == len(original_categories):
        binned_series = binned_series.cat.rename_categories(label_override)

    actual_bins = len(binned_series.cat.categories)
    reduced_bins = bool(requested_bins and actual_bins < requested_bins)

    converted = binned_series.astype("object")
    if config.missing_strategy == "label":
        fill_value = config.missing_label or BINNING_DEFAULT_MISSING_LABEL
        converted = converted.fillna(fill_value)

    def _coerce_value(value: Any) -> Optional[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if isinstance(value, pd.Interval):
            return _format_interval_value(value, config.precision)
        return str(value)

    formatted_series = converted.map(_coerce_value)
    context.working_frame[context.new_column] = formatted_series

    distribution = _build_binned_distribution(
        context.new_column,
        context.working_frame[context.new_column],
        source_column=context.column,
        missing_label=config.missing_label if config.missing_strategy == "label" else None,
    )

    context.signal.applied_columns.append(
        BinningAppliedColumnSignal(
            source_column=context.column,
            output_column=context.new_column,
            strategy="equal_frequency",
            requested_bins=requested_bins,
            actual_bins=actual_bins,
            reduced_bins=reduced_bins,
            drop_original=config.drop_original,
            include_lowest=True,
            precision=config.precision,
            duplicates=config.duplicates,
            label_format=config.label_format,
            missing_strategy=config.missing_strategy,
            missing_label=config.missing_label if config.missing_strategy == "label" else None,
            custom_labels_applied=bool(label_override),
            sample_bins=list(distribution.bins[:5]) if distribution else [],
        )
    )

    detail = (
        f"{context.column}→{context.new_column} (equal-frequency, {actual_bins} bin"
        f"{'s' if actual_bins != 1 else ''})"
    )
    if reduced_bins:
        detail = f"{detail} (requested {requested_bins})"

    if (
        context.pipeline_id
        and context.node_id
        and context.split_info.has_splits
    ):
        storage = get_pipeline_store()
        train_rows, test_rows, validation_rows = _get_split_row_counts(context.split_info)
        bin_edges_payload = (
            [interval.left for interval in original_categories] + [original_categories[-1].right]
            if original_categories
            else []
        )
        storage.store_transformer(
            pipeline_id=context.pipeline_id,
            node_id=context.node_id,
            transformer_name="binning_equal_frequency",
            transformer={
                "type": "pandas_binning",
                "strategy": "equal_frequency",
                "bin_edges": bin_edges_payload,
                "categories": [str(cat) for cat in original_categories],
            },
            column_name=context.column,
            metadata={
                "method": "equal_frequency",
                "method_label": f"Equal-frequency ({actual_bins} bins)",
                "n_bins": actual_bins,
                "requested_bins": requested_bins,
                "train_rows": train_rows,
            },
        )
        _record_split_activity(
            storage,
            context.pipeline_id,
            context.node_id,
            "binning_equal_frequency",
            context.column,
            context.split_info,
            train_rows,
            test_rows,
            validation_rows,
            train_action="transform",
        )

    return context.working_frame, detail


def _apply_custom_strategy(context: _StrategyContext) -> Tuple[pd.DataFrame, Optional[str]]:
    config = context.config
    edges = context.resolved_custom_bins.get(context.column)
    if not edges or len(edges) < 2:
        raise ValueError("custom bins missing")

    requested_bins = len(edges) - 1
    binned_series = pd.cut(
        context.numeric_series,
        bins=edges,
        labels=None,
        include_lowest=config.include_lowest,
        precision=config.precision,
        duplicates=config.duplicates,
    )

    original_categories = list(binned_series.cat.categories)
    if not original_categories:
        raise ValueError("unable to derive bins")

    custom_labels = context.resolved_custom_labels.get(context.column)
    label_override: Optional[List[str]] = None
    if custom_labels and len(custom_labels) >= len(original_categories):
        label_override = custom_labels[: len(original_categories)]
    else:
        label_override = _build_binning_labels(context.column, config, len(original_categories))

    if label_override is not None and len(label_override) == len(original_categories):
        binned_series = binned_series.cat.rename_categories(label_override)

    actual_bins = len(binned_series.cat.categories)

    converted = binned_series.astype("object")
    if config.missing_strategy == "label":
        fill_value = config.missing_label or BINNING_DEFAULT_MISSING_LABEL
        converted = converted.fillna(fill_value)

    def _coerce_value(value: Any) -> Optional[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if isinstance(value, pd.Interval):
            return _format_interval_value(value, config.precision)
        return str(value)

    formatted_series = converted.map(_coerce_value)
    context.working_frame[context.new_column] = formatted_series

    distribution = _build_binned_distribution(
        context.new_column,
        context.working_frame[context.new_column],
        source_column=context.column,
        missing_label=config.missing_label if config.missing_strategy == "label" else None,
    )

    context.signal.applied_columns.append(
        BinningAppliedColumnSignal(
            source_column=context.column,
            output_column=context.new_column,
            strategy="custom",
            requested_bins=requested_bins,
            actual_bins=actual_bins,
            reduced_bins=False,
            drop_original=config.drop_original,
            include_lowest=config.include_lowest,
            precision=config.precision,
            duplicates=config.duplicates,
            label_format=config.label_format,
            missing_strategy=config.missing_strategy,
            missing_label=config.missing_label if config.missing_strategy == "label" else None,
            custom_labels_applied=bool(label_override),
            sample_bins=list(distribution.bins[:5]) if distribution else [],
        )
    )

    detail = (
        f"{context.column}→{context.new_column} (custom, {actual_bins} bin"
        f"{'s' if actual_bins != 1 else ''})"
    )

    if (
        context.pipeline_id
        and context.node_id
        and context.split_info.has_splits
    ):
        storage = get_pipeline_store()
        train_rows, test_rows, validation_rows = _get_split_row_counts(context.split_info)
        bin_edges_payload = (
            [interval.left for interval in original_categories] + [original_categories[-1].right]
            if original_categories
            else []
        )
        storage.store_transformer(
            pipeline_id=context.pipeline_id,
            node_id=context.node_id,
            transformer_name="binning_custom",
            transformer={
                "type": "pandas_binning",
                "strategy": "custom",
                "bin_edges": bin_edges_payload,
                "categories": [str(cat) for cat in original_categories],
            },
            column_name=context.column,
            metadata={
                "method": "custom",
                "method_label": f"Custom ({actual_bins} bins)",
                "n_bins": actual_bins,
                "requested_bins": requested_bins,
                "train_rows": train_rows,
            },
        )
        _record_split_activity(
            storage,
            context.pipeline_id,
            context.node_id,
            "binning_custom",
            context.column,
            context.split_info,
            train_rows,
            test_rows,
            validation_rows,
            train_action="transform",
        )

    return context.working_frame, detail


def _apply_kbins_strategy(context: _StrategyContext) -> Tuple[pd.DataFrame, Optional[str]]:
    config = context.config
    override = context.override

    column_kbins_n_bins = (
        override.kbins_n_bins if override and override.kbins_n_bins else config.kbins_n_bins
    )
    column_kbins_strategy = (
        override.kbins_strategy if override and override.kbins_strategy else config.kbins_strategy
    )
    column_kbins_encode = (
        override.kbins_encode if override and override.kbins_encode else config.kbins_encode
    )

    requested_bins = max(2, column_kbins_n_bins)
    discretizer_kwargs: Dict[str, Any] = {
        "n_bins": requested_bins,
        "encode": column_kbins_encode,
        "strategy": column_kbins_strategy,
        "subsample": None,
    }
    if column_kbins_strategy == "quantile":
        discretizer_kwargs["quantile_method"] = "averaged_inverted_cdf"

    discretizer = KBinsDiscretizer(**discretizer_kwargs)

    if context.split_info.has_splits and context.split_info.train_mask is not None:
        train_data = context.numeric_series[context.split_info.train_mask].dropna()
        if len(train_data) < requested_bins:
            raise ValueError(f"insufficient train samples for {requested_bins} bins")
        _fit_kbins_with_warning_check(discretizer, train_data.values.reshape(-1, 1))
    else:
        all_data = context.numeric_series.dropna()
        if len(all_data) < requested_bins:
            raise ValueError(f"insufficient samples for {requested_bins} bins")
        _fit_kbins_with_warning_check(discretizer, all_data.values.reshape(-1, 1))

    transformed_col = context.numeric_series.copy()
    non_na_mask = ~context.numeric_series.isna()
    if non_na_mask.any():
        transformed_values = _transform_kbins_with_warning_check(
            discretizer,
            context.numeric_series[non_na_mask].values.reshape(-1, 1),
        )
        if column_kbins_encode in ["onehot", "onehot-dense"]:
            transformed_values = transformed_values.argmax(axis=1).reshape(-1, 1)
        transformed_col[non_na_mask] = transformed_values.flatten()

    if config.missing_strategy == "label":
        fill_value = config.missing_label or BINNING_DEFAULT_MISSING_LABEL
        transformed_col = transformed_col.fillna(fill_value)

    if column_kbins_encode == "ordinal":
        transformed_col = transformed_col.astype("Int64")

    context.working_frame[context.new_column] = transformed_col

    actual_bins = int(discretizer.n_bins_[0]) if hasattr(discretizer, "n_bins_") else requested_bins

    distribution = _build_binned_distribution(
        context.new_column,
        context.working_frame[context.new_column],
        source_column=context.column,
        missing_label=config.missing_label if config.missing_strategy == "label" else None,
    )

    context.signal.applied_columns.append(
        BinningAppliedColumnSignal(
            source_column=context.column,
            output_column=context.new_column,
            strategy="kbins",
            requested_bins=requested_bins,
            actual_bins=actual_bins,
            reduced_bins=False,
            drop_original=config.drop_original,
            include_lowest=True,
            precision=0,
            duplicates="raise",
            label_format=column_kbins_encode,
            missing_strategy=config.missing_strategy,
            missing_label=config.missing_label if config.missing_strategy == "label" else None,
            custom_labels_applied=False,
            sample_bins=list(distribution.bins[:5]) if distribution else [],
        )
    )

    detail = (
        f"{context.column}→{context.new_column} (kbins {column_kbins_strategy}, {actual_bins} bin"
        f"{'s' if actual_bins != 1 else ''})"
    )

    if (
        context.pipeline_id
        and context.node_id
        and context.split_info.has_splits
    ):
        storage = get_pipeline_store()
        train_rows, test_rows, validation_rows = _get_split_row_counts(context.split_info)
        storage.store_transformer(
            pipeline_id=context.pipeline_id,
            node_id=context.node_id,
            transformer_name="kbins_discretizer",
            transformer=discretizer,
            column_name=context.column,
            metadata={
                "method": column_kbins_strategy,
                "method_label": f"KBins ({column_kbins_strategy})",
                "n_bins": requested_bins,
                "encode": column_kbins_encode,
                "train_rows": train_rows,
            },
        )
        _record_split_activity(
            storage,
            context.pipeline_id,
            context.node_id,
            "kbins_discretizer",
            context.column,
            context.split_info,
            train_rows,
            test_rows,
            validation_rows,
            train_action="fit_transform",
        )

    return context.working_frame, detail


def _build_binning_signal_from_config(
    config: NormalizedBinningConfig,
    node_identifier: Optional[str],
) -> BinningNodeSignal:
    strategy_literal = cast(
        Literal["equal_width", "equal_frequency", "custom", "kbins"],
        config.strategy,
    )
    signal = BinningNodeSignal(
        node_id=node_identifier,
        strategy=strategy_literal,
        configured_columns=list(dict.fromkeys(config.columns)),
        drop_original=config.drop_original,
        include_lowest=config.include_lowest,
        precision=config.precision,
        duplicates=config.duplicates,
        label_format=config.label_format,
        missing_strategy=config.missing_strategy,
        missing_label=config.missing_label,
    )
    signal.equal_width_bins = config.equal_width_bins
    signal.equal_frequency_bins = config.equal_frequency_bins
    return signal


def _resolve_split_info(frame: pd.DataFrame) -> _SplitInfo:
    has_splits = SPLIT_TYPE_COLUMN in frame.columns
    if not has_splits:
        return _SplitInfo(
            has_splits=False,
            train_mask=None,
            test_mask=None,
            validation_mask=None,
        )

    train_mask = frame[SPLIT_TYPE_COLUMN] == "train"
    test_mask = frame[SPLIT_TYPE_COLUMN] == "test"
    validation_mask = frame[SPLIT_TYPE_COLUMN] == "validation"

    return _SplitInfo(
        has_splits=True,
        train_mask=train_mask,
        test_mask=test_mask,
        validation_mask=validation_mask,
    )


def _prepare_custom_binning_resources(
    config: NormalizedBinningConfig,
) -> Tuple[
    Dict[str, List[float]],
    Dict[str, List[str]],
    Dict[str, Dict[str, Any]],
]:
    resolved_custom_bins: Dict[str, List[float]] = {
        key: list(values) for key, values in config.custom_bins.items()
    }
    resolved_custom_labels: Dict[str, List[str]] = {
        key: list(values) for key, values in config.custom_labels.items()
    }

    overrides_payload: Dict[str, Dict[str, Any]] = {}

    if not config.column_strategies:
        return resolved_custom_bins, resolved_custom_labels, overrides_payload

    for column, override in config.column_strategies.items():
        if override.custom_bins:
            resolved_custom_bins[column] = list(override.custom_bins)
        if override.custom_labels:
            resolved_custom_labels[column] = list(override.custom_labels)

    for column, override in config.column_strategies.items():
        payload: Dict[str, Any] = {}
        if override.strategy:
            payload["strategy"] = override.strategy
        if override.equal_width_bins is not None:
            payload["equal_width_bins"] = override.equal_width_bins
        if override.equal_frequency_bins is not None:
            payload["equal_frequency_bins"] = override.equal_frequency_bins
        if override.kbins_n_bins is not None:
            payload["kbins_n_bins"] = override.kbins_n_bins
        if override.kbins_encode is not None:
            payload["kbins_encode"] = override.kbins_encode
        if override.kbins_strategy is not None:
            payload["kbins_strategy"] = override.kbins_strategy
        if override.custom_bins:
            payload["custom_bins"] = list(override.custom_bins)
        if override.custom_labels:
            payload["custom_labels"] = list(override.custom_labels)
        if payload:
            overrides_payload[column] = payload

    return resolved_custom_bins, resolved_custom_labels, overrides_payload


def _apply_column_processing(
    config: NormalizedBinningConfig,
    working_frame: pd.DataFrame,
    signal: BinningNodeSignal,
    split_info: _SplitInfo,
    resolved_custom_bins: Dict[str, List[float]],
    resolved_custom_labels: Dict[str, List[str]],
    pipeline_id: Optional[str],
    node_identifier: Optional[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    transformed_details: List[str] = []
    skipped_columns: List[str] = []

    for column in config.columns:
        outcome = _process_binning_column(
            column=column,
            working_frame=working_frame,
            config=config,
            signal=signal,
            split_info=split_info,
            resolved_custom_bins=resolved_custom_bins,
            resolved_custom_labels=resolved_custom_labels,
            pipeline_id=pipeline_id,
            node_id=node_identifier,
        )

        working_frame = outcome.frame

        if outcome.skipped_reason:
            skipped_columns.append(outcome.skipped_reason)
            continue

        if outcome.detail:
            transformed_details.append(outcome.detail)

    return working_frame, transformed_details, skipped_columns


def _finalize_signal_lists(signal: BinningNodeSignal) -> None:
    if signal.evaluated_columns:
        signal.evaluated_columns = list(dict.fromkeys(signal.evaluated_columns))
    if signal.skipped_columns:
        signal.skipped_columns = list(dict.fromkeys(signal.skipped_columns))


def _apply_binning_discretization(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, BinningNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None
    node_identifier = str(node_id) if node_id is not None else None

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_binning_config(config_payload)

    signal = _build_binning_signal_from_config(config, node_identifier)

    if frame.empty:
        return frame, "Binning / Discretization: no rows available", signal

    if not config.columns:
        return frame, "Binning / Discretization: no columns selected", signal

    split_info = _resolve_split_info(frame)
    working_frame = frame.copy()
    resolved_custom_bins, resolved_custom_labels, overrides_payload = _prepare_custom_binning_resources(config)

    if overrides_payload:
        signal.column_overrides = overrides_payload
    if resolved_custom_bins:
        signal.custom_bins = resolved_custom_bins
    if resolved_custom_labels:
        signal.custom_labels = resolved_custom_labels

    working_frame, transformed_details, skipped_columns = _apply_column_processing(
        config=config,
        working_frame=working_frame,
        signal=signal,
        split_info=split_info,
        resolved_custom_bins=resolved_custom_bins,
        resolved_custom_labels=resolved_custom_labels,
        pipeline_id=pipeline_id,
        node_identifier=node_identifier,
    )

    _finalize_signal_lists(signal)

    if not transformed_details:
        summary = "Binning / Discretization: no columns transformed"
        if skipped_columns:
            summary = f"{summary}; skipped {', '.join(skipped_columns)}"
        return working_frame, summary, signal

    summary = f"Binning / Discretization: {', '.join(transformed_details)}"
    if skipped_columns:
        summary = f"{summary}; skipped {', '.join(skipped_columns)}"
    return working_frame, summary, signal


def _build_binned_distribution(
    column: str,
    series: pd.Series,
    *,
    source_column: Optional[str] = None,
    missing_label: Optional[str] = None,
) -> Optional[BinnedColumnDistribution]:
    total_rows = int(series.shape[0]) if series is not None else 0
    if total_rows <= 0:
        return None

    working_series = series.copy()

    value_counts = working_series.value_counts(dropna=False)

    bins: List[BinnedColumnBin] = []
    missing_rows = int(working_series.isna().sum())
    non_missing_rows = total_rows - missing_rows
    distinct_bins = int(working_series.dropna().nunique()) if non_missing_rows > 0 else 0

    entries: List[BinnedColumnBin] = []

    for value, count in value_counts.items():
        if pd.isna(value):
            continue
        try:
            numeric_count = int(count)
        except (TypeError, ValueError):
            continue
        if numeric_count < 0:
            continue
        percentage = float(numeric_count) / float(total_rows) * 100.0 if total_rows else 0.0
        percentage = round(percentage, 4)
        label = str(value)
        entries.append(
            BinnedColumnBin(
                label=label,
                count=numeric_count,
                percentage=percentage,
                is_missing=False,
            )
        )

    entries.sort(
        key=lambda item: (
            item.is_missing,
            -item.count,
            item.label.lower(),
        )
    )

    bins.extend(entries)

    if missing_rows > 0:
        missing_percentage = float(missing_rows) / float(total_rows) * 100.0 if total_rows else 0.0
        missing_percentage = round(missing_percentage, 4)
        bins.append(
            BinnedColumnBin(
                label=(missing_label or BINNING_DEFAULT_MISSING_LABEL),
                count=missing_rows,
                percentage=missing_percentage,
                is_missing=True,
            )
        )

    if not bins:
        return None

    top_label: Optional[str] = None
    top_count: Optional[int] = None
    top_percentage: Optional[float] = None

    for entry in bins:
        if entry.is_missing:
            continue
        if top_count is None or entry.count > top_count:
            top_label = entry.label
            top_count = entry.count
            top_percentage = entry.percentage
    if top_count is None and bins:
        fallback = bins[0]
        top_label = fallback.label
        top_count = fallback.count
        top_percentage = fallback.percentage

    return BinnedColumnDistribution(
        column=column,
        source_column=source_column,
        total_rows=total_rows,
        non_missing_rows=non_missing_rows,
        missing_rows=missing_rows,
        distinct_bins=distinct_bins,
        top_label=top_label,
        top_count=top_count,
        top_percentage=top_percentage,
        bins=bins,
    )
def _build_binning_recommendations(
    frame: pd.DataFrame,
) -> Tuple[List[BinningColumnRecommendation], List[BinningExcludedColumn]]:
    recommendations: List[BinningColumnRecommendation] = []
    excluded: List[BinningExcludedColumn] = []

    if frame is None or frame.empty:
        return recommendations, excluded

    for column in frame.columns:
        column_name = str(column)
        if not column_name:
            continue

        series = frame[column]
        dtype = str(series.dtype)

        if pd_types.is_bool_dtype(series):
            excluded.append(
                BinningExcludedColumn(
                    column=column_name,
                    reason="Boolean dtype",
                    dtype=dtype,
                )
            )
            continue

        numeric_series = pd.to_numeric(series, errors="coerce")
        valid = numeric_series.dropna()
        valid_count = int(valid.size)
        missing_count = int(series.size - valid_count)

        if valid_count < 2:
            excluded.append(
                BinningExcludedColumn(
                    column=column_name,
                    reason="Insufficient numeric samples",
                    dtype=dtype,
                )
            )
            continue

        if _is_binary_numeric(valid):
            excluded.append(
                BinningExcludedColumn(
                    column=column_name,
                    reason="Binary indicator column",
                    dtype=dtype,
                )
            )
            continue

        distinct_count = int(valid.nunique()) if valid_count else 0
        if distinct_count < 2:
            excluded.append(
                BinningExcludedColumn(
                    column=column_name,
                    reason="Single unique numeric value",
                    dtype=dtype,
                )
            )
            continue

        minimum = float(valid.min()) if not valid.empty else None
        maximum = float(valid.max()) if not valid.empty else None
        mean = float(valid.mean()) if not valid.empty else None
        median = float(valid.median()) if not valid.empty else None
        stddev = float(valid.std()) if valid_count > 1 else None
        skewness = float(valid.skew()) if valid_count > 2 else None

        has_negative = bool((valid < 0).any())
        has_zero = bool((valid == 0).any())
        has_positive = bool((valid > 0).any())

        stats = BinningColumnStats(
            valid_count=valid_count,
            missing_count=missing_count,
            distinct_count=distinct_count,
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            median=median,
            stddev=stddev,
            skewness=skewness,
            has_negative=has_negative,
            has_zero=has_zero,
            has_positive=has_positive,
        )

        reasons: List[str] = []
        notes: List[str] = []

        abs_skew = abs(skewness) if skewness is not None and math.isfinite(skewness) else 0.0
        numeric_range = None
        if minimum is not None and maximum is not None and math.isfinite(minimum) and math.isfinite(maximum):
            numeric_range = maximum - minimum

        if distinct_count <= 5:
            strategy: Literal["equal_width", "equal_frequency", "kbins"] = "equal_frequency"
            recommended_bins = max(2, distinct_count)
            reasons.append("Limited unique numeric values in the sample")
        elif distinct_count >= 50:
            strategy = "kbins"
            recommended_bins = max(6, min(15, distinct_count // 5 or 6))
            reasons.append("High-cardinality numeric distribution")
            notes.append("KBinsDiscretizer can better adapt to dense ranges")
        elif abs_skew >= 1.0:
            strategy = "equal_frequency"
            recommended_bins = max(4, min(10, distinct_count // 2 or 4))
            reasons.append(f"Skewed distribution detected (|skew|≈{abs_skew:.2f})")
        else:
            strategy = "equal_width"
            recommended_bins = 5 if distinct_count >= 10 else max(3, min(6, distinct_count))
            reasons.append("Continuous numeric span with manageable skew")

        if numeric_range is not None and numeric_range == 0:
            recommended_bins = 2
            reasons.append("Observed zero numeric range; reducing bin count")

        recommended_bins = max(2, min(200, recommended_bins))

        if has_negative and has_positive:
            notes.append("Column includes both negative and positive values")
        elif has_negative:
            notes.append("Column includes negative values only")
        elif has_positive and has_zero:
            notes.append("Column includes zeros and positive values")

        if missing_count > 0:
            notes.append(f"{missing_count} missing value{'s' if missing_count != 1 else ''} observed")

        if valid_count >= 250:
            confidence: Literal["high", "medium", "low"] = "high"
        elif valid_count >= 75:
            confidence = "medium"
        else:
            confidence = "low"

        reasons.append(f"Evaluated on {valid_count} numeric sample{'s' if valid_count != 1 else ''}")

        recommendations.append(
            BinningColumnRecommendation(
                column=column_name,
                dtype=dtype,
                recommended_strategy=strategy,
                recommended_bins=recommended_bins,
                confidence=confidence,
                reasons=reasons,
                notes=notes,
                stats=stats,
            )
        )

    recommendations.sort(key=lambda item: item.column.lower())
    excluded.sort(key=lambda item: item.column.lower())
    return recommendations, excluded


__all__ = [
    "BINNING_STRATEGIES",
    "BINNING_DUPLICATE_MODES",
    "BINNING_LABEL_FORMATS",
    "BINNING_MISSING_STRATEGIES",
    "BINNING_DEFAULT_EQUAL_WIDTH_BINS",
    "BINNING_DEFAULT_EQUAL_FREQUENCY_BINS",
    "BINNING_DEFAULT_PRECISION",
    "BINNING_DEFAULT_SUFFIX",
    "BINNING_DEFAULT_MISSING_LABEL",
    "NormalizedBinningConfig",
    "_normalize_binning_config",
    "_build_binning_labels",
    "_apply_binning_discretization",
    "_build_binned_distribution",
    "_is_binary_numeric",
    "_detect_numeric_columns",
    "_build_binning_recommendations",
]
