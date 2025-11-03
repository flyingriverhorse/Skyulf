"""Binning helpers for feature engineering nodes."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, cast, Literal

import pandas as pd
from pandas.api import types as pd_types
from sklearn.preprocessing import KBinsDiscretizer

logger = logging.getLogger(__name__)

from core.feature_engineering.schemas import (
    BinnedColumnBin,
    BinnedColumnDistribution,
    BinningAppliedColumnSignal,
    BinningNodeSignal,
)

from .utils import _coerce_config_boolean, _format_interval_value
from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store

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
    # KBinsDiscretizer specific
    kbins_n_bins: int
    kbins_encode: str
    kbins_strategy: str


def _normalize_binning_config(config: Any) -> NormalizedBinningConfig:
    if not isinstance(config, dict):
        config = {}

    raw_strategy = str(config.get("strategy") or "").strip().lower()
    strategy = raw_strategy if raw_strategy in BINNING_STRATEGIES else "equal_width"

    columns: List[str] = []
    raw_columns = config.get("columns")
    if isinstance(raw_columns, list):
        seen: Set[str] = set()
        for entry in raw_columns:
            column = str(entry or "").strip()
            if column and column not in seen:
                seen.add(column)
                columns.append(column)
    elif isinstance(raw_columns, str):
        for segment in raw_columns.split(","):
            column = segment.strip()
            if column and column not in columns:
                columns.append(column)

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

    custom_bins: Dict[str, List[float]] = {}
    raw_custom_bins = config.get("custom_bins")
    if isinstance(raw_custom_bins, dict):
        for key, value in raw_custom_bins.items():
            column = str(key or "").strip()
            if not column:
                continue
            bins = _sanitize_number_list(value)
            if bins:
                custom_bins[column] = bins

    custom_labels: Dict[str, List[str]] = {}
    raw_custom_labels = config.get("custom_labels")
    if isinstance(raw_custom_labels, dict):
        for key, value in raw_custom_labels.items():
            column = str(key or "").strip()
            if not column:
                continue
            labels: List[str] = []
            if isinstance(value, list):
                labels = [str(entry).strip() for entry in value if str(entry).strip()]
            elif isinstance(value, str):
                labels = [segment.strip() for segment in value.split(",") if segment.strip()]
            if labels:
                custom_labels[column] = labels

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


def _apply_binning_discretization(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, BinningNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None

    data = node.get("data") or {}
    config_payload = data.get("config") or {}
    config = _normalize_binning_config(config_payload)

    strategy_literal = cast(Literal["equal_width", "equal_frequency", "custom", "kbins"], config.strategy)
    signal = BinningNodeSignal(
        node_id=str(node_id) if node_id is not None else None,
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
    if config.custom_bins:
        signal.custom_bins = {key: list(values) for key, values in config.custom_bins.items()}
    if config.custom_labels:
        signal.custom_labels = {key: list(values) for key, values in config.custom_labels.items()}

    if frame.empty:
        return frame, "Binning: no rows available", signal

    if not config.columns:
        return frame, "Binning: no columns selected", signal

    # Detect if we have train/test/validation splits
    has_splits = SPLIT_TYPE_COLUMN in frame.columns
    train_mask = None
    test_mask = None
    validation_mask = None
    
    if has_splits:
        train_mask = frame[SPLIT_TYPE_COLUMN] == "train"
        test_mask = frame[SPLIT_TYPE_COLUMN] == "test"
        validation_mask = frame[SPLIT_TYPE_COLUMN] == "validation"

    # KBinsDiscretizer path: uses sklearn, fit on train only
    if config.strategy == "kbins":
        return _apply_kbins_discretizer(
            frame=frame,
            config=config,
            node_id=node_id,
            pipeline_id=pipeline_id,
            has_splits=has_splits,
            train_mask=train_mask,
            test_mask=test_mask,
            validation_mask=validation_mask,
            signal=signal,
        )

    # Traditional pandas-based binning (equal_width, equal_frequency, custom)
    working_frame = frame.copy()
    transformed_details: List[str] = []
    skipped_columns: List[str] = []

    for column in config.columns:
        signal.evaluated_columns.append(column)
        if column not in working_frame.columns:
            skipped_columns.append(f"{column} (missing)")
            signal.skipped_columns.append(f"{column} (missing)")
            continue

        raw_series = working_frame[column]
        if pd_types.is_bool_dtype(raw_series):
            skipped_columns.append(f"{column} (boolean column automatically excluded)")
            signal.skipped_columns.append(f"{column} (boolean column automatically excluded)")
            continue

        numeric_series = pd.to_numeric(raw_series, errors="coerce")
        
        # If we have splits, calculate bins from training data only
        if has_splits and train_mask is not None:
            train_numeric = numeric_series[train_mask]
            valid_values = train_numeric.dropna()
        else:
            valid_values = numeric_series.dropna()
            
        valid_count = int(valid_values.size)
        if valid_count < 2:
            skipped_columns.append(f"{column} (insufficient numeric values)")
            signal.skipped_columns.append(f"{column} (insufficient numeric values)")
            continue

        if _is_binary_numeric(valid_values):
            skipped_columns.append(f"{column} (binary indicator column excluded)")
            signal.skipped_columns.append(f"{column} (binary indicator column excluded)")
            continue

        requested_bins = 0
        bin_edges = None
        
        try:
            if config.strategy == "equal_width":
                requested_bins = max(2, config.equal_width_bins)
                # Calculate bin edges from training data (or all data if no splits)
                min_val = valid_values.min()
                max_val = valid_values.max()
                bin_edges = pd.interval_range(start=min_val, end=max_val, periods=requested_bins, closed='right')
                # Convert to edge values for pd.cut
                bin_edges = [min_val] + [interval.right for interval in bin_edges]
                
                # Apply bins to all data
                binned_series = pd.cut(
                    numeric_series,
                    bins=bin_edges,
                    labels=None,
                    include_lowest=config.include_lowest,
                    precision=config.precision,
                    duplicates=config.duplicates,
                )
                
            elif config.strategy == "equal_frequency":
                requested_bins = max(2, config.equal_frequency_bins)
                # Calculate quantile edges from training data (or all data if no splits)
                quantiles = [i / requested_bins for i in range(requested_bins + 1)]
                bin_edges = valid_values.quantile(quantiles).unique().tolist()
                
                # Ensure we have at least 2 unique edges
                if len(bin_edges) < 2:
                    skipped_columns.append(f"{column} (unable to create bins)")
                    signal.skipped_columns.append(f"{column} (unable to create bins)")
                    continue
                
                # Apply bins to all data
                binned_series = pd.cut(
                    numeric_series,
                    bins=bin_edges,
                    labels=None,
                    include_lowest=True,
                    precision=config.precision,
                    duplicates=config.duplicates,
                )
                
            else:  # custom
                edges = config.custom_bins.get(column)
                if not edges:
                    skipped_columns.append(f"{column} (custom bins missing)")
                    signal.skipped_columns.append(f"{column} (custom bins missing)")
                    continue
                if len(edges) < 2:
                    skipped_columns.append(f"{column} (custom bins insufficient)")
                    signal.skipped_columns.append(f"{column} (custom bins insufficient)")
                    continue
                requested_bins = len(edges) - 1
                bin_edges = edges
                
                binned_series = pd.cut(
                    numeric_series,
                    bins=edges,
                    labels=None,
                    include_lowest=config.include_lowest,
                    precision=config.precision,
                    duplicates=config.duplicates,
                )
                
        except ValueError as error:
            skipped_columns.append(f"{column} ({error})")
            signal.skipped_columns.append(f"{column} ({error})")
            continue

        categories = list(binned_series.cat.categories)
        if not categories:
            skipped_columns.append(f"{column} (unable to derive bins)")
            signal.skipped_columns.append(f"{column} (unable to derive bins)")
            continue

        label_override: Optional[List[str]] = None
        if config.strategy == "custom":
            custom_labels = config.custom_labels.get(column)
            if custom_labels and len(custom_labels) >= len(categories):
                label_override = custom_labels[: len(categories)]

        if label_override is None:
            label_override = _build_binning_labels(column, config, len(categories))

        if label_override is not None and len(label_override) == len(categories):
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

        new_column = f"{column}{config.output_suffix}"
        working_frame[new_column] = formatted_series

        if config.drop_original and column in working_frame.columns:
            working_frame = working_frame.drop(columns=[column])

        # Store binning metadata for transformer audit (even for pandas-based methods)
        if pipeline_id and node_id and has_splits:
            storage = get_pipeline_store()
            
            # Determine method label
            if config.strategy == "equal_width":
                method_label = f"Equal-width ({actual_bins} bins)"
                transformer_name = "binning_equal_width"
            elif config.strategy == "equal_frequency":
                method_label = f"Equal-frequency ({actual_bins} bins)"
                transformer_name = "binning_equal_frequency"
            else:  # custom
                method_label = f"Custom ({actual_bins} bins)"
                transformer_name = "binning_custom"
            
            train_rows = int(train_mask.sum()) if train_mask is not None else 0
            test_rows = int(test_mask.sum()) if test_mask is not None else 0
            validation_rows = int(validation_mask.sum()) if validation_mask is not None else 0
            
            # Store bin edges and metadata (not a fitted transformer, just metadata)
            bin_edges = [interval.left for interval in categories] + [categories[-1].right] if categories else []
            
            storage.store_transformer(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name=transformer_name,
                transformer={
                    "type": "pandas_binning",
                    "strategy": config.strategy,
                    "bin_edges": bin_edges,
                    "categories": [str(cat) for cat in categories],
                },
                column_name=column,
                metadata={
                    "method": config.strategy,
                    "method_label": method_label,
                    "n_bins": actual_bins,
                    "requested_bins": requested_bins or actual_bins,
                    "train_rows": train_rows,
                }
            )
            
            # Record split activity
            storage.record_split_activity(
                pipeline_id=pipeline_id,
                node_id=str(node_id),
                transformer_name=transformer_name,
                column_name=column,
                split_name="train",
                action="transform",
                row_count=train_rows,
            )
            if test_rows > 0:
                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name=transformer_name,
                    column_name=column,
                    split_name="test",
                    action="transform",
                    row_count=test_rows,
                )
            if validation_rows > 0:
                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name=transformer_name,
                    column_name=column,
                    split_name="validation",
                    action="transform",
                    row_count=validation_rows,
                )

        detail = f"{column}→{new_column} ({actual_bins} bin{'s' if actual_bins != 1 else ''})"
        if config.strategy == "equal_frequency" and requested_bins and actual_bins < requested_bins:
            detail = f"{detail} (reduced from {requested_bins})"
        if config.strategy == "custom":
            detail = f"{detail} custom"
        transformed_details.append(detail)

        distribution = _build_binned_distribution(
            new_column,
            working_frame[new_column],
            source_column=column,
            missing_label=config.missing_label if config.missing_strategy == "label" else None,
        )

        signal.applied_columns.append(
            BinningAppliedColumnSignal(
                source_column=column,
                output_column=new_column,
                strategy=strategy_literal,
                requested_bins=requested_bins or None,
                actual_bins=actual_bins,
                reduced_bins=bool(requested_bins and actual_bins and actual_bins < requested_bins),
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

    if signal.evaluated_columns:
        signal.evaluated_columns = list(dict.fromkeys(signal.evaluated_columns))
    if signal.skipped_columns:
        signal.skipped_columns = list(dict.fromkeys(signal.skipped_columns))

    if not transformed_details:
        summary = "Binning: no columns transformed"
        if skipped_columns:
            summary = f"{summary}; skipped {', '.join(skipped_columns)}"
        return working_frame, summary, signal

    strategy_label = config.strategy.replace("_", " ")
    summary = f"Binning ({strategy_label}): {', '.join(transformed_details)}"
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


def _is_binary_numeric(series: pd.Series) -> bool:
    unique_values: Set[float] = set()
    for value in series.unique():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(numeric):
            return False
        if abs(numeric) < 1e-9:
            numeric = 0.0
        elif abs(numeric - 1.0) < 1e-9:
            numeric = 1.0
        unique_values.add(numeric)
        if len(unique_values) > 2:
            return False
    return unique_values.issubset({0.0, 1.0})


def _detect_numeric_columns(frame: pd.DataFrame) -> List[str]:
    detected: List[str] = []
    seen: Set[str] = set()

    for column in frame.columns:
        column_name = str(column)
        if not column_name or column_name in seen:
            continue

        series = frame[column]
        dtype = series.dtype
        if pd_types.is_bool_dtype(dtype):
            continue

        numeric_series = pd.to_numeric(series, errors="coerce")
        valid = numeric_series.dropna()
        if valid.empty:
            continue

        if _is_binary_numeric(valid):
            continue

        if valid.nunique(dropna=True) < 2:
            continue

        detected.append(column_name)
        seen.add(column_name)

    return detected


def _apply_kbins_discretizer(
    frame: pd.DataFrame,
    config: NormalizedBinningConfig,
    node_id: Any,
    pipeline_id: Optional[str],
    has_splits: bool,
    train_mask: Optional[pd.Series],
    test_mask: Optional[pd.Series],
    validation_mask: Optional[pd.Series],
    signal: BinningNodeSignal,
) -> Tuple[pd.DataFrame, str, BinningNodeSignal]:
    """Apply KBinsDiscretizer with split-aware fit/transform."""
    
    working_frame = frame.copy()
    transformed_details: List[str] = []
    skipped_columns: List[str] = []
    
    for column in config.columns:
        signal.evaluated_columns.append(column)
        
        if column not in working_frame.columns:
            skipped_columns.append(f"{column} (missing)")
            signal.skipped_columns.append(f"{column} (missing)")
            continue

        raw_series = working_frame[column]
        if pd_types.is_bool_dtype(raw_series):
            skipped_columns.append(f"{column} (boolean column automatically excluded)")
            signal.skipped_columns.append(f"{column} (boolean column automatically excluded)")
            continue

        # Convert to numeric
        numeric_series = pd.to_numeric(raw_series, errors="coerce")
        
        # Check if we have enough valid values
        if has_splits and train_mask is not None:
            train_values = numeric_series[train_mask].dropna()
        else:
            train_values = numeric_series.dropna()
        
        if len(train_values) < 2:
            skipped_columns.append(f"{column} (insufficient numeric values)")
            signal.skipped_columns.append(f"{column} (insufficient numeric values)")
            continue
        
        if _is_binary_numeric(train_values):
            skipped_columns.append(f"{column} (binary indicator column excluded)")
            signal.skipped_columns.append(f"{column} (binary indicator column excluded)")
            continue

        # Create KBinsDiscretizer
        try:
            # Build kwargs for KBinsDiscretizer
            discretizer_kwargs = {
                "n_bins": config.kbins_n_bins,
                "encode": config.kbins_encode,
                "strategy": config.kbins_strategy,
                "subsample": None,
            }
            
            # For quantile strategy, add the recommended quantile_method to avoid FutureWarning
            if config.kbins_strategy == "quantile":
                discretizer_kwargs["quantile_method"] = "averaged_inverted_cdf"
            
            discretizer = KBinsDiscretizer(**discretizer_kwargs)
            
            # Fit on training data only
            if has_splits and train_mask is not None:
                train_data = numeric_series[train_mask].dropna()
                if len(train_data) < config.kbins_n_bins:
                    skipped_columns.append(f"{column} (insufficient train samples for {config.kbins_n_bins} bins)")
                    signal.skipped_columns.append(f"{column} (insufficient train samples)")
                    continue
                
                discretizer.fit(train_data.values.reshape(-1, 1))
            else:
                # No splits, fit on all data
                all_data = numeric_series.dropna()
                if len(all_data) < config.kbins_n_bins:
                    skipped_columns.append(f"{column} (insufficient samples for {config.kbins_n_bins} bins)")
                    signal.skipped_columns.append(f"{column} (insufficient samples)")
                    continue
                
                discretizer.fit(all_data.values.reshape(-1, 1))
            
            # Transform all data (train, test, validation)
            transformed_col = numeric_series.copy()
            non_na_mask = ~numeric_series.isna()
            
            if non_na_mask.any():
                transformed_values = discretizer.transform(
                    numeric_series[non_na_mask].values.reshape(-1, 1)
                )
                
                # Check if one-hot encoding (returns 2D array with multiple columns)
                if config.kbins_encode in ["onehot", "onehot-dense"]:
                    # For one-hot encoding, we need to create multiple columns
                    # For now, convert back to ordinal (bin index) for single column output
                    transformed_values = transformed_values.argmax(axis=1).reshape(-1, 1)
                
                transformed_col[non_na_mask] = transformed_values.flatten()
            
            # Create new column name
            new_column = f"{column}{config.output_suffix}"
            
            # Handle missing values
            if config.missing_strategy == "label":
                fill_value = config.missing_label or BINNING_DEFAULT_MISSING_LABEL
                transformed_col = transformed_col.fillna(fill_value)
            
            # Convert to appropriate type based on encode
            if config.kbins_encode == "ordinal":
                transformed_col = transformed_col.astype("Int64")  # Nullable integer
            
            working_frame[new_column] = transformed_col
            
            # Drop original if requested
            if config.drop_original and column in working_frame.columns:
                working_frame = working_frame.drop(columns=[column])
            
            # Store transformer if we have pipeline_id and node_id
            if pipeline_id and node_id and has_splits:
                storage = get_pipeline_store()
                
                train_rows = int(train_mask.sum()) if train_mask is not None else 0
                test_rows = int(test_mask.sum()) if test_mask is not None else 0
                validation_rows = int(validation_mask.sum()) if validation_mask is not None else 0
                
                storage.store_transformer(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="kbins_discretizer",
                    transformer=discretizer,
                    column_name=column,
                    metadata={
                        "method": config.kbins_strategy,
                        "method_label": f"KBins ({config.kbins_strategy})",
                        "n_bins": config.kbins_n_bins,
                        "encode": config.kbins_encode,
                        "train_rows": train_rows,
                    }
                )
                
                # Record split activity
                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="kbins_discretizer",
                    column_name=column,
                    split_name="train",
                    action="fit_transform",
                    row_count=train_rows,
                )
                if test_rows > 0:
                    storage.record_split_activity(
                        pipeline_id=pipeline_id,
                        node_id=str(node_id),
                        transformer_name="kbins_discretizer",
                        column_name=column,
                        split_name="test",
                        action="transform",
                        row_count=test_rows,
                    )
                if validation_rows > 0:
                    storage.record_split_activity(
                        pipeline_id=pipeline_id,
                        node_id=str(node_id),
                        transformer_name="kbins_discretizer",
                        column_name=column,
                        split_name="validation",
                        action="transform",
                        row_count=validation_rows,
                    )
            
            # Build detail message
            actual_bins = discretizer.n_bins_[0] if hasattr(discretizer, 'n_bins_') else config.kbins_n_bins
            detail = f"{column}→{new_column} ({actual_bins} bins, {config.kbins_strategy})"
            transformed_details.append(detail)
            
            # Add to signal
            signal.applied_columns.append(
                BinningAppliedColumnSignal(
                    source_column=column,
                    output_column=new_column,
                    strategy="kbins",
                    requested_bins=config.kbins_n_bins,
                    actual_bins=actual_bins,
                    reduced_bins=False,
                    drop_original=config.drop_original,
                    include_lowest=True,  # Not applicable for kbins
                    precision=0,  # Not applicable for kbins
                    duplicates="raise",  # Not applicable for kbins
                    label_format=config.kbins_encode,
                    missing_strategy=config.missing_strategy,
                    missing_label=config.missing_label if config.missing_strategy == "label" else None,
                    custom_labels_applied=False,
                    sample_bins=[],
                )
            )
            
        except Exception as e:
            logger.error(f"[KBINS] Exception processing column {column}: {str(e)}", exc_info=True)
            skipped_columns.append(f"{column} ({str(e)})")
            signal.skipped_columns.append(f"{column} ({str(e)})")
            continue
    
    # Build summary
    if not transformed_details:
        summary = "Binning (KBins): no columns transformed"
        if skipped_columns:
            summary = f"{summary}; skipped {', '.join(skipped_columns)}"
        return working_frame, summary, signal
    
    summary = f"Binning (KBins {config.kbins_strategy}): {', '.join(transformed_details)}"
    if skipped_columns:
        summary = f"{summary}; skipped {', '.join(skipped_columns)}"
    
    return working_frame, summary, signal


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
]
