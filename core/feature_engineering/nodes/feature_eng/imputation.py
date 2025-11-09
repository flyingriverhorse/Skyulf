"""Imputation helpers for feature engineering nodes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import StatisticsError
from typing import Any, Dict, Optional, Set, Tuple, cast, List

import pandas as pd

try:  # pragma: no cover - optional dependency guard
    from sklearn.experimental import enable_iterative_imputer  # type: ignore  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - defensive guard
    IterativeImputer = None  # type: ignore[assignment]
    KNNImputer = None  # type: ignore[assignment]

from core.feature_engineering.schemas import (
    ImputationAppliedColumnSignal,
    ImputationConfiguredStrategySignal,
    ImputationMethodName,
    ImputationNodeSignal,
)
from core.feature_engineering.nodes.modeling.dataset_split import SPLIT_TYPE_COLUMN
from core.feature_engineering.sklearn_pipeline_store import get_pipeline_store

from .utils import _auto_detect_numeric_columns, _coerce_string_list


METHOD_LABELS: Dict[ImputationMethodName, str] = {
    cast(ImputationMethodName, "mean"): "Mean",
    cast(ImputationMethodName, "median"): "Median",
    cast(ImputationMethodName, "mode"): "Mode",
    cast(ImputationMethodName, "knn"): "KNN",
    cast(ImputationMethodName, "regression"): "Regression",
    cast(ImputationMethodName, "mice"): "MICE",
}

SIMPLE_METHODS: Set[str] = {"mean", "median", "mode"}
ADVANCED_METHODS: Set[str] = {"knn", "regression", "mice"}


@dataclass
class SplitContext:
    storage: Optional[Any]
    pipeline_id: Optional[str]
    node_id: Optional[str]
    has_splits: bool
    train_mask: Optional[pd.Series]
    split_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class ImputationAccumulator:
    filled_cells: int = 0
    touched_columns: Set[str] = field(default_factory=set)
    used_methods: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    skipped_non_numeric: Set[str] = field(default_factory=set)


@dataclass
class StrategySetup:
    method: ImputationMethodName
    target_columns: List[str]
    auto_detected: bool
    raw_entry: Dict[str, Any]
    config_signal: ImputationConfiguredStrategySignal

    def options(self) -> Dict[str, Any]:
        raw_options = self.raw_entry.get("options")
        return raw_options if isinstance(raw_options, dict) else {}


def _prepare_strategy_setup(
    entry: Any,
    working_frame: pd.DataFrame,
) -> Optional[StrategySetup]:
    if not isinstance(entry, dict):
        return None

    raw_method = entry.get("method") or "mean"
    method = str(raw_method).strip().lower()
    if method not in SIMPLE_METHODS and method not in ADVANCED_METHODS:
        return None

    method_name = cast(ImputationMethodName, method)

    raw_columns = entry.get("columns")
    candidate_columns = _coerce_string_list(raw_columns)
    if candidate_columns:
        auto_detected = False
    else:
        candidate_columns = _auto_detect_numeric_columns(working_frame)
        auto_detected = True

    target_columns = [column for column in candidate_columns if column in working_frame.columns]

    config_signal = ImputationConfiguredStrategySignal(
        method=method_name,
        columns=list(target_columns),
        auto_detected=auto_detected,
        parameters={},
    )

    return StrategySetup(
        method=method_name,
        target_columns=target_columns,
        auto_detected=auto_detected,
        raw_entry=dict(entry),
        config_signal=config_signal,
    )


def _store_simple_imputer(
    column: str,
    method: ImputationMethodName,
    replacement: Any,
    split_context: SplitContext,
) -> None:
    if not (split_context.storage and split_context.has_splits):
        return

    replacement_value = (
        float(replacement)
        if isinstance(replacement, (int, float))
        else str(replacement)
    )
    imputer_metadata = {
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "replacement_value": replacement_value,
    }
    transformer_payload = {
        "value": replacement,
        "method": method,
    }
    split_context.storage.store_transformer(
        pipeline_id=split_context.pipeline_id,
        node_id=str(split_context.node_id),
        transformer_name="imputer",
        transformer=transformer_payload,
        column_name=column,
        metadata=imputer_metadata,
    )

    train_rows = int(split_context.train_mask.sum()) if split_context.train_mask is not None else 0
    split_context.storage.record_split_activity(
        pipeline_id=split_context.pipeline_id,
        node_id=str(split_context.node_id),
        transformer_name="imputer",
        column_name=column,
        split_name="train",
        action="fit_transform",
        row_count=train_rows,
    )

    for split_name in ("test", "validation"):
        rows_processed = int(split_context.split_counts.get(split_name, 0))
        if rows_processed > 0:
            split_context.storage.record_split_activity(
                pipeline_id=split_context.pipeline_id,
                node_id=str(split_context.node_id),
                transformer_name="imputer",
                column_name=column,
                split_name=split_name,
                action="transform",
                row_count=rows_processed,
            )


def _store_advanced_imputer(
    column: str,
    method: ImputationMethodName,
    imputer: Any,
    split_context: SplitContext,
    parameters: Dict[str, Any],
) -> None:
    if not (split_context.storage and split_context.has_splits):
        return

    metadata = {
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "train_rows": int(split_context.train_mask.sum()) if split_context.train_mask is not None else 0,
    }
    metadata.update(parameters)

    split_context.storage.store_transformer(
        pipeline_id=split_context.pipeline_id,
        node_id=str(split_context.node_id),
        transformer_name="imputer",
        transformer=imputer,
        column_name=column,
        metadata=metadata,
    )

    split_context.storage.record_split_activity(
        pipeline_id=split_context.pipeline_id,
        node_id=str(split_context.node_id),
        transformer_name="imputer",
        column_name=column,
        split_name="train",
        action="fit_transform",
        row_count=int(split_context.train_mask.sum()) if split_context.train_mask is not None else 0,
    )

    for split_name in ("test", "validation"):
        rows_processed = int(split_context.split_counts.get(split_name, 0))
        if rows_processed > 0:
            split_context.storage.record_split_activity(
                pipeline_id=split_context.pipeline_id,
                node_id=str(split_context.node_id),
                transformer_name="imputer",
                column_name=column,
                split_name=split_name,
                action="transform",
                row_count=rows_processed,
            )


def _compute_simple_replacement(
    series: pd.Series,
    method: ImputationMethodName,
    split_context: SplitContext,
) -> Tuple[Optional[Any], bool]:
    method_key = cast(str, method)

    if method_key in {"mean", "median"}:
        if split_context.has_splits and split_context.train_mask is not None:
            source_series = series[split_context.train_mask]
        else:
            source_series = series
        numeric_series = pd.to_numeric(source_series, errors="coerce")
        numeric_valid = numeric_series.dropna()
        if numeric_valid.empty:
            return None, True
        if method_key == "mean":
            return numeric_valid.mean(), False
        return numeric_valid.median(), False

    try:
        if split_context.has_splits and split_context.train_mask is not None:
            mode_series = series[split_context.train_mask].mode(dropna=True)
        else:
            mode_series = series.mode(dropna=True)
        replacement = mode_series.iloc[0] if not mode_series.empty else None
    except (IndexError, StatisticsError):  # pragma: no cover - defensive
        replacement = None
    except Exception:  # pragma: no cover - pandas oddities
        replacement = None
    return replacement, False


def _apply_simple_strategy(
    setup: StrategySetup,
    working_frame: pd.DataFrame,
    signal: ImputationNodeSignal,
    accumulator: ImputationAccumulator,
    split_context: SplitContext,
) -> Optional[str]:
    method = setup.method
    method_key = cast(str, method)

    for column in setup.target_columns:
        series = working_frame[column]
        missing_mask = series.isna()
        if not missing_mask.any():
            continue

        replacement, flagged_non_numeric = _compute_simple_replacement(series, method, split_context)
        if flagged_non_numeric:
            accumulator.skipped_non_numeric.add(column)
        if replacement is None or (isinstance(replacement, float) and pd.isna(replacement)):
            continue

        before_fill = int(missing_mask.sum())
        if before_fill == 0:
            continue

        try:
            working_frame[column] = series.fillna(replacement)
        except TypeError as exc:
            bool_message = _describe_bool_imputation_error(column, method_key, exc)
            if bool_message:
                signal.errors.append(bool_message)
                return bool_message
            cast_message = _describe_integer_cast_error(column, method_key, exc)
            if cast_message:
                signal.errors.append(cast_message)
                return cast_message
            raise

        after_fill = int(working_frame[column].isna().sum())
        delta = before_fill - after_fill
        if delta <= 0:
            continue

        accumulator.filled_cells += delta
        accumulator.touched_columns.add(column)
        accumulator.used_methods[method_key] += 1

        signal.applied_columns.append(
            ImputationAppliedColumnSignal(
                column=column,
                method=method,
                method_label=METHOD_LABELS.get(method),
                filled_cells=delta,
                original_missing=before_fill,
                remaining_missing=after_fill,
                dtype=str(series.dtype),
            )
        )

        _store_simple_imputer(column, method, replacement, split_context)

    return None


def _resolve_knn_neighbors(entry: Dict[str, Any], subset: pd.DataFrame) -> Optional[int]:
    options = entry.get("options") if isinstance(entry.get("options"), dict) else {}
    neighbors_value = None
    if isinstance(options, dict):
        neighbors_value = options.get("neighbors")
    if neighbors_value is None:
        neighbors_value = entry.get("neighbors")

    available_rows = subset.dropna(how="all").shape[0]
    if available_rows == 0:
        return None
    max_neighbors = max(1, available_rows)
    return _coerce_positive_int(neighbors_value, default=5, minimum=1, maximum=max_neighbors)


def _resolve_max_iter(entry: Dict[str, Any]) -> int:
    options = entry.get("options") if isinstance(entry.get("options"), dict) else {}
    if isinstance(options, dict):
        max_iter_value = options.get("max_iter", options.get("maxIter"))
    else:
        max_iter_value = None
    if max_iter_value is None:
        max_iter_value = entry.get("max_iter")
    return _coerce_positive_int(max_iter_value, default=10, minimum=1, maximum=100)


def _configure_advanced_imputer(
    setup: StrategySetup,
    subset: pd.DataFrame,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    method = cast(str, setup.method)
    if method == "knn":
        neighbors = _resolve_knn_neighbors(setup.raw_entry, subset)
        if neighbors is None or KNNImputer is None:  # pragma: no cover - defensive guard
            return None, {}
        setup.config_signal.parameters.setdefault("neighbors", neighbors)
        return KNNImputer(n_neighbors=neighbors), {"neighbors": neighbors}

    if IterativeImputer is None:  # pragma: no cover - defensive guard
        return None, {}

    max_iter = _resolve_max_iter(setup.raw_entry)
    sample_posterior = method == "mice"
    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=0,
        sample_posterior=sample_posterior,
        skip_complete=True,
    )
    setup.config_signal.parameters.setdefault("max_iter", max_iter)
    parameters: Dict[str, Any] = {"max_iter": max_iter}
    if sample_posterior:
        setup.config_signal.parameters.setdefault("sample_posterior", True)
        parameters["sample_posterior"] = True
    return imputer, parameters


def _fit_and_transform_imputer(
    imputer: Any,
    subset: pd.DataFrame,
    split_context: SplitContext,
) -> Optional[pd.DataFrame]:
    try:
        if split_context.has_splits and split_context.train_mask is not None:
            train_subset = subset[split_context.train_mask]
            if train_subset.empty or not train_subset.isna().any().any():
                return None
            imputer.fit(train_subset)
            imputed_array = imputer.transform(subset)
        else:
            imputed_array = imputer.fit_transform(subset)
    except Exception:  # pragma: no cover - defensive fallback
        return None

    return pd.DataFrame(imputed_array, columns=subset.columns, index=subset.index)


def _apply_advanced_strategy(
    setup: StrategySetup,
    working_frame: pd.DataFrame,
    signal: ImputationNodeSignal,
    accumulator: ImputationAccumulator,
    split_context: SplitContext,
) -> Optional[str]:
    if KNNImputer is None or IterativeImputer is None:  # pragma: no cover - defensive guard
        message = "Imputation methods: scikit-learn components unavailable"
        signal.errors.append(message)
        return message

    numeric_subset = working_frame[setup.target_columns].apply(pd.to_numeric, errors="coerce")
    valid_columns = [column for column in setup.target_columns if not numeric_subset[column].dropna().empty]
    if not valid_columns:
        accumulator.skipped_non_numeric.update(setup.target_columns)
        return None

    subset = numeric_subset[valid_columns]
    if not subset.isna().any().any():
        return None

    imputer, parameters = _configure_advanced_imputer(setup, subset)
    if imputer is None:
        return None

    imputed_frame = _fit_and_transform_imputer(imputer, subset, split_context)
    if imputed_frame is None:
        return None

    method_key = cast(str, setup.method)
    strategy_filled = 0

    original_missing_counts = {
        column: int(working_frame[column].isna().sum()) for column in valid_columns
    }

    for column in valid_columns:
        original_series = working_frame[column]
        replacement_series = imputed_frame[column]
        mask = original_series.isna() & replacement_series.notna()
        if not mask.any():
            continue

        newly_filled = int(mask.sum())
        strategy_filled += newly_filled
        accumulator.filled_cells += newly_filled
        accumulator.touched_columns.add(column)

        try:
            working_frame.loc[mask, column] = replacement_series.loc[mask]
        except TypeError as exc:
            bool_message = _describe_bool_imputation_error(column, method_key, exc)
            if bool_message:
                signal.errors.append(bool_message)
                return bool_message
            cast_message = _describe_integer_cast_error(column, method_key, exc)
            if cast_message:
                signal.errors.append(cast_message)
                return cast_message
            raise

        remaining_missing = int(working_frame[column].isna().sum())
        signal.applied_columns.append(
            ImputationAppliedColumnSignal(
                column=column,
                method=setup.method,
                method_label=METHOD_LABELS.get(setup.method),
                filled_cells=newly_filled,
                original_missing=original_missing_counts.get(column, newly_filled),
                remaining_missing=remaining_missing,
                dtype=str(original_series.dtype),
            )
        )

        _store_advanced_imputer(column, setup.method, imputer, split_context, parameters)

    if strategy_filled > 0:
        accumulator.used_methods[method_key] += 1

    return None


def _compose_summary(accumulator: ImputationAccumulator) -> str:
    if not accumulator.filled_cells:
        summary = "Imputation methods: no missing values replaced"
        if accumulator.skipped_non_numeric:
            summary += f"; skipped {len(accumulator.skipped_non_numeric)} non-numeric column(s)"
        return summary

    if accumulator.used_methods:
        readable_methods = sorted(set(accumulator.used_methods.keys()))
        method_list = ", ".join(
            METHOD_LABELS.get(cast(ImputationMethodName, name), name) for name in readable_methods
        )
    else:
        method_list = "configured"

    parts = [
        f"Imputation methods: filled {accumulator.filled_cells} cell(s) "
        f"across {len(accumulator.touched_columns)} column(s) using {method_list}"
    ]
    if accumulator.skipped_non_numeric:
        parts.append(f"skipped {len(accumulator.skipped_non_numeric)} non-numeric column(s)")
    return "; ".join(parts)


def _describe_bool_imputation_error(column: str, method: str, error: Exception) -> Optional[str]:
    message_text = str(error)
    lowered = message_text.lower()
    if "bool" in lowered and "need to pass" in lowered:
        readable_method = method.upper() if method in {"knn", "mice"} else method
        return (
            f"Imputation methods: column '{column}' stores boolean values; "
            f"{readable_method} cannot coerce non-boolean replacements. "
            "Use a True/False constant or cast the column before imputing."
        )
    return None


def _describe_integer_cast_error(column: str, method: str, error: Exception) -> Optional[str]:
    message_text = str(error)
    lowered = message_text.lower()
    if "cast" in lowered and "int" in lowered:
        readable_method = method.upper() if method in {"knn", "mice"} else method
        return (
            f"Imputation methods: column '{column}' is integer-typed but {readable_method} "
            "produced non-integer replacements. Cast the column to float or "
            "switch to a constant/compatible strategy before imputing."
        )
    return None


def _coerce_positive_int(value: Any, default: int, minimum: int = 1, maximum: Optional[int] = None) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        numeric = default
    numeric = max(minimum, numeric)
    if maximum is not None:
        numeric = min(numeric, maximum)
    return numeric


def _finalize_signal(
    signal: ImputationNodeSignal,
    filled_cells: int,
    touched_columns: Set[str],
    skipped_non_numeric: Set[str],
    used_methods: Dict[str, int],
) -> None:
    signal.filled_cells = filled_cells
    signal.touched_columns = sorted(touched_columns)

    if skipped_non_numeric:
        existing = set(signal.skipped_columns)
        for column in sorted(skipped_non_numeric):
            reason = f"{column} (non-numeric)"
            if reason not in existing:
                signal.skipped_columns.append(reason)
                existing.add(reason)

    if used_methods:
        signal.method_usage = {
            cast(ImputationMethodName, key): int(value)
            for key, value in used_methods.items()
            if key in {"mean", "median", "mode", "knn", "regression", "mice"}
        }


def apply_imputation_methods(
    frame: pd.DataFrame,
    node: Dict[str, Any],
    pipeline_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, ImputationNodeSignal]:
    node_id = node.get("id") if isinstance(node, dict) else None

    signal = ImputationNodeSignal(node_id=str(node_id) if node_id is not None else None)

    if frame.empty:
        return frame, "Imputation methods: no data available", signal

    data = node.get("data") or {}
    config = data.get("config") or {}
    strategies = config.get("strategies") or []

    if not isinstance(strategies, list) or not strategies:
        return frame, "Imputation methods: no strategies configured", signal

    working_frame = frame.copy()
    accumulator = ImputationAccumulator()

    has_splits = SPLIT_TYPE_COLUMN in working_frame.columns
    storage = get_pipeline_store() if pipeline_id and has_splits else None
    split_counts = working_frame[SPLIT_TYPE_COLUMN].value_counts().to_dict() if has_splits else {}
    train_mask = (working_frame[SPLIT_TYPE_COLUMN] == "train") if has_splits else None

    split_context = SplitContext(
        storage=storage,
        pipeline_id=pipeline_id,
        node_id=str(node_id) if node_id is not None else None,
        has_splits=has_splits,
        train_mask=train_mask,
        split_counts=split_counts,
    )

    for entry in strategies:
        setup = _prepare_strategy_setup(entry, working_frame)
        if setup is None:
            continue

        signal.configured_strategies.append(setup.config_signal)

        if not setup.target_columns:
            continue

        method_key = cast(str, setup.method)
        if method_key in SIMPLE_METHODS:
            error = _apply_simple_strategy(setup, working_frame, signal, accumulator, split_context)
        else:
            error = _apply_advanced_strategy(setup, working_frame, signal, accumulator, split_context)

        if error:
            _finalize_signal(
                signal,
                accumulator.filled_cells,
                accumulator.touched_columns,
                accumulator.skipped_non_numeric,
                dict(accumulator.used_methods),
            )
            return frame, error, signal

    _finalize_signal(
        signal,
        accumulator.filled_cells,
        accumulator.touched_columns,
        accumulator.skipped_non_numeric,
        dict(accumulator.used_methods),
    )

    summary = _compose_summary(accumulator)

    return working_frame, summary, signal
