"""Imputation helpers for feature engineering nodes."""

from __future__ import annotations

from collections import defaultdict
from statistics import StatisticsError
from typing import Any, Dict, List, Optional, Set, Tuple, cast

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
from core.feature_engineering.transformer_storage import get_transformer_storage

from .utils import _auto_detect_numeric_columns, _coerce_string_list


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

    method_labels: Dict[ImputationMethodName, str] = {
        cast(ImputationMethodName, "mean"): "Mean",
        cast(ImputationMethodName, "median"): "Median",
        cast(ImputationMethodName, "mode"): "Mode",
        cast(ImputationMethodName, "knn"): "KNN",
        cast(ImputationMethodName, "regression"): "Regression",
        cast(ImputationMethodName, "mice"): "MICE",
    }

    if frame.empty:
        return frame, "Imputation methods: no data available", signal

    data = node.get("data") or {}
    config = data.get("config") or {}
    strategies = config.get("strategies") or []

    if not isinstance(strategies, list) or not strategies:
        return frame, "Imputation methods: no strategies configured", signal

    working_frame = frame.copy()
    filled_cells = 0
    touched_columns: Set[str] = set()
    used_methods: Dict[str, int] = defaultdict(int)

    simple_methods = {"mean", "median", "mode"}
    advanced_methods = {"knn", "regression", "mice"}
    skipped_non_numeric: Set[str] = set()

    # Check for split-aware processing
    has_splits = SPLIT_TYPE_COLUMN in working_frame.columns
    storage = get_transformer_storage() if pipeline_id and has_splits else None
    
    split_counts: Dict[str, int] = {}
    train_mask: Optional[pd.Series] = None
    if has_splits:
        split_counts = working_frame[SPLIT_TYPE_COLUMN].value_counts().to_dict()
        train_mask = working_frame[SPLIT_TYPE_COLUMN] == "train"

    for entry in strategies:
        if not isinstance(entry, dict):
            continue

        raw_method = entry.get("method") or "mean"
        method = str(raw_method).strip().lower()
        if method not in simple_methods and method not in advanced_methods:
            continue

        method_name = cast(ImputationMethodName, method)

        raw_columns = entry.get("columns")
        candidate_columns = _coerce_string_list(raw_columns)
        if not candidate_columns:
            candidate_columns = _auto_detect_numeric_columns(working_frame)
            auto_detected = True
        else:
            auto_detected = False

        target_columns = [column for column in candidate_columns if column in working_frame.columns]
        if not target_columns:
            signal.configured_strategies.append(
                ImputationConfiguredStrategySignal(
                    method=method_name,
                    columns=[],
                    auto_detected=auto_detected,
                    parameters={},
                )
            )
            continue

        strategy_signal = ImputationConfiguredStrategySignal(
            method=method_name,
            columns=list(target_columns),
            auto_detected=auto_detected,
            parameters={},
        )
        signal.configured_strategies.append(strategy_signal)

        if method in simple_methods:
            for column in target_columns:
                series = working_frame[column]
                original_missing = int(series.isna().sum())
                current_missing = series.isna()
                if not current_missing.any():
                    continue

                # Calculate replacement value from training data only (if splits exist)
                replacement: Any
                if method in {"mean", "median"}:
                    if has_splits and train_mask is not None:
                        # Use only training data to calculate the statistic
                        train_series = series[train_mask]
                        numeric_series = pd.to_numeric(train_series, errors="coerce")
                    else:
                        numeric_series = pd.to_numeric(series, errors="coerce")
                    
                    numeric_valid = numeric_series.dropna()
                    if numeric_valid.empty:
                        skipped_non_numeric.add(column)
                        continue
                    if method == "mean":
                        replacement = numeric_valid.mean()
                    else:
                        replacement = numeric_valid.median()
                else:  # mode
                    try:
                        if has_splits and train_mask is not None:
                            # Use only training data to calculate mode
                            train_series = series[train_mask]
                            mode_series = train_series.mode(dropna=True)
                        else:
                            mode_series = series.mode(dropna=True)
                        replacement = mode_series.iloc[0] if not mode_series.empty else None
                    except (IndexError, StatisticsError):  # pragma: no cover - defensive
                        replacement = None
                    except Exception:  # pragma: no cover - pandas oddities
                        replacement = None

                if replacement is None or (isinstance(replacement, float) and pd.isna(replacement)):
                    continue

                before_fill = int(current_missing.sum())
                if before_fill == 0:
                    continue

                try:
                    working_frame[column] = series.fillna(replacement)
                except TypeError as exc:
                    bool_message = _describe_bool_imputation_error(column, method, exc)
                    if bool_message:
                        signal.errors.append(bool_message)
                        _finalize_signal(signal, filled_cells, touched_columns, skipped_non_numeric, used_methods)
                        return frame, bool_message, signal
                    cast_message = _describe_integer_cast_error(column, method, exc)
                    if cast_message:
                        signal.errors.append(cast_message)
                        _finalize_signal(signal, filled_cells, touched_columns, skipped_non_numeric, used_methods)
                        return frame, cast_message, signal
                    raise
                after_fill_missing = int(working_frame[column].isna().sum())
                delta = before_fill - after_fill_missing
                if delta > 0:
                    filled_cells += delta
                    touched_columns.add(column)
                    used_methods[method] += 1
                    signal.applied_columns.append(
                        ImputationAppliedColumnSignal(
                            column=column,
                            method=method_name,
                            method_label=method_labels.get(method_name),
                            filled_cells=delta,
                            original_missing=original_missing,
                            remaining_missing=after_fill_missing,
                            dtype=str(series.dtype),
                        )
                    )

                    # Track split activity for simple methods (they store the replacement value)
                    if storage is not None and has_splits:
                        # Store the replacement value as "transformer" for tracking
                        imputer_metadata = {
                            "method": method,
                            "method_label": method_labels.get(method_name, method),
                            "replacement_value": float(replacement) if isinstance(replacement, (int, float)) else str(replacement),
                        }
                        
                        storage.store_transformer(
                            pipeline_id=pipeline_id,
                            node_id=str(node_id),
                            transformer_name="imputer",
                            transformer={"value": replacement, "method": method},  # Simple dict instead of sklearn object
                            column_name=column,
                            metadata=imputer_metadata,
                        )
                        
                        storage.record_split_activity(
                            pipeline_id=pipeline_id,
                            node_id=str(node_id),
                            transformer_name="imputer",
                            column_name=column,
                            split_name="train",
                            action="fit_transform",
                            row_count=int(train_mask.sum()) if train_mask is not None else 0,
                        )
                        
                        for split_name in ("test", "validation"):
                            rows_processed = int(split_counts.get(split_name, 0))
                            if rows_processed > 0:
                                storage.record_split_activity(
                                    pipeline_id=pipeline_id,
                                    node_id=str(node_id),
                                    transformer_name="imputer",
                                    column_name=column,
                                    split_name=split_name,
                                    action="transform",
                                    row_count=rows_processed,
                                )

            continue

        if KNNImputer is None or IterativeImputer is None:  # pragma: no cover - defensive guard
            message = "Imputation methods: scikit-learn components unavailable"
            signal.errors.append(message)
            _finalize_signal(signal, filled_cells, touched_columns, skipped_non_numeric, used_methods)
            return frame, message, signal

        numeric_subset = working_frame[target_columns].apply(pd.to_numeric, errors="coerce")
        valid_columns = [column for column in target_columns if not numeric_subset[column].dropna().empty]
        if not valid_columns:
            skipped_non_numeric.update(target_columns)
            continue

        subset = numeric_subset[valid_columns]
        if not subset.isna().any().any():
            continue

        # Configure the imputer based on method
        if method == "knn":
            options = entry.get("options") if isinstance(entry.get("options"), dict) else {}
            neighbors_value = options.get("neighbors", entry.get("neighbors"))
            available_rows = subset.dropna(how="all").shape[0]
            if available_rows == 0:
                continue
            max_neighbors = max(1, available_rows)
            neighbors = _coerce_positive_int(neighbors_value, default=5, minimum=1, maximum=max_neighbors)
            imputer = KNNImputer(n_neighbors=neighbors)
            strategy_signal.parameters.setdefault("neighbors", neighbors)
        else:
            options = entry.get("options") if isinstance(entry.get("options"), dict) else {}
            max_iter_value = options.get("max_iter", options.get("maxIter", entry.get("max_iter")))
            max_iter = _coerce_positive_int(max_iter_value, default=10, minimum=1, maximum=100)
            sample_posterior = method == "mice"
            imputer = IterativeImputer(
                max_iter=max_iter,
                random_state=0,
                sample_posterior=sample_posterior,
                skip_complete=True,
            )
            strategy_signal.parameters.setdefault("max_iter", max_iter)
            if method == "mice":
                strategy_signal.parameters.setdefault("sample_posterior", True)

        # Split-aware: fit on train, transform on all
        try:
            if has_splits and train_mask is not None:
                # Fit on training data only
                train_subset = subset[train_mask]
                if train_subset.empty or not train_subset.isna().any().any():
                    continue
                imputer.fit(train_subset)
                
                # Transform all data (train, test, validation)
                imputed_array = imputer.transform(subset)
            else:
                # No splits: standard fit_transform
                imputed_array = imputer.fit_transform(subset)
        except Exception:  # pragma: no cover - defensive fallback
            continue

        imputed_frame = pd.DataFrame(imputed_array, columns=valid_columns, index=subset.index)

        original_missing_counts = {
            column: int(working_frame[column].isna().sum())
            for column in valid_columns
        }
        strategy_filled = 0
        for column in valid_columns:
            original_series = working_frame[column]
            replacement_series = imputed_frame[column]
            mask = original_series.isna() & replacement_series.notna()
            if not mask.any():
                continue
            newly_filled = int(mask.sum())
            strategy_filled += newly_filled
            filled_cells += newly_filled
            touched_columns.add(column)
            try:
                working_frame.loc[mask, column] = replacement_series.loc[mask]
            except TypeError as exc:
                bool_message = _describe_bool_imputation_error(column, method, exc)
                if bool_message:
                    signal.errors.append(bool_message)
                    _finalize_signal(signal, filled_cells, touched_columns, skipped_non_numeric, used_methods)
                    return frame, bool_message, signal
                cast_message = _describe_integer_cast_error(column, method, exc)
                if cast_message:
                    signal.errors.append(cast_message)
                    _finalize_signal(signal, filled_cells, touched_columns, skipped_non_numeric, used_methods)
                    return frame, cast_message, signal
                raise
            remaining_missing = int(working_frame[column].isna().sum())
            signal.applied_columns.append(
                ImputationAppliedColumnSignal(
                    column=column,
                    method=method_name,
                    method_label=method_labels.get(method_name),
                    filled_cells=newly_filled,
                    original_missing=original_missing_counts.get(column, newly_filled),
                    remaining_missing=remaining_missing,
                    dtype=str(original_series.dtype),
                )
            )
            
            # Track split activity for sklearn methods
            if storage is not None and has_splits:
                # Store the imputer with metadata for this column
                imputer_metadata = {
                    "method": method,
                    "method_label": method_labels.get(method_name, method),
                    "train_rows": int(train_mask.sum()) if train_mask is not None else 0,
                }
                if method == "knn":
                    imputer_metadata["neighbors"] = neighbors
                elif method in {"mice", "regression"}:
                    imputer_metadata["max_iter"] = max_iter
                    if method == "mice":
                        imputer_metadata["sample_posterior"] = True
                
                storage.store_transformer(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="imputer",
                    transformer=imputer,
                    column_name=column,
                    metadata=imputer_metadata,
                )
                
                storage.record_split_activity(
                    pipeline_id=pipeline_id,
                    node_id=str(node_id),
                    transformer_name="imputer",
                    column_name=column,
                    split_name="train",
                    action="fit_transform",
                    row_count=int(train_mask.sum()) if train_mask is not None else 0,
                )
                
                for split_name in ("test", "validation"):
                    rows_processed = int(split_counts.get(split_name, 0))
                    if rows_processed > 0:
                        storage.record_split_activity(
                            pipeline_id=pipeline_id,
                            node_id=str(node_id),
                            transformer_name="imputer",
                            column_name=column,
                            split_name=split_name,
                            action="transform",
                            row_count=rows_processed,
                        )

        if strategy_filled > 0:
            used_methods[method] += 1

    _finalize_signal(signal, filled_cells, touched_columns, skipped_non_numeric, used_methods)

    if not filled_cells:
        summary = "Imputation methods: no missing values replaced"
        if skipped_non_numeric:
            summary = f"{summary}; skipped {len(skipped_non_numeric)} non-numeric column(s)"
    else:
        if used_methods:
            readable_methods = sorted(set(used_methods.keys()))
            method_list = ", ".join(method_labels.get(cast(ImputationMethodName, name), name) for name in readable_methods)
        else:
            method_list = "configured"
        summary_parts = [
            f"Imputation methods: filled {filled_cells} cell(s) "
            f"across {len(touched_columns)} column(s) using {method_list}"
        ]
        if skipped_non_numeric:
            summary_parts.append(f"skipped {len(skipped_non_numeric)} non-numeric column(s)")
        summary = "; ".join(summary_parts)

    return working_frame, summary, signal
