"""Aligned split metadata and explicit temporal/entity fold boundaries."""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from .history_policy import retain_history_metadata
from .schemas import TuningConfig

GROUP_METHODS = {"group_k_fold", "stratified_group_k_fold"}
POLICY_METHODS = GROUP_METHODS | {"time_series_split"}


def effective_cv_type(config: TuningConfig, problem_type: str) -> str:
    """Resolve nested auto without changing the established task-specific defaults."""
    if config.cv_type != "nested_cv":
        return config.cv_type
    if config.cv_nested_type == "auto":
        return "stratified_k_fold" if problem_type == "classification" else "k_fold"
    allowed = POLICY_METHODS | {"k_fold", "stratified_k_fold"}
    if config.cv_nested_type not in allowed:
        raise ValueError("Unsupported cv_nested_type split policy.")
    return config.cv_nested_type


def uses_explicit_policy(config: TuningConfig) -> bool:
    """Identify requests requiring aligned metadata and strict split validation."""
    return config.cv_enabled and (
        config.cv_type in GROUP_METHODS
        or config.cv_type == "nested_cv"
        and config.cv_nested_type != "auto"
        or config.cv_type == "time_series_split"
        and any((config.cv_gap, config.cv_test_size, config.cv_max_train_size))
    )


def take_rows(data: Any, positions: Any) -> Any:
    """Slice by position while retaining named DataFrame inputs for preprocessing."""
    if hasattr(data, "iloc"):
        return data.iloc[positions]
    if isinstance(data, (list, tuple)):
        return np.asarray(data)[positions]
    return data[positions]


def _column(X: Any, name: str | None, kind: str) -> np.ndarray:
    """Require explicit nonnull metadata instead of guessing feature columns."""
    if not name or name not in getattr(X, "columns", ()):
        raise ValueError(f"{kind} CV requires its explicit column in the raw training data.")
    values = np.asarray(X[name])
    if pd.isna(values).any():
        raise ValueError(f"{kind} CV metadata cannot contain missing values.")
    return values


def _times(X: Any, column: str | None) -> np.ndarray:
    """Normalize event timestamps to comparable UTC nanoseconds without guessing epoch units."""
    values = _column(X, column, "time")
    if pd.api.types.is_numeric_dtype(values.dtype):
        raise ValueError("time CV requires timestamps, not ambiguous numeric epoch units.")
    try:
        normalized = pd.Series(pd.to_datetime(values, utc=True, errors="raise"))
        return normalized.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("time CV requires valid timestamps.") from exc


def _drop_column(X: Any, column: str | None) -> Any:
    """Remove metadata before any learned preprocessing or estimator fitting."""
    if column is None:
        raise ValueError("Split metadata requires an explicit column.")
    return X.drop(columns=[column]) if isinstance(X, pd.DataFrame) else X.drop(column)


def prepare_policy_data(
    X: Any,
    y: Any,
    config: TuningConfig,
    problem_type: str,
    preprocessing: Any = None,
    *,
    return_positions: bool = False,
) -> tuple:
    """Separate split metadata, stably align chronological rows, and retain raw labels."""
    method = effective_cv_type(config, problem_type)
    validate_policy(config, problem_type)
    retain_time = retain_history_metadata(preprocessing, method, config.cv_time_column)
    if len(X) != len(y):
        raise ValueError("CV features and labels must have identical row counts.")
    if method == "time_series_split":
        times = _times(X, config.cv_time_column)
        order = np.argsort(times, kind="stable")
        features = X if retain_time else _drop_column(X, config.cv_time_column)
        result = (take_rows(features, order), take_rows(y, order), {"times": times[order]})
        return (*result, order) if return_positions else result
    if method in GROUP_METHODS:
        groups = _column(X, config.cv_group_column, "group")
        result = (_drop_column(X, config.cv_group_column), y, {"groups": groups})
        return (*result, np.arange(len(X))) if return_positions else result
    return (X, y, {}, np.arange(len(X))) if return_positions else (X, y, {})


def prediction_features(X: Any, result: Any) -> Any:
    """Apply the persisted metadata exclusion without reordering prediction requests."""
    for column in getattr(result, "excluded_feature_columns", ()):
        if column in getattr(X, "columns", ()):
            X = _drop_column(X, column)
    return X


def _positive_optional(value: Any, field: str) -> None:
    """Reject bools and nonintegral row-count window settings."""
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{field} must be a positive integer or null.")


def validate_policy(config: TuningConfig, problem_type: str) -> None:
    """Validate the effective policy before constructing or fitting any folds."""
    method = effective_cv_type(config, problem_type)
    if type(config.cv_folds) is not int or config.cv_folds < 2:
        raise ValueError("CV fold count must be an integer of at least two.")
    if type(config.cv_gap) is not int or config.cv_gap < 0:
        raise ValueError("cv_gap must be a nonnegative integer.")
    _positive_optional(config.cv_test_size, "cv_test_size")
    _positive_optional(config.cv_max_train_size, "cv_max_train_size")
    if method == "time_series_split" and config.cv_shuffle:
        raise ValueError("Temporal CV requires cv_shuffle=false.")
    if (
        method in {"stratified_group_k_fold", "stratified_k_fold"}
        and problem_type != "classification"
    ):
        raise ValueError("Stratified CV requires classification.")
    _validate_window_policy(config, method)


def _validate_window_policy(config: TuningConfig, method: str) -> None:
    """Prevent silently ignored temporal settings on non-temporal splitters."""
    if method != "time_series_split" and any(
        (config.cv_gap, config.cv_test_size, config.cv_max_train_size)
    ):
        raise ValueError("CV gap/window settings require time_series_split.")


def _sklearn_splitter(config: TuningConfig, problem_type: str) -> Any:
    """Construct the exact requested splitter without method fallbacks."""
    method = effective_cv_type(config, problem_type)
    seed = config.cv_random_state if config.cv_shuffle else None
    if method == "time_series_split":
        return TimeSeriesSplit(
            config.cv_folds,
            gap=config.cv_gap,
            test_size=config.cv_test_size,
            max_train_size=config.cv_max_train_size,
        )
    if method == "group_k_fold":
        if not config.cv_shuffle:
            return GroupKFold(config.cv_folds)
        return _ShuffledGroupSplit(config.cv_folds, seed)
    if method == "stratified_group_k_fold":
        return StratifiedGroupKFold(config.cv_folds, shuffle=config.cv_shuffle, random_state=seed)
    cls = StratifiedKFold if method == "stratified_k_fold" else KFold
    return cls(config.cv_folds, shuffle=config.cv_shuffle, random_state=seed)


class _ShuffledGroupSplit:
    """Shuffle whole groups consistently on supported sklearn versions including 1.4."""

    def __init__(self, folds: int, seed: int | None) -> None:
        """Keep randomization attached to group identities rather than individual rows."""
        self.folds = folds
        self.seed = seed

    def split(self, X: Any, y: Any, groups: Any) -> Any:
        """Assign shuffled group sets to validation without splitting an entity."""
        unique = np.unique(groups)
        shuffled = np.random.RandomState(self.seed).permutation(unique)
        if len(shuffled) < self.folds:
            raise ValueError("Group CV has fewer groups than folds.")
        for values in np.array_split(shuffled, self.folds):
            mask = np.isin(groups, values)
            yield np.flatnonzero(~mask), np.flatnonzero(mask)


def _membership_digest(values: Any) -> str:
    """Hash bounded membership descriptions without storing raw customer identifiers."""
    encoded = repr(sorted({(type(v).__name__, repr(v)) for v in values})).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fold_evidence(train: Any, test: Any, metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    """Verify a partition and record counts and chronological boundaries."""
    evidence: dict[str, Any] = {"train_rows": len(train), "test_rows": len(test)}
    if "times" in metadata:
        times = metadata["times"]
        if times[train].max() >= times[test].min():
            raise ValueError(
                "A timestamp crosses a time CV fold boundary; aggregate tied events or change folds."
            )
        evidence |= {
            "train_start": int(times[train].min()),
            "train_end": int(times[train].max()),
            "test_start": int(times[test].min()),
            "test_end": int(times[test].max()),
        }
    if "groups" in metadata:
        groups = metadata["groups"]
        a, b = set(groups[train]), set(groups[test])
        if not a.isdisjoint(b):
            raise ValueError("Group CV requires disjoint training and validation groups.")
        evidence |= {
            "train_groups": len(a),
            "test_groups": len(b),
            "train_groups_sha256": _membership_digest(a),
            "test_groups_sha256": _membership_digest(b),
        }
    return evidence


def validate_class_membership(y: np.ndarray, train: Any, test: Any, problem_type: str) -> None:
    """Require complete class coverage before any candidate can win on partial evidence."""
    if problem_type != "classification":
        return
    expected = set(y)
    if len(expected) < 2 or set(y[train]) != expected or set(y[test]) != expected:
        raise ValueError(
            "CV requires every class in each training and validation fold; change folds or add history/groups."
        )


@dataclass
class FrozenSplit:
    """Replay verified positional folds identically across all search strategies."""

    partitions: list[tuple[np.ndarray, np.ndarray]]
    evidence: list[dict[str, Any]]

    def split(self, X: Any = None, y: Any = None, groups: Any = None) -> Any:
        """Yield independent index arrays for sklearn-compatible consumers."""
        for train, test in self.partitions:
            yield train.copy(), test.copy()

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Expose the prevalidated number of folds to searcher implementations."""
        return len(self.partitions)


def policy_splitter(
    config: TuningConfig, problem_type: str, y: Any, metadata: dict[str, np.ndarray]
) -> FrozenSplit:
    """Validate all fold memberships before starting a potentially expensive search."""
    validate_policy(config, problem_type)
    labels = np.asarray(y)
    groups = metadata.get("groups")
    if groups is not None:
        groups = pd.factorize(groups, sort=False)[0]
    cv = _sklearn_splitter(config, problem_type)
    parts = list(cv.split(np.arange(len(labels)), labels, groups))
    evidence = []
    for train, test in parts:
        if len(train) < 2 or len(test) < 2:
            raise ValueError("Each CV fold requires at least two training and validation rows.")
        validate_class_membership(labels, train, test, problem_type)
        evidence.append(fold_evidence(train, test, metadata))
    return FrozenSplit(parts, evidence)


def policy_description(config: TuningConfig, problem_type: str) -> dict[str, Any]:
    """Describe explicit policy defaults with JSON-safe scalar values."""
    return {
        "method": effective_cv_type(config, problem_type),
        "time_column": config.cv_time_column,
        "group_column": config.cv_group_column,
        "gap": config.cv_gap,
        "test_size": config.cv_test_size,
        "max_train_size": config.cv_max_train_size,
        "shuffle": config.cv_shuffle,
        "random_state": config.cv_random_state,
    }


def validate_holdout_metadata(
    train_X: Any, holdout_X: Any, config: TuningConfig, problem_type: str
) -> None:
    """Reject group overlap or reversed chronology in an explicitly reserved partition."""
    if not config.cv_enabled or holdout_X is None or len(holdout_X) == 0:
        return
    method = effective_cv_type(config, problem_type)
    if method in GROUP_METHODS:
        train = set(_column(train_X, config.cv_group_column, "group"))
        held = set(_column(holdout_X, config.cv_group_column, "group"))
        if not train.isdisjoint(held):
            raise ValueError("Reserved holdout must not share groups with training rows.")
    if (
        method == "time_series_split"
        and config.cv_type == "nested_cv"
        and _times(train_X, config.cv_time_column).max()
        >= _times(holdout_X, config.cv_time_column).min()
    ):
        raise ValueError("Reserved temporal holdout must follow all training timestamps.")
