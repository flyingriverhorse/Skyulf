"""Preserve temporal keys for fold-local features, never for model fitting."""

from typing import Any


def history_preprocessor(preprocessing: Any) -> Any:
    """Unwrap the existing audit adapter without coupling tuning to concrete adapters."""
    return getattr(preprocessing, "inner", preprocessing)


def has_temporal_history(preprocessing: Any) -> bool:
    """Recognize preprocessing chains needing fold-local causal context."""
    return bool(getattr(history_preprocessor(preprocessing), "temporal_history_columns", ()))


def uses_history_policy(config: Any, preprocessing: Any) -> bool:
    """Apply fold policy only when cross-validation is enabled."""
    return config.cv_enabled and has_temporal_history(preprocessing)


def retain_history_metadata(preprocessing: Any, method: str, column: str | None) -> bool:
    """Require chronological folds and retain the clock only through feature engineering."""
    adapter = history_preprocessor(preprocessing)
    required = getattr(adapter, "temporal_history_columns", ())
    if not required:
        return False
    if method != "time_series_split":
        raise ValueError("carry history requires time_series_split, including inside nested CV.")
    if column not in required:
        return False
    adapter.retain_split_metadata(column)
    return True
