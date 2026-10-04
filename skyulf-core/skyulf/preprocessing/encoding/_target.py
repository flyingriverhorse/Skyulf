"""Route an embedded target through the same encoder as an explicit target Series."""

from typing import Any

import polars as pl

from ...engines.polars_engine import SkyulfPolarsWrapper
from ..dispatcher import apply_dual_engine, fit_dual_engine


def fit_target_encoder(X: Any, y: Any, config: dict, methods: dict) -> Any:
    """Separate a selected embedded target and remember how to restore its column."""
    target = config.get("target_column")
    columns = config.get("columns")
    embedded = y is None and target in X.columns and (not columns or target in columns)
    if embedded:
        y = X[target]
        X = X.drop(target) if isinstance(X, pl.DataFrame) else X.drop(columns=target)
    artifact = fit_dual_engine((X, y) if y is not None else X, config, methods)
    if embedded and "__target__" in artifact.get("encoders", {}):
        artifact = {**artifact, "target_column": target}
    return artifact


def apply_target_encoder(X: Any, y: Any, params: dict, methods: dict) -> Any:
    """Restore encoded targets only when the input actually contains that target column."""
    target = params.get("target_column")
    embedded = y is None and target in X.columns
    columns = list(X.columns)
    if embedded:
        y = X[target]
        X = X.drop(target) if isinstance(X, pl.DataFrame) else X.drop(columns=target)
    result = apply_dual_engine((X, y) if y is not None else X, params, methods)
    if not embedded:
        return result
    features, labels = result
    if isinstance(features, SkyulfPolarsWrapper):
        return features.with_column(target, labels).select(columns)
    if isinstance(features, pl.DataFrame):
        return features.with_columns(pl.Series(target, labels)).select(columns)
    features[target] = labels
    return features.loc[:, columns]
